import os
import json

import mlx.nn as nn
import mlx.core as mx

from typing import Dict, List, Tuple, Optional

from mlx.utils import tree_flatten
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
from mlx_lm.tuner.utils import linear_to_lora_layers, get_lora_keys
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.utils import load_model as mlx_load_model

from mlx_train.utils import build_model_path
import mlx_train.distributed as dist
from mlx_train.ppp import PipelineSlice, _inner_model

def load_model(model_config: dict) -> tuple[nn.Module, TokenizerWrapper]:
    model_path = build_model_path(model_config['repo_id'])

    model, config = mlx_load_model(model_path, lazy=True, strict=False)
    model_config['hf_config'] = config

    tokenizer = load_tokenizer(model_path)
    assert isinstance(tokenizer, TokenizerWrapper)

    return model, tokenizer

def lora_model(model: nn.Module, lora_config) -> nn.Module:
    linear_to_lora_layers(model, lora_config['num_layers'], lora_config, lora_config.get('use_dora', False))

def apply_gradient_checkpointing(model, model_config: dict) -> None:
    if 'grad_checkpoint' not in model_config:
        return

    dist.rprint(f'applying {model_config["grad_checkpoint"]} checkpointing')
    if model_config['grad_checkpoint'] == 'medium':
        grad_checkpoint(model.model.layers[0].self_attn)
        grad_checkpoint(model.model.layers[0].mlp)
    elif model_config['grad_checkpoint'] == 'max':
        grad_checkpoint(model.model.layers[0])
    elif model_config['grad_checkpoint'] != 'none':
        raise ValueError(f"Invalid gradient_checkpoint option: {model_config['grad_checkpoint']}. Must be 'none', 'medium', or 'max'")

def load_configure_model(model_config: dict):
    dist.rprint('loading model...', all=True)
    model, tokenizer = load_model(model_config)
    model.freeze()

    if 'lora' in model_config:
        dist.rprint('applying lora', all=True)
        lora_model(model, model_config['lora'])

    dist.rprint('applying gradient checkpointing', all=True)
    apply_gradient_checkpointing(model, model_config)

    all_params = tree_flatten(model.parameters())
    trainable_params = tree_flatten(model.trainable_parameters())

    dist.rprint('applying autoparallelization', all=True)
    # Apply auto-parallel if specified in config
    if 'auto_parallel' in model_config:
        auto_parallel_config = model_config['auto_parallel']
        if auto_parallel_config.get('enabled', False):
            total_layers = len(getattr(model.model, 'layers', getattr(model.model, 'h', [])))
            layers_per_rank = total_layers // dist.size
            
            start_layer = dist.rank * layers_per_rank
            end_layer = (dist.rank + 1) * layers_per_rank if dist.rank < dist.size - 1 else total_layers

            # TODO: Allow overrides from config if provided
            # start_layer = auto_parallel_config.get('start_layer', start_layer)
            # end_layer = auto_parallel_config.get('end_layer', end_layer)

            model = PipelineSlice(model, start_layer, end_layer)

    dist.barrier()
    mx.eval(model)
    dist.barrier()

    # Calculate total and trainable parameters
    all_params = tree_flatten(model.parameters())
    trainable_params = tree_flatten(model.trainable_parameters())
    # print([x[0] for x in all_params])
    # print([x[0] for x in trainable_params])
   
    total_params = sum(p.size for _, p in all_params) # type: ignore
    total_trainable = sum(p.size for _, p in trainable_params) # type: ignore
    
    # Calculate memory usage in MB
    total_memory = sum(p.nbytes for _, p in all_params) / (1024 * 1024) # type: ignore
    
    dist.rprint(f'Model loaded - Total params: {total_params:,} | Trainable params: {total_trainable:,} | Memory: {total_memory:.2f} MB', all=True)

    return model, tokenizer


### Vibe Coded from here downwards.

def _layers_total(inner_model: nn.Module) -> int:
    return len(getattr(inner_model, "layers", getattr(inner_model, "h", [])))

def _prefix_to_inner(full_model: nn.Module) -> str:
    if hasattr(full_model, "model"):
        return "model"
    if hasattr(full_model, "transformer"):
        return "transformer"
    raise ValueError("Expected full model to have either `.model` or `.transformer`")

def _rank_bounds(total_layers: int, world_size: int, rank: int) -> Tuple[int, int]:
    per = total_layers // world_size
    start = rank * per
    end = (rank + 1) * per if rank < world_size - 1 else total_layers
    return start, end

def _detect_sep_from_module(m: nn.Module) -> str:
    """Return '.' or '/' depending on how MLX flattens parameter paths."""
    items = tree_flatten(m.trainable_parameters())
    for k, _ in items:
        if "/" in k:
            return "/"
        if "." in k:
            return "."
    return "/"  # sensible default

def _split_any_sep(key: str) -> List[str]:
    """Split key by '/' or '.' separator."""
    return key.split("/") if "/" in key else key.split(".")

def _find_layer_idx(parts: List[str]) -> Tuple[int, int]:
    """
    Find ('layers' or 'h', idx) in parts and return (pos, idx).
    Raises ValueError if not found.
    """
    for i, p in enumerate(parts):
        if p in ("layers", "h"):
            # defensive: only accept int-ish next segment
            if i + 1 < len(parts):
                try:
                    return i, int(parts[i + 1])
                except ValueError:
                    pass
    raise ValueError(f"Could not find layer index in key parts: {parts}")

def _relkey_to_id(rel_key: str, start: int) -> Tuple[int, str]:
    """
    Convert a slice-local key (e.g., 'layers.0.self_attn.q_proj.lora_b')
    into a canonical ID: (global_layer_idx, 'self_attn.q_proj.lora_b').
    """
    parts = _split_any_sep(rel_key)
    pos, local_idx = _find_layer_idx(parts)
    tail = ".".join(parts[pos + 2 :])  # after the integer index
    return start + local_idx, tail

def _refkey_to_id(ref_key: str) -> Tuple[int, str]:
    """
    Convert a full ref-model key (e.g., 'model.layers.12.self_attn.q_proj.lora_b')
    into the same canonical ID tuple.
    """
    parts = _split_any_sep(ref_key)
    pos, global_idx = _find_layer_idx(parts)
    tail = ".".join(parts[pos + 2 :])
    return global_idx, tail

def _canon_rel(key: str) -> str:
    """Canonicalize a relative (slice-local) key to dot-separated for matching."""
    return key.replace("/", ".")

def _join(parts: List[str], sep: str) -> str:
    return sep.join(parts)

def _rekey_slice_to_global(rel_key: str, start: int, prefix: str, out_sep: str) -> str:
    """
    Convert a PipelineSlice-local key ('.' or '/' delimited) to a full-model key,
    using the ref model's separator (out_sep).
    """
    parts = _split_any_sep(rel_key)
    if parts[0] == "layers":
        local_idx = int(parts[1])
        parts[1] = str(start + local_idx)
        return f"{prefix}{out_sep}" + _join(parts, out_sep)
    if parts[0] == "tok_embeddings":
        return f"{prefix}{out_sep}embed_tokens{out_sep}" + _join(parts[1:], out_sep)
    if parts[0] == "norm":
        return f"{prefix}{out_sep}norm{out_sep}" + _join(parts[1:], out_sep)
    # Default: treat as inner-model child (q_proj, v_proj, mlp, etc.)
    return f"{prefix}{out_sep}" + _join(parts, out_sep)

def _sorted_rel_items(m: nn.Module) -> List[Tuple[str, mx.array]]:
    items = tree_flatten(m.trainable_parameters())
    items.sort(key=lambda kv: kv[0])
    return items


def write_adapters_distributed(local_slice: PipelineSlice, model_config: dict, output_filepath: Optional[str] = None) -> None:
    """
    Ring-gather LoRA tensors (neighbor-only) and write adapters.safetensors directly
    using the reference model's official key names. No Module.update() calls.
    """
    if output_filepath:
        output_dir = model_config["output_location"] + output_filepath
    else:
        output_dir = model_config["output_location"] 
    os.makedirs(output_dir, exist_ok=True)

    dist.rprint(f'writing to file {output_dir}')

    # 1) Reference model (lazy) only to discover official LoRA key names
    full_model_path = build_model_path(model_config["repo_id"])
    ref_model, _cfg = mlx_load_model(full_model_path, lazy=True, strict=False)
    if "lora" in model_config:
        lora_model(ref_model, model_config["lora"])
    inner = _inner_model(ref_model)
    total_layers = _layers_total(inner)

    # 2) Plan: which canonical IDs (layer_idx, tail) belong to each rank
    ids_by_rank: Dict[int, List[Tuple[int, str]]] = {}
    meta_by_id: Dict[Tuple[int, str], Tuple[Tuple[int, ...], mx.Dtype]] = {}

    for r in range(dist.size):
        s, e = _rank_bounds(total_layers, dist.size, r)
        shadow = PipelineSlice(ref_model, s, e)
        rel_items = _sorted_rel_items(shadow)
        rel_items = [(k, v) for (k, v) in rel_items if "lora_" in k]
        ids: List[Tuple[int, str]] = []
        for rk, arr in rel_items:
            cid = _relkey_to_id(rk, s)  # (global_layer_idx, tail)
            ids.append(cid)
            if cid not in meta_by_id:
                meta_by_id[cid] = (arr.shape, arr.dtype)
        ids_by_rank[r] = ids

    assert any(ids_by_rank.values()), "No LoRA tensors discovered on reference model."

    # 3) Build my local map: ID -> array (cast to ref shape/dtype)
    my_start, my_end = _rank_bounds(total_layers, dist.size, dist.rank)
    my_rel_items = _sorted_rel_items(local_slice)
    my_rel_items = [(k, v) for (k, v) in my_rel_items if "lora_" in k]

    my_map_by_id: Dict[Tuple[int, str], mx.array] = {}
    for rk, arr in my_rel_items:
        cid = _relkey_to_id(rk, my_start)
        shape, dtype = meta_by_id[cid]
        if arr.shape != shape or arr.dtype != dtype:
            arr = arr.astype(dtype).reshape(shape)
        my_map_by_id[cid] = arr

    # Optional: visibility of what this rank owns
    # dist.rprint([rk for (rk, _) in my_rel_items], all=True)

    # 4) Ring gather (neighbor-only) by canonical ID
    agg_by_id: Dict[Tuple[int, str], mx.array] = {}
    if dist.size == 1:
        agg_by_id.update(my_map_by_id)

    def id_order_for_src(src: int) -> List[Tuple[int, str]]:
        order: List[Tuple[int, str]] = []
        for q in range(dist.size - 1, src - 1, -1):
            order.extend(ids_by_rank[q])
        return order

    for src in range(dist.size - 1, 0, -1):
        dest = src - 1
        order = id_order_for_src(src)

        if dist.rank == src:
            # ensure my params are included before sending
            for cid in ids_by_rank[src]:
                if cid not in agg_by_id:
                    agg_by_id[cid] = my_map_by_id[cid]
            send_tokens = []
            for cid in order:
                tok = mx.distributed.send(agg_by_id[cid], dst=dest)
                send_tokens.append(tok)
            if send_tokens:
                mx.eval(*send_tokens)

        elif dist.rank == dest:
            recv_bufs = []
            for cid in order:
                shape, dtype = meta_by_id[cid]
                v = mx.distributed.recv(shape=shape, dtype=dtype, src=src)
                agg_by_id[cid] = v
                recv_bufs.append(v)
            if recv_bufs:
                mx.eval(*recv_bufs)

            # merge my own so I become the next src
            for cid in ids_by_rank[dest]:
                if cid not in agg_by_id:
                    agg_by_id[cid] = my_map_by_id[cid]

        dist.barrier()

    # 5) Rank 0: assemble adapters.safetensors directly (no update())
    if dist.rank == 0:
        ref_train = dict(tree_flatten(ref_model.trainable_parameters()))
        ref_lora_keys = [k for k in ref_train.keys() if "lora_" in k]

        adapter_weights: Dict[str, mx.array] = {}
        missing: List[str] = []

        for k in ref_lora_keys:
            cid = _refkey_to_id(k)        # (global_layer_idx, tail)
            arr = agg_by_id.get(cid, None)
            if arr is None:
                missing.append(k)
                continue
            shape, dtype = meta_by_id[cid]
            if arr.shape != shape or arr.dtype != dtype:
                arr = arr.astype(dtype).reshape(shape)
            adapter_weights[k] = arr

        # If you see this assert, your LoRA injection targets don't match training slices.
        assert not missing, f"Missing {len(missing)} LoRA tensors when assembling adapters: e.g. {missing[:4]}"

        mx.save_safetensors(os.path.join(output_dir, "adapters.safetensors"), adapter_weights)

        adapter_config = {
            "fine_tune_type": "lora",
            "num_layers": model_config["lora"]["num_layers"],
            "lora_parameters": {
                "keys": list(get_lora_keys(ref_model)),  # exact key names in saved file
                **model_config["lora"],
            },
        }
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f, indent=2)

        dist.rprint(f"✓ Saved LoRA adapters to {output_dir}", only=0)

    dist.barrier()


def write_model(model, model_config):
    output_path = model_config['output_location']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if 'lora' in model_config:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(model_config['output_location'] + 'adapters.safetensors'), adapter_weights)

        adapter_config = {
            "fine_tune_type": "lora",
            'num_layers': model_config['lora']['num_layers'],
            "lora_parameters": {
                "keys": list(get_lora_keys(model)),
            },
        }
        adapter_config['lora_parameters'].update(model_config['lora'])
        
        with open(os.path.join(output_path, 'adapter_config.json'), 'w') as f:
            json.dump(adapter_config, f, indent=2)

    else:
        raise NotImplementedError()

