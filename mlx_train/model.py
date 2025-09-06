import os
import json

import mlx.nn as nn
import mlx.core as mx

from mlx.utils import tree_flatten
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
# from mlx_lm.tuner.utils import linear_to_lora_layers, get_lora_keys
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.utils import load_config, load_model as mlx_load_model

from mlx_train.utils import build_model_path
import mlx_train.distributed as dist
from mlx_train.ppp import PipelineSlice

def load_model(model_config: dict) -> tuple[nn.Module, TokenizerWrapper]:
    model_path = build_model_path(model_config['repo_id'])

    model, config = mlx_load_model(model_path, lazy=False, strict=False)
    model_config['hf_config'] = config

    tokenizer = load_tokenizer(model_path)
    assert isinstance(tokenizer, TokenizerWrapper)

    return model, tokenizer

def lora_model(model: nn.Module, lora_config) -> nn.Module:
    model.freeze()
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
    model, tokenizer = load_model(model_config)

    if 'lora' in model_config:
        lora_model(model, model_config['lora'])

    apply_gradient_checkpointing(model, model_config)

    all_params = tree_flatten(model.parameters())
    trainable_params = tree_flatten(model.trainable_parameters())
    # print([x[0] for x in all_params])
    # print([x[0] for x in trainable_params])
    
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
                # "num_layers": model_config['lora'],
                # "rank": 16,
                # "alpha": 16,
                # "scale": 1.0,
                # "dropout": 0.0
            },
        }
        adapter_config['lora_parameters'].update(model_config['lora'])
        
        with open(os.path.join(output_path, 'adapter_config.json'), 'w') as f:
            json.dump(adapter_config, f, indent=2)

    else:
        raise NotImplementedError()

