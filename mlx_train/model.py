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
from mlx_train.ppp import auto_parallel, IdentityLayer

def load_model(model_config: dict) -> tuple[nn.Module, TokenizerWrapper]:
    model_path = build_model_path(model_config['repo_id'])

    model, config = mlx_load_model(model_path, lazy=False, strict=False)
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
    model, tokenizer = load_model(model_config)

    if 'lora' in model_config:
        lora_model(model, model_config['lora'])

    apply_gradient_checkpointing(model, model_config)
    
    # Apply auto-parallel if specified in config
    if 'auto_parallel' in model_config:
        auto_parallel_config = model_config['auto_parallel']
        if auto_parallel_config.get('enabled', False):
            # Calculate layer range for this rank
            total_layers = len(getattr(model.model, 'layers', getattr(model.model, 'h', [])))
            layers_per_rank = total_layers // dist.size
            
            # Calculate start and end layers for this rank
            start_layer = dist.rank * layers_per_rank
            end_layer = (dist.rank + 1) * layers_per_rank if dist.rank < dist.size - 1 else total_layers

            model._ppp_start = start_layer
            model._ppp_end = end_layer

            inner = getattr(model, 'model', getattr(model, 'transformer', None))
            hidden_size = model_config['hf_config']['hidden_size']
            vocab_size = model_config['hf_config']['vocab_size']
            hidden_dtype = inner.tok_embeddings.weight.dtype

            model._ppp_hidden_size = hidden_size
            model._ppp_vocab_size = vocab_size
            model._ppp_hidden_dtype = hidden_dtype
            
            # Allow overrides from config if provided
            start_layer = auto_parallel_config.get('start_layer', start_layer)
            end_layer = auto_parallel_config.get('end_layer', end_layer)
            
            model = auto_parallel(model, start_layer, end_layer)

            # Set in_shape based on whether this is the first layer or not
            if dist.rank == 0:
                # First rank processes embeddings, so input is vocab_size (token IDs)
                model.in_shape = model_config['hf_config']['vocab_size']
            else:
                # Other ranks receive hidden states from previous layers
                model.in_shape = model_config['hf_config']['hidden_size']

    # Calculate total and trainable parameters
    all_params = tree_flatten(model.parameters())
    trainable_params = tree_flatten(model.trainable_parameters())
   
    total_params = sum(p.size for _, p in all_params) # type: ignore
    total_trainable = sum(p.size for _, p in trainable_params) # type: ignore
    
    # Calculate memory usage in MB
    total_memory = sum(p.nbytes for _, p in all_params) / (1024 * 1024) # type: ignore
    
    dist.rprint(f'Model loaded - Total params: {total_params:,} | Trainable params: {total_trainable:,} | Memory: {total_memory:.2f} MB')

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

