import os
import json

import mlx.nn as nn
import mlx.core as mx

from mlx.utils import tree_flatten
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
# from mlx_lm.tuner.utils import linear_to_lora_layers, get_lora_keys
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.utils import load_model as mlx_load_model

from mlx_train.utils import build_model_path
import mlx_train.distributed as dist

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
    model = load_model(model_config)

    model, tokenizer = load_model(model_config)

    if 'lora' in model_config:
        lora_model(model, model_config['lora'])

    apply_gradient_checkpointing(model, model_config)

    # Calculate total and trainable parameters
    all_params = tree_flatten(model.parameters())
    trainable_params = tree_flatten(model.trainable_parameters())
   
    total_params = sum(p.size for _, p in all_params) # type: ignore
    total_trainable = sum(p.size for _, p in trainable_params) # type: ignore
    
    # Calculate memory usage in MB
    total_memory = sum(p.nbytes for _, p in all_params) / (1024 * 1024) # type: ignore
    
    dist.rprint(f'Model loaded - Total params: {total_params:,} | Trainable params: {total_trainable:,} | Memory: {total_memory:.2f} MB', selective=True)

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

