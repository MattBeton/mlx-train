import os
import json

import mlx.nn as nn
import mlx.core as mx

from typing import Dict, List, Tuple, Optional, override

from mlx.nn.layers.distributed import shard_linear, shard_inplace
from mlx.utils import tree_flatten
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
from mlx_lm.tuner.utils import linear_to_lora_layers, get_lora_keys
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.utils import load_model as mlx_load_model

from mlx_train.utils import build_model_path
import mlx_train.distributed as dist
from mlx_train.ppp import PipelineSlice, _inner_model

from mlx_lm.models.deepseek_v3 import DeepseekV3DecoderLayer, DeepseekV3MLP

class FakeGroup(mx.distributed.Group):
    def __init__(self):
        pass    

    @override
    def rank(self) -> int:
        return 0

    @override
    def size(self) -> int:
        return 4

class TPModel(nn.Module):
    def __init__(self, full_model: nn.Module, group: Optional[mx.distributed.Group] = None):
        super().__init__()

        # group = group or mx.distributed.init()
        group = FakeGroup()
        N = group.size()

        inner = _inner_model(full_model)

        for layer in inner.layers:
            print(layer)
            layer.self_attn.q_b_proj = shard_linear(
                layer.self_attn.q_b_proj, 'all-to-sharded', group=group
            )
            layer.self_attn.kv_b_proj = shard_linear(
                layer.self_attn.kv_b_proj, 'all-to-sharded', group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, 'sharded-to-all', group=group
            )

            layer.self_attn.num_heads //= N

            # Shard the MLP
            if isinstance(layer.mlp, DeepseekV3MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            # Shard the MoE. Shard in place since the MoE should be responsible
            # for aggregating the results.
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_experts.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_experts.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_experts.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )


        raise Exception('stop')
