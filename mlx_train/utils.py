from pydantic import DirectoryPath

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import Linear
from mlx.utils import tree_map_with_path, tree_flatten

from shared.const import EXO_HOME

def build_model_path(model_id: str) -> DirectoryPath:
    return EXO_HOME / "models" / model_id.replace("/", "--")

def export_graph(loss, g_s, tokens, x, y, model):
    named = {}
    tree_map_with_path(lambda path, a: named.__setitem__("grads/" + path, a), g_s)
    
    # Add tokens from dict to named
    for key, value in tokens.items():
        if value is not None:
            named[key] = value  # Keep original naming for graph
    
    named['loss'] = loss
    named['x'] = x
    named['y'] = y

    for layer in model.layers:
        if isinstance(layer, Linear):
            named['model_weight'] = layer.weight
    mx.export_to_dot('g_s.dot', **named)


def fmt_bytes(n):  # pretty printer
    for unit in ("B","KB","MB","GB","TB"):
        if n < 1024: return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def bytes_of_tree(tree) -> int:
    leaves = [x[1] for x in tree_flatten(tree)]
    return sum(getattr(x, "nbytes", 0) for x in leaves if isinstance(x, mx.array))

def bytes_of_module(mod: nn.Module) -> int:
    # parameters() returns all mx.arrays in the module that are parameters
    return bytes_of_tree(mod.parameters())

def bytes_of_optimizer(opt) -> int:
    # opt.state holds all optimizer state arrays (e.g., Adam m/v, etc.)
    return bytes_of_tree(opt.state)

