from pydantic import DirectoryPath

import mlx.core as mx
from mlx.nn import Linear
from mlx.utils import tree_map_with_path

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
