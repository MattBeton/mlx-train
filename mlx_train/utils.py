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
    (y_s, tok_y, dy_s, tok_dx) = tokens
    if y_s is not None: named['y_s'] = y_s
    if tok_y is not None: named['tok_y'] = tok_y
    if dy_s is not None: named['dy_s'] = dy_s
    if tok_dx is not None: named['tok_dx'] = tok_dx
    named['loss'] = loss
    named['x'] = x
    named['y'] = y
    for layer in model.layers:
        if isinstance(layer, Linear):
            named['model_weight'] = layer.weight
    mx.export_to_dot('g_s.dot', **named)
