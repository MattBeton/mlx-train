from ctypes import cast
import mlx.nn as nn

from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
from mlx_lm.utils import load_model as mlx_load_model

from mlx_train.utils import build_model_path

def load_model(model_config: dict) -> tuple[nn.Module, TokenizerWrapper]:
    model_path = build_model_path(model_config['repo_id'])

    model, config = mlx_load_model(model_path, lazy=False, strict=False)
    model_config['hf_config'] = config

    tokenizer = load_tokenizer(model_path)
    assert isinstance(tokenizer, TokenizerWrapper)

    return model, tokenizer