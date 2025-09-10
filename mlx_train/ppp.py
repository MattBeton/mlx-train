from typing import Protocol, cast, override

import mlx.core as mx
import mlx.nn as nn  # pyright: ignore[reportMissingTypeStubs]

import mlx_train.distributed as dist

from mlx_train.train import masked_loss
from mlx_lm.models.base import create_attention_mask


class IdentityLayer(nn.Module):
    @override
    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        return x


class _LayerCallable(Protocol):
    """Structural type that any compatible layer must satisfy.

    We require a single positional input of type ``mx.array`` and an
    ``mx.array`` output, while permitting arbitrary *args / **kwargs so this
    protocol matches the vast majority of `mlx.nn.Module` subclasses.
    """

    weight: mx.array

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array: ...

def _depends_like(x: mx.array, dep: mx.array) -> mx.array:
    """Return x, but make it depend on dep (no numerical effect)."""
    z = mx.sum(mx.stop_gradient(dep))
    zero = mx.array(0, dtype=x.dtype)
    return x + zero * z # broadcasts scalar z

def _inner_model(m: nn.Module) -> nn.Module:
    inner = getattr(m, "model", None) or getattr(m, "transformer", None)
    if not isinstance(inner, nn.Module):
        raise ValueError("Expected a model with .model or .transformer")
    return inner

class PipelineSlice(nn.Module):
    """
    Wrap a loaded mlx-lm model and expose a stage-local forward.

    On the first rank, call with tokens:     __call__(tokens, mask=None)
    On other ranks, call with hidden states: __call__(None, input_embeddings=h, mask=None)

    If this slice includes the final layer, we apply norm + lm_head and return LOGITS.
    Otherwise we return hidden states (B, L, D).
    """
    def __init__(self, full_model: nn.Module, start: int, end: int):
        super().__init__()

        inner = _inner_model(full_model)

        self.model_type = getattr(full_model, "model_type", getattr(inner, "model_type", None))
        if self.model_type is None:
            self.model_type = 'llama'

        self.start = start
        self.end = end
        self.total = len(getattr(inner, "layers", getattr(inner, "h", [])))

        # Keep layers we need
        if start == 0:
            self.tok_embeddings = inner.embed_tokens

        self.layers = getattr(inner, "layers", getattr(inner, "h", []))[start:end]

        if end == self.total:
            self.norm = inner.norm
            self.lm_head = inner.embed_tokens.as_linear
            dist.rprint(f'{inner.embed_tokens.weight.shape=}', all=True)

        # static info for shape/dtype on this model
        self.hidden_size = int(inner.norm.weight.shape[0])
        self.vocab_size = inner.embed_tokens.weight.shape[0] if end == self.total else None

    def __call__(self,
                 tokens: mx.array | None,
                 mask: mx.array | None = None,
                 cache=None,
                 input_embeddings: mx.array | None = None) -> mx.array:
        # Produce h
        if tokens is not None:
            if not hasattr(self, "tok_embeddings"):
                raise ValueError("This slice doesn't own tok_embeddings; pass input_embeddings instead")
            h = self.tok_embeddings(tokens)
        else:
            assert input_embeddings is not None, "Either tokens or input_embeddings must be provided"
            h = input_embeddings

        if mask is None:
            mask = cast(mx.array, create_attention_mask(h, return_array=True))
            mask = mx.expand_dims(mask, axis=0)
            mask = mx.expand_dims(mask, axis=0)

        # Run only our layer range
        for layer in self.layers:
            h = layer(h, mask, cache=None)

        # Head only on the last slice
        if hasattr(self, "norm"):
            h = self.norm(h)
            return self.lm_head(h)
            # return self._lm_head_chunked(h)   # (B, L, V)
        else:
            return h                 # (B, L, D)

class PipelineFirstLayer(nn.Module):
    def __init__(self, original_layer: _LayerCallable, r: int, s: int):
        super().__init__()
        self.original_layer: _LayerCallable = original_layer
        self.r: int = r
        self.s: int = s

    @override
    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self.r != 0:
            x = mx.distributed.recv_like(x, (self.r - 1))
        return self.original_layer(x, *args, **kwargs)


class PipelineLastLayer(nn.Module):
    def __init__(self, original_layer: _LayerCallable, r: int, s: int):
        super().__init__()
        self.original_layer: _LayerCallable = original_layer
        self.r: int = r
        self.s: int = s

    @override
    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        output: mx.array = self.original_layer(x, *args, **kwargs)
        if self.r != self.s - 1:
            output = mx.distributed.send(output, (self.r + 1) % self.s)
        output = mx.distributed.all_gather(output)[-output.shape[0] :]  # pyright: ignore[reportUnknownMemberType]
        return output

def step_graph(model: PipelineSlice, tokens: mx.array, targets: mx.array, lengths: mx.array):
    """
    tokens  : (B, L) int32
    targets : (B, L) int32 (shifted by 1 in caller)
    lengths : (B, 2) label start/end (your masked_loss)
    """
    tokens_out = {}

    B, L = tokens.shape
    H = model.hidden_size
    activation_dtype = mx.bfloat16 # TODO: Should we be training LoRA at 16 or 32?

    y = None
    if dist.rank == 0:
        x = tokens
        y = model(tokens)
    elif dist.rank != dist.size - 1:
        dist.rprint(f'recv shape {(B, L, H)}, dtype {activation_dtype}', all=True)
        x = mx.distributed.recv(shape=(B, L, H), dtype=activation_dtype, src=dist.rank - 1)
        tokens_out['x'] = x
        y = model(None, input_embeddings=x)
    else: # last rank
        dist.rprint(f'recv shape {(B, L, H)}, dtype {activation_dtype}', all=True)
        x = mx.distributed.recv(shape=(B, L, H), dtype=activation_dtype, src=dist.rank - 1)
        tokens_out['x'] = x
        y = model(None, input_embeddings=x)
        tokens_out['y'] = y

    if y is not None and y.dtype != activation_dtype and y.ndim == 3:
        dist.rprint(f'mismatched dtypes for y: {y.dtype}, {activation_dtype}', all=True)
        y = y.astype(activation_dtype)

    tok_y = None
    dy = None
    if dist.rank != dist.size - 1:
        assert y.ndim == 3 and y.shape[2] == H, 'stage output must be (B, L, H)'

        dist.rprint(f'send shape {y.shape}, dtype {y.dtype}', all=True)
        tok_y = mx.distributed.send(y, dst=dist.rank + 1)

        tokens_out['y'] = tok_y

        # Receive gradients from backwards pass
        dy = mx.distributed.recv(shape=y.shape, dtype=activation_dtype, src=dist.rank + 1)
        dy = _depends_like(dy, tok_y) # We must have sent the y tokens before we try to receive the cotangents
        tokens_out['dy'] = dy

    def local_loss(params, x):
        model.update(params)
        if dist.rank == 0:
            y_loc = model(x)
        else:
            y_loc = model(None, input_embeddings=x)

        if dist.rank != dist.size - 1:
            dy_loc = dy.astype(y_loc.dtype) if dy.dtype != y_loc.dtype else dy
            return mx.sum(y_loc * mx.stop_gradient(dy_loc))
        else:
            ce, _ = masked_loss(y_loc, targets, lengths)
            return ce

    if dist.rank == 0 and dist.rank != dist.size - 1:
        loss, g_s = mx.value_and_grad(local_loss, argnums=(0,))(
            model.trainable_parameters(), x
        )
        dx = None
    else:
        loss, (g_s, dx) = mx.value_and_grad(local_loss, argnums=(0,1))(
            model.trainable_parameters(), x
        )

    loss, g_s = None, None

    tok_dx = None
    if dist.rank != 0:
        if dx is not None and dx.dtype != activation_dtype:
            dx = dx.astype(activation_dtype)

        tok_dx = mx.distributed.send(dx, dst=dist.rank - 1)
        tokens_out['dx'] = tok_dx

    return loss, g_s, tokens_out

