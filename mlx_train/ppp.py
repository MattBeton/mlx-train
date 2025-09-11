from typing import Protocol, cast, override

import mlx.core as mx
import mlx.nn as nn  # pyright: ignore[reportMissingTypeStubs]

import mlx_train.distributed as dist

from mlx_train.train import masked_loss
from mlx_lm.models.base import create_attention_mask


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
            if hasattr(full_model, 'lm_head'):
                self.lm_head = full_model.lm_head
                dist.rprint('using lm_head for output projection', all=True)
            else:
                self.lm_head = inner.embed_tokens.as_linear
                dist.rprint('using inner.embed_tokens.as_linear for output projection', all=True)

        # static info for shape/dtype on this model
        self.hidden_size = int(inner.norm.weight.shape[0])
        # dist.rprint(inner.layers[0].input_layernorm)
        # self.hidden_dtype = inner.embed_tokens.as_linear.dtype
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

def build_graph(model: PipelineSlice, tokens: mx.array, targets: mx.array, lengths: mx.array):
    """
    tokens  : (B, L) int32
    targets : (B, L) int32 (shifted by 1 in caller)
    lengths : (B, 2) label start/end (your masked_loss)
    """
    tokens_out = {}

    B, L = tokens.shape
    H = model.hidden_size
    activation_dtype = mx.bfloat16 # TODO: Should we be training LoRA at 16 or 32?

    f_tokens = getattr(model, "_compiled_fwd_tokens", None)
    f_embeds = getattr(model, "_compiled_fwd_embeds", None)

    y = None
    if dist.rank == 0:
        x = tokens
        y = f_tokens(tokens) if f_tokens else model(tokens)
    elif dist.rank != dist.size - 1:
        x = mx.distributed.recv(shape=(B, L, H), dtype=activation_dtype, src=dist.rank - 1)
        tokens_out['x'] = x

        y = f_embeds(x) if f_embeds else model(None, input_embeddings=x)
    else: # last rank
        x = mx.distributed.recv(shape=(B, L, H), dtype=activation_dtype, src=dist.rank - 1)
        tokens_out['x'] = x

        y = f_embeds(x) if f_embeds else model(None, input_embeddings=x)
        tokens_out['y'] = y

    if y is not None and y.dtype != activation_dtype and y.ndim == 3:
        y = y.astype(activation_dtype)

    tok_y = None
    dy = None
    if dist.rank != dist.size - 1:
        assert y.ndim == 3 and y.shape[2] == H, 'stage output must be (B, L, H)'

        tok_y = mx.distributed.send(y, dst=dist.rank + 1)

        tokens_out['y'] = tok_y

        # Receive gradients from backwards pass
        dy = mx.distributed.recv(shape=y.shape, dtype=activation_dtype, src=dist.rank + 1)

        # We must have sent the y tokens before we try to receive the cotangents.
        # This is only necessary when we do the whole train in a single step.
        # dy = _depends_like(dy, tok_y) 
        tokens_out['dy'] = dy

    def local_loss(params, x):
        model.update(params)
        if dist.rank == 0:
            y_loc = f_tokens(x) if f_tokens else model(x)
        else:
            y_loc = f_embeds(x) if f_embeds else model(None, input_embeddings=x)

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

    tok_dx = None
    if dist.rank != 0:
        if dx is not None and dx.dtype != activation_dtype:
            dx = dx.astype(activation_dtype)

        tok_dx = mx.distributed.send(dx, dst=dist.rank - 1)
        tokens_out['dx'] = tok_dx

    return loss, g_s, tokens_out


def install_compiled_forward(model: PipelineSlice, *, shapeless: bool = True) -> None:
    """
    Compile the per-rank forward calls and attach them to the model instance.
    We intentionally do NOT compile the outer build_graph (it returns distributed tokens).
    """
    def _fwd_tokens(tokens: mx.array, mask: mx.array | None = None):
        return model(tokens, mask)

    def _fwd_embeds(input_embeddings: mx.array, mask: mx.array | None = None):
        return model(None, input_embeddings=input_embeddings, mask=mask)

    compiled_tokens = mx.compile(
        _fwd_tokens, 
        inputs=[model.state, mx.random.state],
        outputs=[model.state, mx.random.state],
        shapeless=shapeless
    )
    compiled_embeds = mx.compile(
        _fwd_embeds, 
        inputs=[model.state, mx.random.state],
        outputs=[model.state, mx.random.state],
        shapeless=shapeless
    )

    setattr(model, "_compiled_fwd_tokens", compiled_tokens)
    setattr(model, "_compiled_fwd_embeds", compiled_embeds)
