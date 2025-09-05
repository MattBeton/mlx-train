from typing import Protocol, cast, override

import mlx.core as mx
import mlx.nn as nn  # pyright: ignore[reportMissingTypeStubs]

import mlx_train.distributed as dist


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


def inner_model(model: nn.Module) -> nn.Module:
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        return inner

    inner = getattr(model, "transformer", None)
    if isinstance(inner, nn.Module):
        return inner

    raise ValueError("Model must either have a 'model' or 'transformer' attribute")


def auto_parallel(
    model: nn.Module, start_layer: int, end_layer: int
) -> nn.Module:
    """
    Automatically parallelize a model across multiple devices.

    Args:
      model: The model to parallelize (must have a 'layers' or 'h' property)

    Returns:
      The parallelized model
    """
    inner_model_instance: nn.Module = inner_model(model)

    # Handle both model.layers and model.h cases
    layers: list[_LayerCallable]
    if hasattr(inner_model_instance, "layers"):
        layers = cast(list[_LayerCallable], inner_model_instance.layers)
    else:
        layers = cast(list[_LayerCallable], inner_model_instance.h)

    layers[:start_layer] = [
        IdentityLayer() for _ in range(start_layer)
    ]
    layers[end_layer :] = [
        IdentityLayer() for _ in range(len(layers) - end_layer)
    ]
    layers[start_layer] = PipelineFirstLayer(
        layers[start_layer],
        dist.rank,
        dist.size,
    )
    layers[end_layer - 1] = PipelineLastLayer(
        layers[end_layer - 1],
        dist.rank,
        dist.size,
    )

    # At this point `layers` *must* be a concrete list.
    assert isinstance(layers, list), (
        "Expected a list of layers after auto-parallel initialisation"
    )

    return model
