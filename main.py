from typing import cast

from mlx.nn import Linear
import mlx.optimizers as optim

from mlx_train.dataset import *
import mlx_train.distributed as dist
from mlx_train.model import *
from mlx_train.train import train, masked_loss
from mlx_train.ppp import _LayerCallable, PipelineFirstLayer, PipelineLastLayer, IdentityLayer

from mlx.utils import tree_map_with_path, tree_flatten
from mlx_lm.tuner.trainer import default_loss
from mlx_train.utils import export_graph

from shared.config import load_config

class PPPTrainModel(nn.Module):
    def __init__(self, num_layers=4, in_dim=2, hidden=64, out_dim=1, bias=True, distributed=False):
        super().__init__()
        self.layers = [nn.Linear(in_dim, hidden, bias=bias)] + \
            [nn.Linear(hidden, hidden, bias=bias) for _ in range(num_layers - 2)] + \
            [nn.Linear(hidden, out_dim, bias=bias)]

        if distributed:
            self.layers = cast(list[_LayerCallable], self.layers)
            # if dist.rank == 0:
            #     self.layers[0] = PipelineLastLayer(self.layers[0], dist.rank, dist.size)
            # if dist.rank == 3:
            #     self.layers[0] = PipelineFirstLayer(self.layers[0], dist.rank, dist.size)
            # else:
            #     self.layers[0] = PipelineFirstLayer(PipelineLastLayer(self.layers[0], dist.rank, dist.size), dist.rank, dist.size)

            for i, layer in enumerate(self.layers):
                # if isinstance(layer, nn.Linear):
                if i != dist.rank:
                    self.layers[i] = IdentityLayer()

    def __call__(self, x: mx.array):
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, IdentityLayer):
                if i != dist.size - 1:
                    x = nn.relu(layer(x))
                else:
                    x = layer(x)


        return x

    @property
    def in_shape(self):
        return [layer for layer in self.layers if not isinstance(layer, IdentityLayer)][0].weight.shape[1]

    # @property
    # def out_shape(self):
    #     return [layer for layer in self.layers if not isinstance(layer, IdentityLayer)][-1].weight.shape[0]


def _depends_like(x: mx.array, dep: mx.array) -> mx.array:
    """Return x, but make it depend on dep (no numerical effect)."""
    z = mx.sum(mx.stop_gradient(dep))
    zero = mx.array(0, dtype=x.dtype)
    return x + zero * z # broadcasts scalar z

def step_graph(model: PPPTrainModel, x: mx.array, y: mx.array, lengths: mx.array):
    """
    tokens  : (B, L) int32
    targets : (B, L) int32 (shifted by 1 in caller)
    lengths : (B, 2) label start/end (your masked_loss)
    """
    # Build the slice runner for this rank
    start = getattr(model, "_ppp_start")
    end = getattr(model, "_ppp_end")
    inner = getattr(model, "model", getattr(model, "transformer", None))
    total_layers = len(getattr(inner, "layers", getattr(inner, "h", [])))
    slice_runner = PipelineSlice(model, start, end)

    tokens = {}

    dist.rprint(f'{x.shape}, {y.shape}, {lengths.shape}', all=True)

    if dist.rank != 0:
        x = mx.distributed.recv(shape=(x.shape[0], model.in_shape), dtype=x.dtype, src=dist.rank - 1) # here we make the assumption that the residuals are always the same shape

    y_s = model(x)

    tok_y = None
    dy_s = None
    if dist.rank != dist.size - 1:
        # Send residual stream to next device
        tok_y = mx.distributed.send(y_s, dst=dist.rank + 1)
        tokens['y_s'] = tok_y

        # Receive gradients from backwards pass
        dy_s = mx.distributed.recv_like(y_s, src=dist.rank + 1)
        dy_s = _depends_like(dy_s, tok_y) # We must have sent the y tokens before we try to receive the cotangents

    def local_loss(params, x):
        model.update(params)
        if dist.rank == 0:
            y_s = model(x)
        else:
            y_s = model(None, input_embeddings=x)

        dist.rprint(y_s.shape, all=True)

        if dist.rank != dist.size - 1:
            return mx.sum(y_s * mx.stop_gradient(dy_s))
        else:
            return masked_loss(y_s, y, lengths)

    loss, (g_s, dx) = mx.value_and_grad(local_loss, argnums=(0,1))(
        model.trainable_parameters(), x
    )

    tok_dx = None
    if dist.rank != 0:
        tok_dx = mx.distributed.send(dx, dst=dist.rank - 1)
        tokens['dx'] = tok_dx

    return loss, g_s, tokens

def main():
    dist.init_process_group()

    config = load_config()

    # model = PPPTrainModel(distributed=True)
    #
    # x = mx.random.normal((4, 2))
    # y = mx.random.normal((4, 1))
    # mx.eval(x, model, y)
    #
    # loss, g_s, tokens = step_graph(model, x, y)

    # export_graph(loss, g_s, tokens, x, y, model)
    # eval_roots = [loss, g_s, *[x for x in list(tokens.values()) if x is not None]]
    # mx.eval(*eval_roots)
    
    # optimizer = optim.Adam(**config['optimizer'])
    # optimizer = optim.SGD(learning_rate=0.01)
    #
    # def dataset_fn(n=config['dataset']['batch_size']):
    #     Xtr = mx.random.uniform(shape=(n, 2))
    #     ytr = (Xtr[:, 0:1] * Xtr[:, 1:2])
    #     return Xtr, ytr
    # train_simple(model, step_graph, optimizer, dataset_fn, config)

    model, tokenizer = load_configure_model(config['model'])

    dataset_iter = iterate_dataset(config['dataset'], tokenizer)
    optimizer = optim.Adam(**config['optimizer'])

    train(model, step_graph, optimizer, dataset_iter, config)

    if dist.rank == 0:
        write_model(model, config['model'])

    print(f'Peak memory usage: {mx.get_peak_memory()/1024**3:.2f}GB')

if __name__ == "__main__":
    main()
