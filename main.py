from typing import cast

import mlx.optimizers as optim

from mlx_train.dataset import *
import mlx_train.distributed as dist
from mlx_train.model import *
from mlx_train.train import train
from mlx_train.ppp import _LayerCallable, PipelineFirstLayer, PipelineLastLayer, IdentityLayer

from mlx.utils import tree_map_with_path, tree_flatten

from shared.config import load_config

class PPPTrainModel(nn.Module):
    def __init__(self, num_layers=4, dim=16, distributed=False):
        super().__init__()
        self.dim = dim

        self.layers = [
            nn.Linear(self.dim, self.dim, bias=False) for _ in range(num_layers)
        ]

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
        for layer in self.layers:
            # x = nn.relu(layer(x))
            x = layer(x)

        return x

def step_graph(model: PPPTrainModel, x: mx.array, y: mx.array):
    tokens = []

    if dist.rank != 0:
        x = mx.distributed.recv_like(x, src=dist.rank - 1) # here we make the assumption that the residuals are always the same shape

    y_s = model(x)

    if dist.rank != dist.size - 1:
        tok_y = mx.distributed.send(y_s, dst=dist.rank + 1)
        tokens.append(tok_y)

    if dist.rank != dist.size - 1:
        dy_s = mx.distributed.recv_like(y_s, src=dist.rank + 1)

        # (dy_s,) = mx.depends([dy_s], [tok_y])

    def local_loss(params, x):
        model.update(params)
        y_s = model(x)

        if dist.rank != dist.size - 1:
            return mx.sum(y_s * mx.stop_gradient(dy_s))
        else:
            return nn.losses.mse_loss(y_s, y)

    loss, (g_s, dx) = mx.value_and_grad(local_loss, argnums=(0,1))(
        model.trainable_parameters(), x
    )

    if dist.rank != 0:
        tok_dx = mx.distributed.send(dx, dst=dist.rank - 1)
        tokens.append(tok_dx)

    return loss, g_s, tokens

def main():
    dist.init_process_group()

    config = load_config()

    model = PPPTrainModel(distributed=True)
    # model = PPPTrainModel()

    x = mx.random.normal((4, 16))
    y = mx.random.normal((4, 16))
    mx.eval(x, model, y)

    loss, g_s, tokens = step_graph(model, x, y)

    named = {}
    tree_map_with_path(lambda path, a: named.__setitem__("grads/" + path, a), g_s)
    mx.export_to_dot('g_s.dot', loss, *tokens, **named)
    mx.eval(loss, *tokens, **named)
    # mx.eval(loss, g_s, *tokens)

    # named = {}
    # tree_map_with_path(lambda path, a: named.__setitem__("grads/" + path, a), g_s)
    # mx.export_to_dot('g_s.dot', **named)


    # def loss(model, x, y):
    #     yhat = model(x)
    #     return nn.losses.mse_loss(yhat, y)
    #
    # yhat = model(x)
    # mx.export_to_dot('forwards.dot', yhat)

    # loss, grads = nn.value_and_grad(model, loss)(model, x, y)

    # named = {}
    # tree_map_with_path(lambda path, a: named.__setitem__("grads/" + path, a), grads)
    # mx.export_to_dot("backwards2.dot", **named)





    # leaves = tree_flatten(grads)
    # mx.export_to_dot('backwards.dot', *leaves)
    # mx.eval(loss, grads)

    # model, tokenizer = load_configure_model(config['model'])
    #
    # dataset_iter = iterate_dataset(config['dataset'], tokenizer)
    # optimizer = optim.Adam(**config['optimizer'])
    #
    # train(model, optimizer, dataset_iter, config)
    #
    # if dist.rank == 0:
    #     write_model(model, config['model'])
    #
    # print(f'Peak memory usage: {mx.get_peak_memory()/1024**3:.2f}GB')
    #
if __name__ == "__main__":
    main()
