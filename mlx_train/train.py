from functools import partial
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.trainer import default_loss

from mlx_train.utils import *
import mlx_train.distributed as dist

def simple_loss_fn(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    ce = nn.losses.cross_entropy(model(inputs), targets, reduction='mean')
    ntoks = targets.shape[0] * targets.shape[1]

    return ce, ntoks

def train_step(model, loss_and_grad_fn, optimizer, batch, lengths):
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, lengths):
        (loss, ntoks), grads = loss_and_grad_fn(model, batch, lengths)
        grads = nn.average_gradients(grads)
        optimizer.update(model, grads)

        # mx.eval(grads)
        # print(tree_map(lambda x: x.dtype, grads))

        return loss, ntoks

    loss, ntoks = step(batch, lengths)
    mx.eval(loss, model.parameters())
    dist.barrier()

    return loss.item()

def build_loss_and_grad(model: nn.Module, config, loss=default_loss):
    return nn.value_and_grad(model, loss)

def train(model: nn.Module, optimizer, dataset_iter, config):
    model.train()

    loss_and_grad_fn = build_loss_and_grad(model, config)

    tokens_trained = 0

    for batch, lengths in dataset_iter:
        loss = train_step(model, loss_and_grad_fn, optimizer, batch, lengths)

        tokens_trained += batch.shape[0] * dist.size

        dist.rprint(f'{loss=}, {tokens_trained=}')

        if tokens_trained >= config['dataset']['dataset_examples'] * config['dataset']['epochs']:
            break
