import time
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

    loss, _ = step(batch, lengths)
    mx.eval(loss, model.parameters())
    dist.barrier()

    return loss.item()

def build_loss_and_grad(model: nn.Module, loss=default_loss):
    return nn.value_and_grad(model, loss)

def train(model: nn.Module, optimizer, dataset_iter, config):
    model.train()

    loss_and_grad_fn = build_loss_and_grad(model)

    examples_trained = 0
    step_times = []

    for batch, lengths in dataset_iter:
        start_time = time.time()
        loss = train_step(model, loss_and_grad_fn, optimizer, batch, lengths)
        step_time = time.time() - start_time
        step_times.append(step_time)

        examples_trained += batch.shape[0] * dist.size

        avg_step_time = sum(step_times) / len(step_times)
        dist.rprint(f'{loss=}, {examples_trained=}, avg_step_time={avg_step_time:.3f}s')

        if examples_trained >= config['dataset']['dataset_examples'] * config['dataset']['epochs']:
            break




def train_step_simple(model, build_graph, optimizer, batch):
    state = [model.state, optimizer.state, mx.random.state]

    # @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, g_s, tokens = build_graph(model, x, y)
        optimizer.update(model, g_s)

        # mx.eval(grads)
        # print(tree_map(lambda x: x.dtype, grads))

        return loss, tokens

    loss, tokens = step(*batch)
    eval_roots = [loss, *[x for x in list(tokens.values()) if x is not None]]
    mx.eval(eval_roots, model.state)
    dist.barrier()

    return loss.item()

def train_simple(model: nn.Module, build_graph, optimizer, dataset_fn, config):
    model.train()

    examples_trained = 0
    step_times = []

    while True:
        batch = dataset_fn()

        batch = list(batch)
        for i in range(2):
            if dist.rank != 0:
                batch[i] = mx.zeros_like(batch[i])
            batch[i] = mx.distributed.all_sum(batch[i])
            mx.eval(batch[i])

        # dist.rprint(batch[0].flatten()[:10], all=True)
        # dist.rprint(batch[1].flatten()[:5], all=True)
        # break

        start_time = time.time()
        loss = train_step_simple(model, build_graph, optimizer, batch)
        step_time = time.time() - start_time
        step_times.append(step_time)

        examples_trained += batch[0].shape[0]

        avg_step_time = sum(step_times) / len(step_times)
        dist.rprint(f'{loss=}, {examples_trained=}, avg_step_time={avg_step_time:.3f}s', only=dist.size-1)

        if examples_trained >= 1000:
            break
