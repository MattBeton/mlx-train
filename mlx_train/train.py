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

def masked_loss(logits, targets, lengths):
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce = ce.astype(mx.float32).sum() / ntoks

    return ce, ntoks

# def train_step(model, build_graph, optimizer, batch, lengths):
#     state = [model.state, optimizer.state, mx.random.state]
#
#     # @partial(mx.compile, inputs=state, outputs=state)
#     def step(batch, lengths):
#         inputs = batch[:, :-1]
#         targets = batch[:, 1:]
#
#         loss, g_s, tokens = build_graph(model, inputs, targets, lengths)
#         optimizer.update(model, g_s)
#
#         return loss, tokens
#
#     loss, tokens = step(batch, lengths)
#     eval_roots = [loss, *[x for x in list(tokens.values()) if x is not None]]
#     mx.eval(*eval_roots, model.state)
#     dist.barrier()
#
#     return loss.item()

def train_step_two(model, build_graph, optimizer, batch, lengths):
    # TODO: The fact that we need to do this in three explicit stages shows that the computational graph built by build_graph isn't completely correct.
    # It doesn't capture some of the dependencies that we need to be able to do this entirely in a single step.
    model.train()
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    loss, g_s, tokens = build_graph(model, inputs, targets, lengths)

    fwd_eval = [t for k,t in tokens.items() if k != 'dx' and t is not None]
    if dist.rank == dist.size - 1:
        fwd_eval.append(loss)
    mx.eval(*fwd_eval)
    dist.barrier()

    optimizer.update(model, g_s)
    bwd_eval = [t for k,t in tokens.items() if k == 'dx' and t is not None]
    mx.eval(*bwd_eval, g_s)
    dist.barrier()

    mx.eval(model.state, optimizer.state)
    dist.barrier()

    return float(loss.item())

def train(model: nn.Module, build_graph, optimizer, dataset_iter, config):
    model.train()

    examples_trained = 0
    step_times = []

    for batch, lengths in dataset_iter:
        start_time = time.time()
        loss = train_step_two(model, build_graph, optimizer, batch, lengths)
        step_time = time.time() - start_time
        step_times.append(step_time)

        examples_trained += batch.shape[0]

        avg_step_time = sum(step_times) / len(step_times)
        dist.rprint(f'{loss=}, {examples_trained=}, avg_step_time={avg_step_time:.3f}s', only=dist.size-1)

        if examples_trained >= config['dataset']['dataset_examples'] * config['dataset']['epochs']:
            break
