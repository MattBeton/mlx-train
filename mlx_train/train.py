import time

import mlx.core as mx
import mlx.nn as nn

from mlx_train.utils import *
import mlx_train.distributed as dist
from mlx_train.logger import TrainingLogger

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

def train_step_two(model, build_graph, optimizer, batch, lengths):
    model.train()
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
 
    loss, g_s, tokens = build_graph(model, inputs, targets, lengths)

    fwd_eval = [t for k,t in tokens.items() if k == 'y' and t is not None]
    fwd_recv_eval = [t for k,t in tokens.items() if k == 'x' and t is not None]
    bwd_eval = [t for k,t in tokens.items() if k == 'dx' and t is not None]
    bwd_recv_eval = [t for k,t in tokens.items() if k == 'dy' and t is not None]

    if dist.rank == dist.size - 1:
        fwd_eval.append(loss)
    
    for stage in range(3):
        dist.barrier()
        if dist.rank == stage + 1:
            mx.eval(*fwd_recv_eval)
        if dist.rank == stage:
            mx.eval(*fwd_eval)

    for stage in range(3, 0, -1):
        dist.barrier()
        if dist.rank == stage - 1:
            mx.eval(*bwd_recv_eval)
        if dist.rank == stage:
            mx.eval(*bwd_eval)

    dist.barrier()

    optimizer.update(model, g_s)
    mx.eval(model.state, optimizer.state)

    dist.barrier()

    return float(loss.item())

def train(model: nn.Module, build_graph, optimizer, dataset_iter, config, write_adapters_fn=None):
    model.train()
    
    total_steps = config['dataset'].get('dataset_examples', 1) * config['dataset'].get('epochs', 1) // config['dataset'].get('batch_size', 1)
    logger = TrainingLogger(config, total_steps=total_steps)
    logger.log_pre_training()

    last_loss = 0.0
    for batch, lengths in dataset_iter:
        start_time = time.time()
        loss = train_step_two(model, build_graph, optimizer, batch, lengths)
        step_time = time.time() - start_time
        
        logger.log_step(loss, batch.shape, step_time, optimizer, model)
        last_loss = loss

        if logger.examples_trained >= config['dataset']['dataset_examples'] * config['dataset']['epochs']:
            break

        if logger.global_step % 10 == 0 and 'lora' in config['model'] and write_adapters_fn:
            write_adapters_fn(model, config['model'], output_filepath=f'{logger.examples_trained}')
            logger.log_checkpoint(logger.examples_trained)
    
    logger.finish(last_loss)
