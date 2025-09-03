from functools import partial
import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import tree_map
import mlx.optimizers as optim
import mlx.utils as utils
import time
import dataclasses
from mlx_lm.models.gpt2 import Model as GPT2, ModelArgs
from mlx_train.utils import *
import json
import sys
from tqdm import tqdm
import dotenv
import os

import mlx_train.distributed as dist

def train_step(model, loss_and_grad_fn, optimizer, batch):
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(X, y):
        loss, grads = loss_and_grad_fn(model, X, y)
        grads = nn.average_gradients(grads)
        optimizer.update(model, grads)

        return loss

    loss = step(*batch)
    mx.eval(loss, model.parameters())
    dist.barrier()

    return loss.item()

def build_loss_and_grad(model: nn.Module, config):
    def loss_fn(model, X, y):
        return nn.losses.cross_entropy(model(X), y, reduction='mean')

    return nn.value_and_grad(model, loss_fn)

def train(model: nn.Module, optimizer, dataset_iter, config):
    model.train()

    loss_and_grad_fn = build_loss_and_grad(model, config)

    if dist.rank == 0:
        dataset_iter = tqdm(dataset_iter, desc="Training")
    
    for batch in dataset_iter:
        loss = train_step(model, loss_and_grad_fn, optimizer, batch)

        if dist.rank == 0:
            dataset_iter.set_postfix(loss=f"{loss:.4f}")
        else:
            dist.rprint(f'{loss=}')