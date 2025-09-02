import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils
import time
import dataclasses
from mlx_lm.models.gpt2 import Model as GPT2, ModelArgs
from utils import *
import json
import sys
from tqdm import tqdm
from data import get_dataset
import dotenv
import os
import wandb

def train_step(model, loss_and_grad_fn, optimizer, batch, minibatch_size: int):
    # TODO: compile
    def step(X, y):
        loss, grads = loss_and_grad_fn(model, X, y)
        grads = nn.average_gradients(grads)
        optimizer.update(model, grads)
        return loss

    mx.eval(loss, accumulated_grads)
    barrier()

    accumulated_loss = accumulated_loss / grad_accumulation_steps
    accumulated_grads = utils.tree_map(lambda x: x / grad_accumulation_steps, accumulated_grads)
    grads = nn.average_gradients(accumulated_grads)

    optimizer.update(model, grads)
    mx.synchronize()

    return accumulated_loss

def build_loss_and_grad(model: nn.Module, config):
    def loss_fn(model, X, y):
        return nn.losses.cross_entropy(model(X), y, reduction='mean')

    return nn.value_and_grad(model, loss_fn)

def train(model: nn.Module, optimizer, data, config):
    loss_and_grad_fn = build_loss_and_grad(model, config)

    