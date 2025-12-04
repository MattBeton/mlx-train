import sys

import mlx.core as mx
import mlx.utils as utils

rank, size = 0, 1

def rprint(msg: str, all: bool = False, only: int = 0):
    if all or rank == only:
        print(f"Rank {rank}: {msg}")
        sys.stdout.flush()

def init_process_group():
    global world, rank, size
    try:
        world = mx.distributed.init()
        rank = world.rank()
        size = world.size()
        if rank == 0:
            print(f"Rank {rank}: Initialized with {size} processes")
        sys.stdout.flush()
        return rank, size
    except RuntimeError as e:
        print(f"Error initializing mx.distributedributed environment: {e}")
        exit(1)

# @assert_distributed
def barrier():
    """Creates a synchronization barrier across all processes."""
    if size > 1:
        mx.eval(mx.distributed.all_sum(mx.array(0, dtype=mx.int32))) # Ensure tensor for all_sum


def all_reduce_tree(object):
  return utils.tree_map(lambda x: mx.distributed.all_sum(x) / size, object)

def assert_equal_across_ranks(x):
    x_clone = mx.array(x)

    # Sum across all ranks → (shape * world_size) IF all shapes match
    average = mx.distributed.all_sum(x_clone) / mx.distributed.init().size()
    mx.eval(average)

    if not mx.all(average == mx.array(x)):
        rprint(f"Values of tensor {x} differ across devices")
        raise AssertionError(f"Values of tensor {x} differ across devices")

