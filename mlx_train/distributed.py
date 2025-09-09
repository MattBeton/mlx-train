import sys

import mlx.core as mx
import mlx.utils as utils

rank, size = 0, 1

def rprint(msg: str, all: bool = False, only: int = 0):
    if all or rank == only:
        print(f"Rank {rank}: {msg}")
        sys.stdout.flush()

# def assert_distributed(func):
#     def wrapper(*args, **kwargs):
#         if rank is None or size is None:
#             raise Exception(f'Distributed communication performed when process group not initialized. {rank=}, {size=}')
#         return func(*args, **kwargs)

#     return wrapper

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
