import sys
import mlx
import mlx.core as mx

world = None
rank = None

def init_process_group():
  global world, rank
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

def rprint(msg: str):
  if rank == 0:
    print(f"Rank 0: {msg}")
    sys.stdout.flush()

def main():
    init_process_group()

    

if __name__ == "__main__":
    main()
