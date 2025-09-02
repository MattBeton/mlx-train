import sys
import mlx
import mlx.core as mx
from mlx.nn.utils import tree_map

import mlx_train.distributed as dist
from mlx_train.model import *
from shared.config import load_config

def rprint(msg: str):
    if dist.rank == 0:
        print(f"Rank 0: {msg}")
        sys.stdout.flush()

def main():
    dist.init_process_group()

    config = load_config()    

    model, tokenizer = load_model_tokenizer(config['model'])

    tree_map(lambda x: print(x), model.parameters())

    model = dist.all_reduce_tree(model)

if __name__ == "__main__":
    main()
