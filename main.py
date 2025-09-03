import sys
import mlx
import mlx.core as mx
import mlx.utils as utils
from mlx.nn.utils import tree_map
import mlx.optimizers as optim

from mlx_train.dataset import *
import mlx_train.distributed as dist
from mlx_train.model import *
from mlx_train.train import train

from shared.config import load_config


def main():
    dist.init_process_group()

    config = load_config()    

    model, tokenizer = load_model(config['model'])
    print(f'rank {dist.rank} model loaded')

    dataset_iter= iterate_dataset(config['dataset'], tokenizer)
    optimizer = optim.Adam(learning_rate=0.0004)

    train(model, optimizer, dataset_iter, config)

    # tree_map(lambda x: print(x.shape), model.parameters())

    model = dist.all_reduce_tree(model)

if __name__ == "__main__":
    main()
