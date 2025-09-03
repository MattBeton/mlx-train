import mlx.optimizers as optim

from mlx_train.dataset import *
import mlx_train.distributed as dist
from mlx_train.model import *
from mlx_train.train import train

# from lx_lm.tuner.utils import dequantize

from shared.config import load_config


def main():
    dist.init_process_group()

    config = load_config()

    model, tokenizer = load_configure_model(config['model'])

    dataset_iter = iterate_dataset(config['dataset'], tokenizer)
    optimizer = optim.Adam(**config['optimizer'])

    train(model, optimizer, dataset_iter, config)

    if dist.rank == 0:
        write_model(model, config['model'])

    print(f'Peak memory usage: {mx.get_peak_memory()/1024**3:.2f}GB')

if __name__ == "__main__":
    main()
