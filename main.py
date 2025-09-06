from typing import cast

from mlx.nn import Linear
import mlx.optimizers as optim

from mlx_train.dataset import *
import mlx_train.distributed as dist
from mlx_train.model import *
from mlx_train.train import train, masked_loss
from mlx_train.ppp import _LayerCallable, PipelineFirstLayer, PipelineLastLayer, IdentityLayer, step_graph

from mlx.utils import tree_map_with_path, tree_flatten
from mlx_lm.tuner.trainer import default_loss
from mlx_train.utils import export_graph

from shared.config import load_config


def main():
    dist.init_process_group()

    config = load_config()
    # mx.random.seed = 42
    # np.random.seed = 42

    model, tokenizer = load_configure_model(config['model'])

    dataset_iter = iterate_dataset(config['dataset'], tokenizer)
    optimizer = optim.Adam(**config['optimizer'])

    train(model, step_graph, optimizer, dataset_iter, config)

    if dist.rank == 0:
        write_model(model, config['model'])

    print(f'Peak memory usage: {mx.get_peak_memory()/1024**3:.2f}GB')

if __name__ == "__main__":
    main()
