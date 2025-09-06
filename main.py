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
    mx.random.seed = 42
    np.random.seed = 42

    # model = PPPTrainModel(distributed=True)
    #
    # x = mx.random.normal((4, 2))
    # y = mx.random.normal((4, 1))
    # mx.eval(x, model, y)
    #
    # loss, g_s, tokens = step_graph(model, x, y)

    # export_graph(loss, g_s, tokens, x, y, model)
    # eval_roots = [loss, g_s, *[x for x in list(tokens.values()) if x is not None]]
    # mx.eval(*eval_roots)
    
    # optimizer = optim.Adam(**config['optimizer'])
    # optimizer = optim.SGD(learning_rate=0.01)
    #
    # def dataset_fn(n=config['dataset']['batch_size']):
    #     Xtr = mx.random.uniform(shape=(n, 2))
    #     ytr = (Xtr[:, 0:1] * Xtr[:, 1:2])
    #     return Xtr, ytr
    # train_simple(model, step_graph, optimizer, dataset_fn, config)

    # model, tokenizer = load_configure_model(config['model'])
    # del config['model']['auto_parallel']
    model, tokenizer = load_configure_model(config['model'])
    # test = model.lm_head
    # dist.rprint(str(dir(model)))

    dataset_iter = iterate_dataset(config['dataset'], tokenizer)
    optimizer = optim.Adam(**config['optimizer'])

    train(model, step_graph, optimizer, dataset_iter, config)

    if dist.rank == 0:
        write_model(model, config['model'])

    print(f'Peak memory usage: {mx.get_peak_memory()/1024**3:.2f}GB')

if __name__ == "__main__":
    main()
