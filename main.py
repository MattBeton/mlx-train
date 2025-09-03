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

    model, tokenizer = load_model(config['model'])

    if 'lora' in config['model']:
        model = lora_model(model, config['model']['lora'])

    # model = dequantize(model)
    # print(tree_map(lambda x: x.dtype, model.parameters()))
    print(f'rank {dist.rank} model loaded')

    dataset_iter = iterate_dataset(config['dataset'], tokenizer)
    optimizer = optim.Adam(**config['optimizer'])

    # for batch, lengths in dataset_iter:
    #     print([x.item() for x in list(batch[0])])
    #
    #     targets = batch[:, 1:]
    #     steps = mx.arange(1, targets.shape[1] + 1)
    #     mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    #
    #     detokenizer = tokenizer.detokenizer
    #     for tok, m in zip(targets[0], mask[0]):
    #         if m.item():
    #             detokenizer.add_token(tok.item())
    #     print(detokenizer.text)
    #
    #     raise Exception('stop')

    train(model, optimizer, dataset_iter, config)

    if dist.rank == 0:
        write_model(model, config['model'])

if __name__ == "__main__":
    main()
