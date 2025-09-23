
import mlx.optimizers as optim

from mlx_train.dataset import *
import mlx_train.distributed as dist
from mlx_train.model import *
from mlx_train.train import train
from mlx_train.ppp import build_graph, install_compiled_forward
from mlx_train.model import write_adapters_distributed


from shared.config import load_config


def main():
    dist.init_process_group()

    config = load_config()

    model, tokenizer = load_configure_model(config['model'])

    dataset_iter = iterate_dataset(
        config['dataset'], 
        tokenizer, 
        parallelism=config.get('model', {}).get('distributed'))

    optimizer = optim.Adam(**config['optimizer'])

    dist.rprint(f'Pre-training peak memory usage: {mx.get_peak_memory()/1024**3:.2f}GB', all=True)

    if config['model']['compile_graph'] and \
        config.get('model', {}).get('distributed') == 'pp':
        assert isinstance(model, PipelineSlice)
        install_compiled_forward(model)

    train(model, build_graph, optimizer, dataset_iter, config, write_adapters_distributed)

    dist.rprint(f'Peak memory usage: {mx.get_peak_memory()/1024**3:.2f}GB', all=True)

    if 'lora' in config['model']:
        if config.get('model', {}).get('distributed') == 'pp':
            write_adapters_distributed(model, config['model'])
        elif config.get('model', {}).get('distributed') == 'dp':
            write_adapters(model, config['model']) # TODO
    else:
        dist.rprint('Non-LoRA model checkpointing not implemented.')


if __name__ == "__main__":
    main()
