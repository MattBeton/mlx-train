import json

from mlx_lm.tuner.datasets import CacheDataset, ChatDataset, create_dataset
from mlx_lm.tuner.trainer import iterate_batches

def iterate_dataset(dataset_config, tokenizer, parallelism=None):
    with open(dataset_config['path'], 'r') as f:
        data = [json.loads(l) for l in f]

    create_dataset_config = type('Config', (), {'mask_prompt': True})()
    dataset = create_dataset(data, tokenizer, create_dataset_config)
    assert isinstance(dataset, ChatDataset)

    # dist.rprint(dataset[0], all=True)

    batched_dataset = iterate_batches(
        CacheDataset(dataset), 
        batch_size=dataset_config['batch_size'], 
        max_seq_length=dataset_config['max_seq_length'],
        loop=False,
        seed=42,
    )

    dataset_config['dataset_examples'] = len(dataset)

    for batch, length in batched_dataset:
        # batch has shape [batch, seq]
        # meta has shape [seq, 2], where the entries 

        yield batch, length
