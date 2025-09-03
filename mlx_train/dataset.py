import json
import mlx.core as mx

from mlx_lm.tuner.datasets import CacheDataset, ChatDataset, create_dataset
from mlx_lm.tuner.trainer import iterate_batches

import mlx_train.distributed as dist

def iterate_dataset(dataset_config, tokenizer):
    with open(dataset_config['path'], 'r') as f:
        data = [json.loads(l) for l in f]

    create_dataset_config = type('Config', (), {'mask_prompt': True})()
    dataset = create_dataset(data, tokenizer, create_dataset_config)
    assert isinstance(dataset, ChatDataset)

    batched_dataset = iterate_batches(
        CacheDataset(dataset), 
        batch_size=dataset_config['batch_size'], 
        max_seq_length=dataset_config['max_seq_length'],
        train=True,
    )

    dataset_config['dataset_examples'] = len(dataset)

    for batch, meta in batched_dataset:
        # batch has shape [batch, seq]
        # meta has shape [seq, 2], where the entries 
        yield batch, meta

# def iterate_dataset_random(dataset_config, tokenizer):
#     while True:
#         yield transform_batch_autoregressive(get_random_batch(
#             batch_size=dataset_config['batch_size'],
#             seq_len=dataset_config['max_seq_length'],
#             vocab_size=100000,
#         ))
#
# def transform_batch_autoregressive(batch: mx.array) -> tuple[mx.array, mx.array]:
#     return batch[:,:-1], batch[:,1:]
#
# def get_random_batch(batch_size: int, seq_len: int, vocab_size: int) -> mx.array:
#     return mx.random.randint(0, vocab_size, (batch_size, seq_len))
