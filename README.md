# MLX Train

**Model-parallel training and finetuning on Apple Silicon devices**

Finetune DeepSeek V3.1 (8 bit) on two Mac Studios!

## Codebase Structure

### MLX Train

##### `ppp.py`

This is where the magic happens! The pipeline parallel file provides two things:
- `build_graph` method, which builds the computational graph for a full forwards-backwards pass, including send/recv operations between ranks for both activations (`y`) and partial derivatives (`dx`)
- `PipelineSlice` class, which splits a full (not-yet-loaded) model into just the layers needed on this rank

##### `train.py`

Exposes a standard training loop with MLX compilation. Evaluation is currently performed in two steps (forwards and backwards pass).

##### `model.py` 

- Load a model from safetensors
- Apply LoRA, pipeline splitting, gradient checkpointing to the model
- Output trained LoRA adapters, sent from all ranks in the ring

##### `distributed.py` 

Utilities for distributed training on MLX, such as `dist.barrier` and `dist.rprint`

#### `mlx_configure`

Tooling for automatically running distributed MLX experiments across devices:
- Automatic setup of [MLX Ring](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) over thunderbolt using `mlx.distributed_config`
- Synchronization of code & UV environment across devices
- Peer liveness & peer ping checks
