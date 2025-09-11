import time
from typing import Dict, Any, Optional
import mlx.core as mx
import wandb
import mlx_train.distributed as dist


class TrainingLogger:
    def __init__(self, config: Dict[str, Any], total_steps: Optional[int] = None):
        self.config = config
        self.rank = dist.rank
        self.size = dist.size
        self.is_logging_rank = (dist.rank == dist.size - 1)
        
        self.global_step = 0
        self.examples_trained = 0
        self.tokens_trained = 0
        self.start_time = time.time()
        self.step_times = []
        self.initial_memory = 0.0
        
        self.total_steps = total_steps
        
        if self.is_logging_rank:
            self._init_wandb()
    
    def _init_wandb(self):
        wandb.init(
            project=self.config.get('wandb', {}).get('project', 'mlx-train'),
            name=self.config.get('wandb', {}).get('run_name', None),
            config={
                'model': self.config.get('model', {}),
                'optimizer': self.config.get('optimizer', {}),
                'dataset': self.config.get('dataset', {}),
                'topology': self.config.get('topology', {}),
                'distributed_size': self.size,
                'distributed_rank': self.rank,
            },
            tags=self.config.get('wandb', {}).get('tags', []),
            mode=self.config.get('wandb', {}).get('mode', 'online'),
        )
        
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("performance/*", step_metric="train/step")
        wandb.define_metric("memory/*", step_metric="train/step")
        wandb.define_metric("gradients/*", step_metric="train/step")
    
    def log_pre_training(self):
        peak_memory = mx.get_peak_memory() / 1024**3
        dist.rprint(f'Pre-train peak memory: {peak_memory:.2f} GB', all=True)
        self.initial_memory = peak_memory
        mx.reset_peak_memory()
    
    def log_step(
        self,
        loss: float,
        batch_shape: tuple,
        step_time: float,
        optimizer: Any,
        model: Any = None,
    ):
        self.global_step += 1
        self.step_times.append(step_time)
        
        batch_size, seq_length = batch_shape
        self.examples_trained += batch_size
        self.tokens_trained += batch_size * seq_length
        
        tps = batch_size * seq_length / step_time
        avg_step_time = sum(self.step_times) / len(self.step_times)
        
        current_memory = mx.get_peak_memory() / 1024**3
        
        if self.is_logging_rank:
            learning_rate = optimizer.learning_rate.item() if hasattr(optimizer.learning_rate, 'item') else optimizer.learning_rate
            
            grad_norm = 0.0
            if model and hasattr(model, 'parameters'):
                for p in model.parameters().values():
                    if hasattr(p, 'grad') and p.grad is not None:
                        grad_norm += mx.sum(p.grad ** 2).item()
                grad_norm = grad_norm ** 0.5
            
            metrics = {
                "train/step": self.global_step,
                "train/loss": loss,
                "train/learning_rate": learning_rate,
                "train/examples_trained": self.examples_trained,
                "train/tokens_trained": self.tokens_trained,
                "train/epoch": self.examples_trained / self.config['dataset'].get('dataset_examples', 1),
                "performance/tokens_per_second": tps,
                "performance/step_time": step_time,
                "performance/avg_step_time": avg_step_time,
                "performance/batch_size": batch_size,
                "performance/sequence_length": seq_length,
                "memory/peak_gb": current_memory,
                "memory/delta_gb": current_memory - self.initial_memory,
                "gradients/grad_norm": grad_norm,
            }
            
            wandb.log(metrics, step=self.global_step)
        
        # Print status every step on logging rank
        if self.global_step % 1 == 0:  # Print every step
            dist.rprint(
                f'Step {self.global_step}: loss={loss:.4f}, examples={self.examples_trained}, '
                f'tokens={self.tokens_trained:,}, tps={tps:.0f}, avg_time={avg_step_time:.3f}s',
                only=self.size-1
            )
    
    def log_checkpoint(self, examples_trained: int):
        if self.is_logging_rank:
            wandb.log({
                "train/checkpoint_saved": 1,
                "train/checkpoint_examples": examples_trained,
            }, step=self.global_step)
        
        dist.rprint(f"Checkpoint saved at {examples_trained} examples", only=self.size-1)
    
    def finish(self, final_loss: float):
        if self.is_logging_rank:
            final_memory = mx.get_peak_memory() / 1024**3
            total_training_time = sum(self.step_times)
            
            wandb.log({
                "summary/total_examples": self.examples_trained,
                "summary/total_tokens": self.tokens_trained,
                "summary/total_steps": self.global_step,
                "summary/total_training_time": total_training_time,
                "summary/avg_tokens_per_second": self.tokens_trained / total_training_time if total_training_time > 0 else 0,
                "summary/final_loss": final_loss,
                "summary/peak_memory_gb": final_memory,
            })
            
            wandb.finish()
        
        # Print summary on logging rank
        dist.rprint(f"\nTraining completed:", only=self.size-1)
        dist.rprint(f"  Total steps: {self.global_step}", only=self.size-1)
        dist.rprint(f"  Total examples: {self.examples_trained}", only=self.size-1)
        dist.rprint(f"  Total tokens: {self.tokens_trained:,}", only=self.size-1)
        dist.rprint(f"  Final loss: {final_loss:.4f}", only=self.size-1)
        
        if self.is_logging_rank:
            total_training_time = sum(self.step_times)
            dist.rprint(f"  Training time: {total_training_time:.1f}s", only=self.size-1)
            dist.rprint(f"  Avg TPS: {self.tokens_trained / total_training_time:.0f}", only=self.size-1)