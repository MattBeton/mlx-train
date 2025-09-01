import argparse
import asyncio
from pathlib import Path
from shared.config import *
from mlx_configure.distributed_config import distributed_config
from mlx_configure.run import run
from mlx_configure.rsync import sync_all_hosts, load_hosts


async def rsync_to_hosts():
    """
    Run rsync to sync code to all hosts using the rsync module.
    """
    # Load hosts from hosts.json
    hosts = load_hosts("hosts.json")
    destination = "/Users/Shared/mlx-train"
    
    # Run rsync synchronously in executor to avoid blocking
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, sync_all_hosts, hosts, destination, 5)
    
    # Check if all syncs were successful
    failed = sum(1 for r in results if not r["success"])
    return 0 if failed == 0 else 1


async def handle_run_command(args):
    """Handle the run command - rsync code and configure distributed setup in parallel, then run training."""
    # Load configuration
    config = load_config()
    hosts = config['topology']['hosts']
    
    print(f"Setting up distributed training with hosts: {hosts}")
    
    # Step 1: Run rsync and distributed_config in parallel
    print("\n1. Syncing code and configuring distributed setup in parallel...")
    
    # Create tasks for rsync and distributed config
    rsync_task = rsync_to_hosts()
    config_task = distributed_config(hosts)
    
    # Run both tasks in parallel
    results = await asyncio.gather(rsync_task, config_task, return_exceptions=True)
    
    # Check rsync result
    rsync_result = results[0]
    if isinstance(rsync_result, Exception):
        print(f"✗ Rsync failed: {rsync_result}")
        return 1
    elif rsync_result != 0:
        print(f"✗ Rsync failed with code {rsync_result}")
        return rsync_result
    
    # Check distributed_config result
    config_result = results[1]
    if isinstance(config_result, Exception):
        print(f"✗ Distributed configuration failed: {config_result}")
        return 1
    elif config_result != 0:
        print(f"✗ Distributed configuration failed with code {config_result}")
        return config_result
    
    print("✓ Code synced and distributed configuration complete")
    
    # Step 2: Run the training
    print(f"\n2. Starting training on {len(hosts)} hosts...")
    script_path = args.script if hasattr(args, 'script') else "main.py"
    run_result = await run(script_path)
    
    if run_result != 0:
        print(f"Error: Training failed with code {run_result}")
        return run_result
    
    print("✓ Training complete")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Network debugging tool for checking host connectivity and IP assignments."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    parser_run = subparsers.add_parser(
        "run",
        help="Run training code according to the config in config.yaml",
        description="Configure distributed MLX setup and run training",
    )
    parser_run.add_argument(
        "script",
        nargs="?",
        default="main.py",
        help="Python script to run (default: main.py)"
    )

    args = parser.parse_args()
    
    if args.command == "run":
        # Run the async handler
        result = asyncio.run(handle_run_command(args))
        exit(result)


if __name__ == "__main__":
    main()