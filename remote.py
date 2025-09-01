import argparse
import asyncio
import os
from pathlib import Path
from shared.config import *
from mlx_configure.distributed_config import distributed_config
from mlx_configure.run import run
from mlx_configure.rsync import sync_all_hosts, load_hosts

async def rsync_to_hosts(active_hosts):
    """
    Run rsync to sync code to active hosts using the rsync module.
    
    Args:
        active_hosts: List of hostnames from config.yaml to sync to
    """
    destination = "/Users/Shared/mlx-train"
    
    # Run rsync synchronously in executor to avoid blocking
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, sync_all_hosts, active_hosts, destination, 5)
    
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
    rsync_task = rsync_to_hosts(hosts)
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


async def handle_sudo_command(args):
    """Handle the sudo command - enable password-less sudo on hosts."""
    # Load configuration
    config = load_config()
    active_hosts = config['topology']['hosts']
    
    # We'll use the hostnames directly from config.yaml
    # Since password-less SSH is guaranteed, we don't need hosts.json
    hosts_to_configure = active_hosts
    
    if not hosts_to_configure:
        print("Error: No hosts found in config.yaml")
        return 1
    
    # Set defaults
    remote_user = args.remote_user or os.environ.get("USER", "admin")
    rule_target = args.rule_target
    
    # Show what we're about to do
    print(f"Enable Password-less Sudo Configuration")
    print(f"=" * 40)
    print(f"Active hosts from config.yaml: {active_hosts}")
    print(f"Number of hosts to configure: {len(hosts_to_configure)}")
    print(f"Remote user: {remote_user}")
    print(f"Rule target: {rule_target}")
    print(f"=" * 40)
    
    # Warn about security implications
    if rule_target == "%admin":
        print("\n⚠️  WARNING: This will enable password-less sudo for ALL admin users!")
    else:
        print(f"\n📝 This will enable password-less sudo for: {rule_target}")
    
    print("\nThis creates /etc/sudoers.d/fleet-nopasswd with NOPASSWD rule.")
    
    # Confirm unless --no-prompt
    if not args.no_prompt:
        response = input("\nContinue? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Aborted.")
            return 0
    
    # Get sudo password
    sudo_password = getpass.getpass(f"\nAdmin password for sudo on remote hosts: ")
    
    print(f"\nApplying configuration to {len(hosts_to_configure)} hosts...")
    print("-" * 40)
    
    # Apply to all hosts (run synchronously in executor)
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        enable_nopasswd_all_hosts,
        hosts_to_configure,
        remote_user,
        rule_target,
        sudo_password,
        5  # max_workers
    )
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print("-" * 40)
    print(f"Summary: {successful} successful, {failed} failed")
    
    if successful > 0:
        print(f"\n✅ Password-less sudo enabled on {successful} host(s)")
        print(f"   Rule: {rule_target} ALL=(ALL) NOPASSWD:ALL")
        print(f"   File: /etc/sudoers.d/fleet-nopasswd")
    
    # Return error if any failed
    return 1 if failed > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="Remote management tool for distributed MLX training."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # Run command
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
    
    # Sudo configuration command
    parser_sudo = subparsers.add_parser(
        "sudo",
        help="Enable password-less sudo on hosts in config.yaml",
        description="Configure password-less sudo for fleet management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Enable for admin group (all admin users):
  python remote.py sudo
  
  # Enable for a specific user:
  python remote.py sudo --rule-target matt
  
  # Use different SSH username:
  python remote.py sudo --remote-user administrator
  
  # Skip confirmation prompt:
  python remote.py sudo --no-prompt
  
Note: Use % for group names (e.g., %admin, %wheel)
        """
    )
    parser_sudo.add_argument(
        "--remote-user",
        default=None,
        help="Remote username to SSH as (default: current user)"
    )
    parser_sudo.add_argument(
        "--rule-target",
        default="%admin",
        help="Target for sudo rule: username or %%groupname (default: %%admin)"
    )
    parser_sudo.add_argument(
        "--no-prompt",
        action="store_true",
        help="Don't prompt for confirmation (use with caution)"
    )

    args = parser.parse_args()
    
    if args.command == "run":
        # Run the async handler
        result = asyncio.run(handle_run_command(args))
        exit(result)
    elif args.command == "sudo":
        # Run the sudo configuration handler
        result = asyncio.run(handle_sudo_command(args))
        exit(result)


if __name__ == "__main__":
    main()