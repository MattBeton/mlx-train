
import argparse
import asyncio
import os
import getpass
import subprocess
from mlx_configure.hosts import check_ring
from shared.config import *
from mlx_configure.distributed_config import distributed_config
from mlx_configure.run import run
from mlx_configure.rsync import sync_all_hosts
from mlx_configure.restart import restart_all_hosts, restart_host, load_hosts_from_json
from mlx_configure.enable_nopasswd_sudo import enable_nopasswd_all_hosts
from mlx_configure.ring_sync import sync_through_ring

async def rsync_to_hosts(active_hosts):
        """
        Run rsync to sync code to active hosts using the rsync module.
        
        Args:
                active_hosts: List of hostnames from config.yaml to sync to
        """
        destination = "/Users/Shared/mlx-train"
        
        # Run rsync synchronously in executor to avoid blocking
        loop = asyncio.get_event_loop()
        # sync_all_hosts expects: (hosts, destination_path, source_path, setup_uv, max_workers)
        results = await loop.run_in_executor(None, sync_all_hosts, active_hosts, destination, "./", True, 5)
        
        # Check if all syncs were successful
        failed = sum(1 for r in results if not r["success"])
        return 0 if failed == 0 else 1


async def sync_lora_adapters_from_rank0(hosts):
        """
        Sync LoRA adapters from rank 0 host (first host) to local machine.
        
        Args:
                hosts: List of hostnames from config.yaml
        """
        if not hosts:
                print("✗ No hosts found to sync from")
                return 1
        
        rank0_host = hosts[0]
        print(f"\n📦 Syncing LoRA adapters from {rank0_host} (rank 0)...")
        
        # Use rsync to sync model_output directory from rank 0 host
        rsync_cmd = [
                "rsync",
                "-avz",
                "--delete",
                f"{rank0_host}:/Users/Shared/mlx-train/model_output/",
                "./model_output/"
        ]
        
        try:
                # Run rsync command
                loop = asyncio.get_event_loop()
                proc = await asyncio.create_subprocess_exec(
                        *rsync_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await proc.communicate()
                
                if proc.returncode == 0:
                        print(f"✓ Successfully synced LoRA adapters from {rank0_host}")
                        return 0
                else:
                        print(f"✗ Failed to sync LoRA adapters from {rank0_host}")
                        if stderr:
                                print(f"  Error: {stderr.decode().strip()}")
                        return proc.returncode
                        
        except Exception as e:
                print(f"✗ Error syncing LoRA adapters: {e}")
                return 1


async def handle_run_command(args):
        """Handle the run command - rsync code and check ring in parallel, setup ring if needed, then run training."""
        # Load configuration
        config = load_config()
        hosts = config['topology']['hosts']
        
        print(f"Setting up distributed training with hosts: {hosts}")
        
        # Check if force_dist_init flag is set
        if hasattr(args, 'force_dist_init') and args.force_dist_init:
                # Run rsync and distributed_config in parallel when force_dist_init is set
                print("\n1. Syncing code and initializing distributed configuration in parallel (--force_dist_init)...")
                
                # Create tasks for rsync and distributed config
                rsync_task = rsync_to_hosts(hosts)
                dist_config_task = distributed_config(hosts)
                
                # Run both tasks in parallel
                results = await asyncio.gather(rsync_task, dist_config_task, return_exceptions=True)
                
                # Check rsync result
                rsync_result = results[0]
                if isinstance(rsync_result, Exception):
                        print(f"✗ Rsync failed: {rsync_result}")
                        return 1
                elif rsync_result != 0:
                        print(f"✗ Rsync failed with code {rsync_result}")
                        return rsync_result
                
                print("✓ Code synced successfully")
                
                # Check distributed config result
                dist_config_result = results[1]
                if isinstance(dist_config_result, Exception):
                        print(f"✗ Distributed configuration failed: {dist_config_result}")
                        return 1
                elif dist_config_result != 0:
                        print(f"✗ Distributed configuration failed with code {dist_config_result}")
                        return dist_config_result
                
                print("✓ Distributed configuration complete")
                
                # Verify ring connectivity after setup
                print("\n2. Verifying ring connectivity after setup...")
                ring_verify_result = await check_ring()
                
                if not ring_verify_result:
                        print("✗ Ring connectivity verification failed after setup")
                        return 1
                
                print("✓ Ring connectivity verified successfully")
        else:
                # Default behavior: Run rsync and check_ring in parallel
                print("\n1. Syncing code and checking ring connectivity in parallel...")
                
                # Create tasks for rsync and ring check
                rsync_task = rsync_to_hosts(hosts)
                ring_check_task = check_ring()
                
                # Run both tasks in parallel
                results = await asyncio.gather(rsync_task, ring_check_task, return_exceptions=True)
                
                # Check rsync result
                rsync_result = results[0]
                if isinstance(rsync_result, Exception):
                        print(f"✗ Rsync failed: {rsync_result}")
                        return 1
                elif rsync_result != 0:
                        print(f"✗ Rsync failed with code {rsync_result}")
                        return rsync_result
                
                print("✓ Code synced successfully")
                
                # Check ring connectivity result
                ring_check_result = results[1]
                ring_is_healthy = False
                
                if isinstance(ring_check_result, Exception):
                        print(f"✗ Ring check failed with exception: {ring_check_result}")
                elif ring_check_result:
                        print("✓ Ring connectivity is already healthy")
                        ring_is_healthy = True
                else:
                        print("✗ Ring connectivity check failed")
                
                # Step 2: If ring check failed, run distributed_config to set up the ring
                if not ring_is_healthy:
                        print("\n2. Setting up distributed configuration...")
                        config_result = await distributed_config(hosts)
                        
                        if config_result != 0:
                                print(f"✗ Distributed configuration failed with code {config_result}")
                                return config_result
                        
                        print("✓ Distributed configuration complete")
                        
                        # Step 3: Verify ring connectivity after setup
                        print("\n3. Verifying ring connectivity after setup...")
                        ring_verify_result = await check_ring()
                        
                        if not ring_verify_result:
                                print("✗ Ring connectivity verification failed after setup")
                                return 1
                        
                        print("✓ Ring connectivity verified successfully")
        
        # Run the training
        next_step = "3" if (hasattr(args, 'force_dist_init') and args.force_dist_init) else "4"
        print(f"\n{next_step}. Starting training on {len(hosts)} hosts...")
        script_path = args.script if hasattr(args, 'script') else "main.py"
        run_result = await run(script_path)
        
        if run_result != 0:
                print(f"Error: Training failed with code {run_result}")
                return run_result
        
        print("✓ Training complete")
        
        # Post-training: Sync LoRA adapters from rank 0 host
        sync_result = await sync_lora_adapters_from_rank0(hosts)
        
        if sync_result != 0:
                print(f"⚠️  Warning: Failed to sync LoRA adapters from rank 0, but training was successful")
                # Don't fail the entire command if sync fails
        
        return 0



async def handle_ring_command(args):
        result = await check_ring()

        if not result:
                print('Ring failed!')
        
        print("✓ Training complete")
        return 0


async def handle_restart_command(args):
        """Handle the restart command - reboot hosts."""
        if args.host:
                # Restart a specific host
                print(f"Attempting to reboot {args.host}...")
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, restart_host, args.host)
                
                if result["success"]:
                        print(f"{args.host}: {result['message']}")
                        return 0
                else:
                        print(f"{args.host}: {result['message']}")
                        return 1
        else:
                # Restart all hosts from config or hosts.json
                if args.use_hosts_json:
                        # Use hosts.json
                        try:
                                loop = asyncio.get_event_loop()
                                hosts = await loop.run_in_executor(None, load_hosts_from_json)
                        except FileNotFoundError as e:
                                print(f"Error: {e}")
                                return 1
                        
                        if not hosts:
                                print("No hosts found in hosts.json")
                                return 1
                else:
                        # Use config.yaml (default)
                        config = load_config()
                        hosts = config['topology']['hosts']
                        
                        if not hosts:
                                print("No hosts found in config.yaml")
                                return 1
                
                print(f"Rebooting {len(hosts)} hosts...")
                print("-" * 40)
                
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, restart_all_hosts, hosts, 5)
                
                # Summary
                successful = sum(1 for r in results if r["success"])
                failed = len(results) - successful
                
                print("-" * 40)
                print(f"Summary: {successful} successful, {failed} failed")
                
                return 1 if failed > 0 else 0


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
        print("Enable Password-less Sudo Configuration")
        print("=" * 40)
        print(f"Active hosts from config.yaml: {active_hosts}")
        print(f"Number of hosts to configure: {len(hosts_to_configure)}")
        print(f"Remote user: {remote_user}")
        print(f"Rule target: {rule_target}")
        print("=" * 40)
        
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
        sudo_password = getpass.getpass("\nAdmin password for sudo on remote hosts: ")
        
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
                print("   File: /etc/sudoers.d/fleet-nopasswd")
        
        # Return error if any failed
        return 1 if failed > 0 else 0


async def handle_ring_sync_command(args):
        """Handle the ring-sync command - sync files through the ring topology."""
        print("Starting ring-based synchronization...")
        
        # Run the ring sync
        result = await sync_through_ring(
                args.source,
                args.destination,
                args.initial_host,
                args.hosts_file,
                args.propagate_keys,
                args.verbose,
                args.parallel
        )
        
        return result


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
        parser_run.add_argument(
                "--force_dist_init",
                action='store_true',
        )

        # Check ring command 
        parser_run = subparsers.add_parser(
                "ring",
                help="Check that the ring is built",
                description="Asserts that all rings are ping-able",
        )
        
        # Restart command
        parser_restart = subparsers.add_parser(
                "restart",
                help="Restart (reboot) hosts via SSH",
                description="Send reboot commands to hosts",
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""Examples:
    # Restart all hosts from config.yaml:
    python remote.py restart
    
    # Restart all hosts from hosts.json:
    python remote.py restart --use-hosts-json
    
    # Restart a specific host:
    python remote.py restart --host james
    
Note: Requires password-less sudo on target hosts
                """
        )
        parser_restart.add_argument(
                "--host",
                help="Specific host to restart (if not provided, restarts all hosts)"
        )
        parser_restart.add_argument(
                "--use-hosts-json",
                action="store_true",
                help="Use hosts.json instead of config.yaml for host list"
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
        
        # Ring sync command
        parser_ring_sync = subparsers.add_parser(
                "ring-sync",
                help="Sync files through the ring of hosts",
                description="Propagate files from one node to all others via ring topology",
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""Examples:
    # Sync local folder to all nodes through the ring:
    python remote.py ring-sync /path/to/folder /remote/destination
    
    # Sync from a specific host through the ring:
    python remote.py ring-sync /path/to/folder /remote/destination --initial-host james
    
    # Propagate SSH keys first to ensure connectivity:
    python remote.py ring-sync /path/to/folder /remote/destination --propagate-keys
    
    # Use verbose output:
    python remote.py ring-sync /path/to/folder /remote/destination --verbose
    
Note: This syncs files sequentially through the ring topology defined in hosts.json
                """
        )
        parser_ring_sync.add_argument(
                "source",
                help="Source path to sync (local or on initial-host)"
        )
        parser_ring_sync.add_argument(
                "destination",
                help="Destination path on all remote hosts"
        )
        parser_ring_sync.add_argument(
                "--initial-host",
                help="Host that initially has the files (default: local machine)"
        )
        parser_ring_sync.add_argument(
                "--hosts-file",
                default="hosts.json",
                help="Path to hosts JSON file (default: hosts.json)"
        )
        parser_ring_sync.add_argument(
                "--propagate-keys",
                action="store_true",
                help="Propagate SSH keys around the ring before syncing"
        )
        parser_ring_sync.add_argument(
                "--verbose",
                action="store_true",
                help="Enable verbose output"
        )
        parser_ring_sync.add_argument(
                "--parallel",
                action="store_true",
                help="Sync to multiple nodes in parallel (faster but progress may be unclear)"
        )

        args = parser.parse_args()
        
        if args.command == "run":
                # Run the async handler
                result = asyncio.run(handle_run_command(args))
                exit(result)
        elif args.command == "ring":
                result = asyncio.run(handle_ring_command(args))
                exit(result)
        elif args.command == "restart":
                # Run the restart handler
                result = asyncio.run(handle_restart_command(args))
                exit(result)
        elif args.command == "sudo":
                # Run the sudo configuration handler
                result = asyncio.run(handle_sudo_command(args))
                exit(result)
        elif args.command == "ring-sync":
                # Run the ring sync handler
                result = asyncio.run(handle_ring_sync_command(args))
                exit(result)


if __name__ == "__main__":
        main()