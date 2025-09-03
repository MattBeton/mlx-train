#!/usr/bin/env python3

import json
import subprocess
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

async def load_ring_hosts(hostfile: str = 'hosts.json') -> List[Dict]:
    """Load hosts from hosts.json file."""
    try:
        with open(hostfile, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Hosts file '{hostfile}' not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from '{hostfile}'.", file=sys.stderr)
        sys.exit(1)


async def rsync_between_nodes(source_host: str, source_path: str, 
                             target_host: str, target_path: str,
                             target_ip: str = None,
                             target_user: str = None,
                             verbose: bool = False) -> Tuple[bool, str]:
    """
    Rsync files from source host to target host via SSH hop.
    Uses SSH to run rsync command on source host that pushes to target.
    
    Args:
        source_host: SSH name of source host
        source_path: Path on source host
        target_host: SSH name of target host (for display)
        target_path: Path on target host  
        target_ip: IP address of target host (for actual connection)
        verbose: Enable verbose output
    """
    # Ensure paths are strings and handle trailing slashes properly
    source_path = str(source_path)
    target_path = str(target_path)
    
    # Add trailing slash for directories to sync contents
    if not source_path.endswith('/'):
        source_path += '/'
    if not target_path.endswith('/'):
        target_path += '/'
    
    # Use IP address if provided, otherwise fall back to hostname
    # Include username if provided (for nodes that use the same username)
    if target_user:
        target_connection = f"{target_user}@{target_ip if target_ip else target_host}"
    else:
        target_connection = target_ip if target_ip else target_host
    
    # Build the rsync command to run on source host
    # Create a compound command that sets up SSH options and runs rsync
    # Use --progress for compatibility with older rsync versions
    compound_cmd = (
        f"export RSYNC_RSH='ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' && "
        f"rsync -az --delete --progress {source_path} {target_connection}:{target_path}"
    )
    
    # SSH into source host and run the compound command
    command = f"ssh {source_host} \"{compound_cmd}\""
    
    if verbose:
        print(f"DEBUG: Running: {command}")
        print(f"DEBUG: Target IP: {target_ip}, Target host: {target_host}")
    
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Collect output while showing progress
        stdout_lines = []
        stderr_lines = []
        last_progress_line = ""
        
        async def read_stream(stream, lines_list, is_stderr=False):
            while True:
                line = await stream.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                lines_list.append(line_str)
                
                # Show progress updates (rsync progress appears on stderr)
                if is_stderr and '--progress' in compound_cmd:
                    # Check if this is a progress line (contains % or shows file being transferred)
                    if '%' in line_str or 'bytes' in line_str or '/' in line_str:
                        # Overwrite the previous progress line
                        print(f"\r  {target_host}: {line_str[:80]}", end='', flush=True)
                        nonlocal last_progress_line
                        last_progress_line = line_str
        
        # Read both stdout and stderr concurrently
        await asyncio.gather(
            read_stream(process.stdout, stdout_lines, False),
            read_stream(process.stderr, stderr_lines, True)
        )
        
        # Clear the progress line if one was shown
        if last_progress_line:
            print()  # New line after progress
        
        # Wait for process to complete
        await asyncio.wait_for(process.wait(), timeout=3600)  # 1 hour timeout for large files
        
        if process.returncode != 0:
            error_msg = '\n'.join(stderr_lines) if stderr_lines else "Unknown error"
            return False, f"Rsync failed: {error_msg}"
        
        return True, "Successfully synced"
        
    except asyncio.TimeoutError:
        return False, "Timeout (1 hour) while syncing"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


async def propagate_ssh_keys(hosts: List[Dict], verbose: bool = False) -> bool:
    """
    Propagate SSH keys around the ring so each node can SSH to its successor.
    This ensures each node has the public key of its predecessor.
    """
    if len(hosts) < 2:
        print("Need at least 2 hosts for key propagation")
        return True  # Not an error, just nothing to do
    
    print("\n📝 Propagating SSH keys around the ring...")
    tasks = []
    task_info = []
    
    for i, host in enumerate(hosts):
        current_ssh = host['ssh']
        next_index = (i + 1) % len(hosts)
        next_ssh = hosts[next_index]['ssh']
        next_ip = hosts[next_index]['ips'][0] if hosts[next_index].get('ips') else None
        
        # Copy current host's public key to next host's authorized_keys
        # This allows current -> next SSH access using IP address
        task = propagate_key_to_next(current_ssh, next_ssh, next_ip, verbose)
        tasks.append(task)
        task_info.append((current_ssh, next_ssh))
    
    results = await asyncio.gather(*tasks)
    
    all_success = all(success for success, _ in results)
    
    for (current_ssh, next_ssh), (success, message) in zip(task_info, results):
        if success:
            print(f"✓ {current_ssh} -> {next_ssh}: {message}")
        else:
            print(f"✗ {current_ssh} -> {next_ssh}: {message}")
    
    return all_success


async def propagate_key_to_next(current_host: str, next_host: str, next_ip: str = None,
                               verbose: bool = False) -> Tuple[bool, str]:
    """
    Copy the public key from current host to next host's authorized_keys.
    This enables current_host to SSH into next_host using IP address.
    """
    # Get the public key from current host
    get_key_cmd = f"ssh {current_host} 'cat ~/.ssh/id_*.pub 2>/dev/null || cat ~/.ssh/exo.pub 2>/dev/null'"
    
    try:
        if verbose:
            print(f"DEBUG: Getting public key from {current_host}")
        
        process = await asyncio.create_subprocess_shell(
            get_key_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        
        if process.returncode != 0:
            # Try to generate a key if none exists
            gen_key_cmd = f"ssh {current_host} 'ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N \"\" -q'"
            
            if verbose:
                print(f"DEBUG: No key found, generating one on {current_host}")
            
            gen_process = await asyncio.create_subprocess_shell(
                gen_key_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await asyncio.wait_for(gen_process.communicate(), timeout=10)
            
            # Try to get the key again
            process = await asyncio.create_subprocess_shell(
                get_key_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            
            if process.returncode != 0:
                return False, "No SSH key found and couldn't generate one"
        
        public_key = stdout.decode().strip()
        
        if not public_key:
            return False, "Empty public key"
        
        # Add the key to next host's authorized_keys
        # Use a unique comment to avoid duplicates
        add_key_cmd = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {next_host} " \
                     f"'mkdir -p ~/.ssh && chmod 700 ~/.ssh && " \
                     f"grep -q \"{public_key}\" ~/.ssh/authorized_keys 2>/dev/null || " \
                     f"echo \"{public_key}\" >> ~/.ssh/authorized_keys && " \
                     f"chmod 600 ~/.ssh/authorized_keys'"
        
        if verbose:
            print(f"DEBUG: Adding key to {next_host}")
        
        add_process = await asyncio.create_subprocess_shell(
            add_key_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await asyncio.wait_for(add_process.communicate(), timeout=10)
        
        if add_process.returncode != 0:
            return False, "Failed to add key to authorized_keys"
        
        # Test the connection using IP address
        test_target = next_ip if next_ip else next_host
        test_cmd = f"ssh {current_host} 'ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {test_target} echo ok'"
        
        test_process = await asyncio.create_subprocess_shell(
            test_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        test_stdout, _ = await asyncio.wait_for(test_process.communicate(), timeout=10)
        
        if test_process.returncode == 0 and b"ok" in test_stdout:
            return True, "SSH key propagated successfully"
        else:
            return True, "Key added but connection test failed (may need host key acceptance)"
        
    except asyncio.TimeoutError:
        return False, "Timeout during key propagation"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


async def sync_through_ring(source_path: str, destination_path: str,
                           initial_host: Optional[str] = None,
                           hostfile: str = 'hosts.json',
                           propagate_keys: bool = False,
                           verbose: bool = False,
                           parallel: bool = False) -> int:
    """
    Sync a folder through the ring of hosts.
    
    Args:
        source_path: Local path to sync (or path on initial_host if specified)
        destination_path: Target path on all remote hosts
        initial_host: Host that has the files initially (None = local machine)
        hostfile: Path to hosts.json file
        propagate_keys: Whether to propagate SSH keys first
        verbose: Enable verbose output
    
    Returns:
        0 on success, 1 on failure
    """
    # Load hosts
    hosts = await load_ring_hosts(hostfile)
    
    if len(hosts) < 2:
        print("Need at least 2 hosts for ring sync")
        return 1
    
    # Handle path expansion for source (local expansion only when syncing from local)
    if not initial_host:
        # Expand source path locally only when syncing from local machine
        source_path = str(Path(source_path).expanduser())
    
    # For destination: if it starts with ~, keep it for remote expansion
    # If it's already expanded to a user home path, convert it back to ~
    if destination_path.startswith('~'):
        # Keep as-is for remote expansion
        remote_destination = destination_path
    elif destination_path.startswith(str(Path.home())):
        # Convert absolute home path back to ~ for remote expansion
        remote_destination = '~' + destination_path[len(str(Path.home())):]
    else:
        # Keep absolute paths as-is
        remote_destination = destination_path
    
    print("Ring Sync Configuration")
    print("=" * 40)
    print(f"Source: {source_path}")
    print(f"Destination: {remote_destination}")
    print(f"Initial host: {initial_host or 'local machine'}")
    print(f"Ring size: {len(hosts)} hosts")
    print(f"Ring order: {' -> '.join([h['ssh'] for h in hosts])} -> {hosts[0]['ssh']} (loops back)")
    print("=" * 40)
    
    # Step 1: Propagate SSH keys if requested
    if propagate_keys:
        keys_success = await propagate_ssh_keys(hosts, verbose)
        if not keys_success:
            print("\n⚠️  Warning: Some SSH key propagation failed. Ring sync may fail.")
            # Continue anyway, as some connections might still work
    
    # Step 2: Initial sync from local or initial_host to first node in ring
    print("\n📤 Initial sync to first node in ring...")
    
    first_host = hosts[0]['ssh']
    
    if initial_host:
        # Sync from initial_host to first host in ring
        if initial_host == first_host:
            print(f"✓ {first_host} already has the files (initial host)")
        else:
            first_ip = hosts[0]['ips'][0] if hosts[0].get('ips') else None
            success, message = await rsync_between_nodes(
                initial_host, source_path,
                first_host, remote_destination,
                first_ip,
                verbose
            )
            if success:
                print(f"✓ {initial_host} -> {first_host}: {message}")
            else:
                print(f"✗ {initial_host} -> {first_host}: {message}")
                return 1
    else:
        # Sync from local machine to first host
        # Create parent directory on remote if needed (use bash -c for tilde expansion)
        parent_dir = os.path.dirname(remote_destination.rstrip('/'))
        if parent_dir and parent_dir != '/' and parent_dir != '~':
            mkdir_cmd = ["ssh", first_host, "bash", "-c", f"mkdir -p {parent_dir}"]
            try:
                subprocess.run(mkdir_cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError:
                pass  # Directory might already exist
        
        # Use rsync from local to first host with progress
        # Use --progress instead of --info=progress2 for compatibility with older rsync
        rsync_cmd = [
            "rsync", "-az", "--delete", "--progress",
            str(source_path) + ("/" if os.path.isdir(source_path) else ""),
            f"{first_host}:{remote_destination}"
        ]
        
        try:
            # Run rsync combining stdout and stderr for simpler handling
            process = subprocess.Popen(
                rsync_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            last_was_progress = False
            all_output = []
            
            # Read output line by line
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                    
                line = line.strip()
                if line:
                    all_output.append(line)
                    
                    # Check if this is a progress line (--progress format)
                    if '%' in line or ('bytes' in line and '/' in line):
                        # Truncate long lines for display
                        display_line = line[:80] if len(line) > 80 else line
                        print(f"\r  local -> {first_host}: {display_line}", end='', flush=True)
                        last_was_progress = True
                    else:
                        if last_was_progress:
                            print()  # New line after progress
                            last_was_progress = False
                        
                        # Show non-progress output in verbose mode or if it's an error
                        if verbose or 'error' in line.lower() or 'failed' in line.lower():
                            print(f"  {line}")
            
            if last_was_progress:
                print()  # Final newline after progress
            
            return_code = process.wait()
            
            if return_code == 0:
                print(f"✓ local -> {first_host}: Successfully synced")
            else:
                print(f"✗ local -> {first_host}: Rsync failed with return code {return_code}")
                # Show last few lines of output for debugging
                if all_output and not verbose:
                    print("  Last output lines:")
                    for line in all_output[-5:]:
                        print(f"    {line}")
                return 1
                
        except Exception as e:
            print(f"✗ local -> {first_host}: Exception: {str(e)}")
            return 1
    
    # Step 3: Propagate through the ring
    print("\n🔄 Propagating through the ring...")
    
    if parallel:
        print("Note: Syncing in parallel (progress may be interleaved)\n")
        
        tasks = []
        sync_pairs = []
        
        for i in range(len(hosts) - 1):
            current_host = hosts[i]['ssh']
            next_host = hosts[i + 1]['ssh']
            next_ip = hosts[i + 1]['ips'][0] if hosts[i + 1].get('ips') else None
            next_user = next_host  # Use the SSH hostname as username
            
            task = rsync_between_nodes(
                current_host, remote_destination,
                next_host, remote_destination,
                next_ip,
                next_user,
                verbose
            )
            tasks.append(task)
            sync_pairs.append((current_host, next_host))
        
        # Run all ring syncs in parallel
        results = await asyncio.gather(*tasks)
        
        # Check results
        all_success = True
        for (current, next_host), (success, message) in zip(sync_pairs, results):
            if success:
                print(f"✓ {current} -> {next_host}: {message}")
            else:
                print(f"✗ {current} -> {next_host}: {message}")
                all_success = False
    else:
        print("Note: Syncing sequentially for clear progress visibility\n")
        
        all_success = True
        
        for i in range(len(hosts) - 1):
            current_host = hosts[i]['ssh']
            next_host = hosts[i + 1]['ssh']
            next_ip = hosts[i + 1]['ips'][0] if hosts[i + 1].get('ips') else None
            next_user = next_host  # Use the SSH hostname as username
            
            print(f"Syncing {current_host} -> {next_host}...")
            
            # Run sync sequentially to show progress clearly
            success, message = await rsync_between_nodes(
                current_host, remote_destination,
                next_host, remote_destination,
                next_ip,
                next_user,
                verbose
            )
            
            if success:
                print(f"✓ {current_host} -> {next_host}: {message}")
            else:
                print(f"✗ {current_host} -> {next_host}: {message}")
                all_success = False
                # Continue trying other nodes even if one fails
    
    # Summary
    if all_success:
        print("\n✅ Ring sync completed successfully!")
        print(f"   {source_path} has been synced to all {len(hosts)} nodes")
    else:
        print("\n❌ Ring sync failed - some nodes may not have received the files")
        return 1
    
    return 0


async def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sync files through a ring of hosts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
    # Sync local folder to all nodes through the ring:
    python ring_sync.py /path/to/folder /remote/destination
    
    # Sync from a specific host through the ring:
    python ring_sync.py /path/to/folder /remote/destination --initial-host james
    
    # Propagate SSH keys first to ensure connectivity:
    python ring_sync.py /path/to/folder /remote/destination --propagate-keys
    
    # Use a different hosts file:
    python ring_sync.py /path/to/folder /remote/destination --hosts-file my_hosts.json
    
    # Enable verbose output:
    python ring_sync.py /path/to/folder /remote/destination --verbose
        """
    )
    
    parser.add_argument(
        "source",
        help="Source path to sync (local or on initial-host)"
    )
    parser.add_argument(
        "destination",
        help="Destination path on all remote hosts"
    )
    parser.add_argument(
        "--initial-host",
        help="Host that initially has the files (default: local machine)"
    )
    parser.add_argument(
        "--hosts-file",
        default="hosts.json",
        help="Path to hosts JSON file (default: hosts.json)"
    )
    parser.add_argument(
        "--propagate-keys",
        action="store_true",
        help="Propagate SSH keys around the ring before syncing"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Sync to multiple nodes in parallel (faster but progress may be unclear)"
    )
    
    args = parser.parse_args()
    
    # Run the sync
    result = await sync_through_ring(
        args.source,
        args.destination,
        args.initial_host,
        args.hosts_file,
        args.propagate_keys,
        args.verbose,
        args.parallel
    )
    
    sys.exit(result)


if __name__ == "__main__":
    asyncio.run(main())