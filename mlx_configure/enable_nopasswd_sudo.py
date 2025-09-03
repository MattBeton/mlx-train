#!/usr/bin/env python3

import subprocess
import sys
import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, List

# Add parent directory to path to import shared module
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import load_config

import dotenv
dotenv.load_dotenv()


def load_hosts() -> List[str]:
    """
    Load host information from config.yaml.
    
    Returns:
        List of hostnames as strings
    """
    try:
        config = load_config()
        hosts = config.get('topology', {}).get('hosts', [])
        # Return list of hostnames, filtering out comments
        return [host for host in hosts if host and not host.startswith('#')]
    except FileNotFoundError:
        print("Error: config.yaml not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config.yaml: {e}", file=sys.stderr)
        sys.exit(1)


def enable_nopasswd_for_host(ssh_host: str, remote_user: str, rule_target: str, sudo_password: str) -> Dict:
    """
    Enable password-less sudo on a single macOS host.
    
    Args:
        ssh_host: Hostname to SSH to
        remote_user: Remote username to SSH as
        rule_target: Target for the sudo rule (username or %groupname)
        sudo_password: Admin password for sudo on remote host
        
    Returns:
        Dictionary with status information
    """
    if not ssh_host:
        return {
            "host": "<empty>",
            "success": False,
            "details": "Empty hostname provided",
        }
    
    # The remote script that will be executed
    remote_script = f'''
set -euo pipefail

# Authenticate sudo and keep session alive
echo "{sudo_password}" | sudo -S -v
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to authenticate sudo" >&2
    exit 1
fi

# Keep refreshing sudo timestamp in background
(while true; do sudo -n -v; sleep 50; done) &
SUDO_REFRESH_PID=$!
trap "kill $SUDO_REFRESH_PID 2>/dev/null || true; sudo -k" EXIT

# Sanity check: macOS only
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: Not macOS, skipping" >&2
    exit 2
fi

# 1) Ensure /etc/sudoers includes /etc/sudoers.d
if ! sudo -n grep -Eq "^[[:space:]]*#includedir[[:space:]]+/etc/sudoers\\.d[[:space:]]*$" /etc/sudoers; then
    echo "Adding #includedir /etc/sudoers.d to /etc/sudoers..."
    tmp=$(mktemp /tmp/sudoers.XXXXXX)
    sudo -n cp /etc/sudoers "$tmp"
    echo "#includedir /etc/sudoers.d" | sudo -n tee -a "$tmp" >/dev/null
    
    # Validate before applying
    if ! sudo -n visudo -cf "$tmp"; then
        echo "ERROR: Modified sudoers file failed validation" >&2
        rm -f "$tmp"
        exit 3
    fi
    
    sudo -n cp "$tmp" /etc/sudoers
    sudo -n chown root:wheel /etc/sudoers
    sudo -n chmod 0440 /etc/sudoers
    rm -f "$tmp"
fi

# 2) Create /etc/sudoers.d directory if it doesn't exist
sudo -n mkdir -p /etc/sudoers.d
sudo -n chown root:wheel /etc/sudoers.d
sudo -n chmod 0755 /etc/sudoers.d

# 3) Write rule into /etc/sudoers.d/fleet-nopasswd
echo "Creating sudo rule for {rule_target}..."
tmp=$(mktemp /tmp/sudoers_rule.XXXXXX)
echo "{rule_target} ALL=(ALL) NOPASSWD:ALL" > "$tmp"

# Validate the rule file
if ! sudo -n visudo -cf "$tmp"; then
    echo "ERROR: Sudo rule validation failed" >&2
    rm -f "$tmp"
    exit 4
fi

# Move to final location
sudo -n mv "$tmp" /etc/sudoers.d/fleet-nopasswd
sudo -n chown root:wheel /etc/sudoers.d/fleet-nopasswd
sudo -n chmod 0440 /etc/sudoers.d/fleet-nopasswd

# 4) Verify the configuration works
if sudo -n true 2>/dev/null; then
    echo "SUCCESS: Password-less sudo enabled for {rule_target} on $(hostname)"
else
    echo "ERROR: Verification failed - sudo -n check failed" >&2
    exit 5
fi
'''

    # Build SSH command
    # Just use the hostname - SSH config handles the user and connection details
    ssh_cmd = [
        "ssh",
        "-o", "BatchMode=no",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
        ssh_host,
        "bash -s"
    ]
    
    try:
        # Execute the remote script
        result = subprocess.run(
            ssh_cmd,
            input=remote_script,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=60
        )
        
        if result.returncode == 0:
            return {
                "host": ssh_host,
                "success": True,
                "details": result.stdout.strip()
            }
        else:
            return {
                "host": ssh_host,
                "success": False,
                "details": f"Exit code {result.returncode}\nSTDOUT: {result.stdout.strip()}\nSTDERR: {result.stderr.strip()}"
            }
            
    except subprocess.TimeoutExpired:
        return {
            "host": ssh_host,
            "success": False,
            "details": "Command timed out after 60 seconds"
        }
    except Exception as e:
        return {
            "host": ssh_host,
            "success": False,
            "details": f"Unexpected error: {e}"
        }


def enable_nopasswd_all_hosts(
    hosts: List[str],
    remote_user: str,
    rule_target: str,
    sudo_password: str,
    max_workers: int = 5
) -> List[Dict]:
    """
    Enable password-less sudo on all hosts in parallel.
    
    Args:
        hosts: List of hostnames as strings
        remote_user: Remote username to SSH as
        rule_target: Target for the sudo rule (username or %groupname)
        sudo_password: Admin password for sudo on remote hosts
        max_workers: Maximum number of parallel operations
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_host = {
            executor.submit(
                enable_nopasswd_for_host, 
                host, 
                remote_user, 
                rule_target, 
                sudo_password
            ): host 
            for host in hosts
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_host):
            result = future.result()
            results.append(result)
            
            # Print status for each host
            if result["success"]:
                print(f"✓ {result['host']}: {result['details']}")
            else:
                print(f"✗ {result['host']}: {result['details']}", file=sys.stderr)
    
    return results


def main():
    """Main entry point for the enable_nopasswd_sudo script."""
    parser = argparse.ArgumentParser(
        description="Enable password-less sudo on macOS remote hosts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enable for a specific user (uses hosts from config.yaml):
  %(prog)s --rule-target matt
  
  # Enable for admin group (all admin users):
  %(prog)s --rule-target %%admin
  
  # Use different SSH username:
  %(prog)s --remote-user administrator --rule-target %%admin
  
Note: Use %% for group names (e.g., %%admin, %%wheel)
        """
    )
    
    parser.add_argument(
        "--remote-user",
        default=None,
        help="Remote username to SSH as (default: current user)"
    )
    parser.add_argument(
        "--rule-target",
        default="%admin",
        help="Target for sudo rule: username or %%groupname (default: %%admin)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of parallel operations (default: 5)"
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Don't prompt for confirmation (use with caution)"
    )
    
    args = parser.parse_args()
    
    # Set remote user to current user if not specified
    if args.remote_user is None:
        import os
        args.remote_user = os.environ.get("USER", "admin")
    
    # Load hosts from config.yaml
    hosts = load_hosts()
    
    # Show what we're about to do
    print("Enable Password-less Sudo Configuration")
    print("=" * 40)
    print("Config file: config.yaml")
    print(f"Number of hosts: {len(hosts)}")
    print(f"Remote user: {args.remote_user}")
    print(f"Rule target: {args.rule_target}")
    print(f"Max parallel: {args.max_workers}")
    print("=" * 40)
    
    # Get sudo password
    sudo_password = os.environ.get("EXO_DEVICE_PASSWORD")
    if not sudo_password:
        raise Exception('Sudo password should be set as an env variable.')
    
    print(f"\nApplying configuration to {len(hosts)} hosts...")
    print("-" * 40)
    
    # Apply to all hosts
    results = enable_nopasswd_all_hosts(
        hosts,
        args.remote_user,
        args.rule_target,
        sudo_password,
        args.max_workers
    )
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print("-" * 40)
    print(f"Summary: {successful} successful, {failed} failed")
    
    if successful > 0:
        print(f"\n✅ Password-less sudo enabled on {successful} host(s)")
        print(f"   Rule: {args.rule_target} ALL=(ALL) NOPASSWD:ALL")
        print("   File: /etc/sudoers.d/fleet-nopasswd")
    
    # Exit with error if any failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()