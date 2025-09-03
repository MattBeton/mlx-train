"""
Restart (reboot) hosts via SSH.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import load_config


def restart_host(hostname: str, timeout: int = 10) -> Dict[str, any]:
    """
    Send a reboot command to a single host via SSH.
    
    Args:
        hostname: The hostname or SSH alias to reboot
        timeout: SSH connection timeout in seconds
    
    Returns:
        Dict with 'host', 'success', and 'message' keys
    """
    try:
        # Use ssh -n to prevent reading from stdin
        # Add timeout to avoid hanging on unresponsive hosts
        cmd = [
            "ssh", "-n",
            "-o", f"ConnectTimeout={timeout}",
            "-o", "BatchMode=yes",  # Fail instead of prompting for password
            hostname,
            "sudo reboot"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5  # Give extra time for the command itself
        )
        
        if result.returncode == 0:
            return {
                "host": hostname,
                "success": True,
                "message": "Reboot command sent successfully"
            }
        else:
            return {
                "host": hostname,
                "success": False,
                "message": f"Failed with exit code {result.returncode}: {result.stderr.strip()}"
            }
    
    except subprocess.TimeoutExpired:
        return {
            "host": hostname,
            "success": False,
            "message": f"SSH connection timed out after {timeout} seconds"
        }
    except Exception as e:
        return {
            "host": hostname,
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        }


def restart_all_hosts(hosts: List[str], max_workers: int = 5) -> List[Dict]:
    """
    Restart multiple hosts in parallel.
    
    Args:
        hosts: List of hostnames to restart
        max_workers: Maximum number of parallel SSH connections
    
    Returns:
        List of result dictionaries for each host
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all restart tasks
        future_to_host = {
            executor.submit(restart_host, host): host 
            for host in hosts
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_host):
            result = future.result()
            results.append(result)
            
            # Print progress
            status = "✓" if result["success"] else "✗"
            print(f"{status} {result['host']}: {result['message']}")
    
    return results


def load_hosts_from_json(filepath: Path = Path("hosts.json")) -> List[str]:
    """
    Load host SSH aliases from hosts.json file.
    
    Args:
        filepath: Path to the hosts.json file
    
    Returns:
        List of SSH aliases
    """
    if not filepath.exists():
        raise FileNotFoundError(f"hosts.json not found at {filepath}")
    
    with open(filepath, 'r') as f:
        hosts_data = json.load(f)
    
    # Extract SSH aliases, skipping any entries without 'ssh' field
    hosts = []
    for entry in hosts_data:
        if isinstance(entry, dict) and 'ssh' in entry and entry['ssh']:
            hosts.append(entry['ssh'])
    
    return hosts


def main():
    """
    Main entry point for restart script.
    Can be called with a specific hostname or will use all hosts from hosts.json.
    """
    if len(sys.argv) == 2:
        # Single host provided
        hostname = sys.argv[1]
        print(f"Attempting to reboot {hostname}...")
        result = restart_host(hostname)
        
        if result["success"]:
            print(f"{hostname}: {result['message']}")
            sys.exit(0)
        else:
            print(f"{hostname}: {result['message']}", file=sys.stderr)
            sys.exit(1)
    
    else:
        try:
            hosts = load_config()['topology']['hosts']
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        
        if not hosts:
            print("No hosts found in config.yaml", file=sys.stderr)
            sys.exit(1)
        
        print(f"No specific host provided. Rebooting all {len(hosts)} hosts from config.yaml...")
        print("-" * 40)
        
        results = restart_all_hosts(hosts)
        
        # Summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        print("-" * 40)
        print(f"Summary: {successful} successful, {failed} failed")
        
        if failed > 0:
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()