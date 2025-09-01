#!/usr/bin/env python3

import json
import subprocess
import os
import sys
import shutil
import argparse
from pathlib import Path
import concurrent.futures

GITIGNORE_FILE = '.gitignore'
REQUIREMENTS_FILE = 'requirements.txt'  # Keep for backwards compatibility
PYPROJECT_FILE = 'pyproject.toml'
UV_LOCK_FILE = 'uv.lock'

def check_rsync():
  """Check if rsync is installed and available in PATH."""
  if not shutil.which("rsync"):
    print(
      "Error: 'rsync' command not found. Please install rsync.", file=sys.stderr
    )
    sys.exit(1)

def load_hosts(filename):
  """Load host information from the JSON file."""
  try:
    with open(filename, "r") as f:
      return json.load(f)
  except FileNotFoundError:
    print(f"Error: Hosts file '{filename}' not found.", file=sys.stderr)
    sys.exit(1)
  except json.JSONDecodeError:
    print(f"Error: Could not parse JSON from '{filename}'.", file=sys.stderr)
    sys.exit(1)


def rsync(host: str, destination_path_str: str):
  """Sync files to a single host using rsync and set up environment with uv. Returns a dict with status."""
  ssh_host = host

  remote_dest = f"{ssh_host}:{destination_path_str}/"
  source_dir = "./"

  cmd_rsync = [
    "rsync",
    "-az",  # Archive, compress. (No -v for less rsync verbosity)
    "--delete",
    "--exclude=.git",
    "--exclude=mlx*/",
    "--exclude=bench*/",
    "--exclude=cache*/",
  ]

  if os.path.exists(GITIGNORE_FILE):
    cmd_rsync.append(f"--exclude-from={GITIGNORE_FILE}")
  
  cmd_rsync.extend([source_dir, remote_dest])

  try:
    subprocess.run(
      cmd_rsync, check=True, capture_output=True, text=True, encoding='utf-8'
    )
  except subprocess.CalledProcessError as e:
    error_details = (
      f"Rsync failed.\n"
      f"  Command: {' '.join(e.cmd)}\n"
      f"  Return code: {e.returncode}\n"
      f"  STDOUT: {e.stdout.strip()}\n"
      f"  STDERR: {e.stderr.strip()}"
    )
    return {"host": ssh_host, "success": False, "details": error_details}
  except Exception as e:
    return {
      "host": ssh_host,
      "success": False,
      "details": f"Rsync unexpected error: {e}",
    }

  # After successful sync, set up environment with uv on remote host
  try:
    # Check if uv is installed on remote host (try both common locations)
    check_uv_cmd = [
      "ssh",
      ssh_host,
      "command -v uv || test -f $HOME/.cargo/bin/uv && echo $HOME/.cargo/bin/uv || test -f $HOME/.local/bin/uv && echo $HOME/.local/bin/uv",
    ]
    
    try:
      result = subprocess.run(
        check_uv_cmd, check=True, capture_output=True, text=True, encoding='utf-8'
      )
      # uv is already installed, figure out where it is
      uv_location = result.stdout.strip()
      if '/cargo/bin/' in uv_location:
        uv_path_prefix = "export PATH=\"$HOME/.cargo/bin:$PATH\" && "
      elif '/.local/bin/' in uv_location:
        uv_path_prefix = "export PATH=\"$HOME/.local/bin:$PATH\" && "
      else:
        uv_path_prefix = ""
    except subprocess.CalledProcessError:
      # If uv is not installed, try to install it
      print(f"  Installing uv on {ssh_host}...")
      install_uv_cmd = [
        "ssh",
        ssh_host,
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
      ]
      subprocess.run(
        install_uv_cmd, check=True, capture_output=True, text=True, encoding='utf-8'
      )
      # After installation, uv is typically in ~/.cargo/bin or ~/.local/bin
      # Try to detect which one
      detect_uv_cmd = [
        "ssh",
        ssh_host,
        "test -f $HOME/.cargo/bin/uv && echo cargo || test -f $HOME/.local/bin/uv && echo local || echo unknown",
      ]
      result = subprocess.run(
        detect_uv_cmd, check=True, capture_output=True, text=True, encoding='utf-8'
      )
      location = result.stdout.strip()
      if location == "cargo":
        uv_path_prefix = "export PATH=\"$HOME/.cargo/bin:$PATH\" && "
      elif location == "local":
        uv_path_prefix = "export PATH=\"$HOME/.local/bin:$PATH\" && "
      else:
        # Try both common locations
        uv_path_prefix = "export PATH=\"$HOME/.cargo/bin:$HOME/.local/bin:$PATH\" && "

    # Check if pyproject.toml and uv.lock exist on remote
    remote_pyproject_path = os.path.join(destination_path_str, PYPROJECT_FILE)
    remote_uvlock_path = os.path.join(destination_path_str, UV_LOCK_FILE)
    
    check_files_cmd = [
      "ssh",
      ssh_host,
      f"test -f {remote_pyproject_path} && test -f {remote_uvlock_path}",
    ]
    subprocess.run(
      check_files_cmd, check=True, capture_output=True, text=True, encoding='utf-8'
    )

    # Use uv sync to set up the environment and install dependencies
    # First verify uv is accessible with our path
    verify_uv_cmd = [
      "ssh",
      ssh_host,
      f"{uv_path_prefix}which uv && uv --version",
    ]
    try:
      result = subprocess.run(
        verify_uv_cmd, check=True, capture_output=True, text=True, encoding='utf-8'
      )
      print(f"  uv found on {ssh_host}: {result.stdout.strip().split()[0]}")
    except subprocess.CalledProcessError as e:
      raise Exception(f"uv not accessible after installation. PATH prefix: {uv_path_prefix}")
    
    # Now run uv sync
    sync_cmd = [
      "ssh",
      ssh_host,
      f"{uv_path_prefix}cd {destination_path_str} && uv sync",
    ]
    subprocess.run(sync_cmd, check=True, capture_output=True, text=True, encoding='utf-8')

  except subprocess.CalledProcessError as e:
    error_details = (
      f"Environment setup with uv failed.\n"
      f"  Command: {' '.join(e.cmd)}\n"
      f"  Return code: {e.returncode}\n"
      f"  STDOUT: {e.stdout.strip()}\n"
      f"  STDERR: {e.stderr.strip()}"
    )
    return {"host": ssh_host, "success": False, "details": error_details}
  except Exception as e: # Catches other errors during uv setup
    return {
      "host": ssh_host,
      "success": False,
      "details": f"Environment setup with uv unexpected error: {e}",
    }

  return {
    "host": ssh_host,
    "success": True,
    "details": "Successfully synced and set up environment with uv.",
  }


def sync_all_hosts(hosts: list[str], destination_path: str, max_workers: int = 5):
  """Sync to all hosts in parallel."""
  results = []
  
  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all rsync tasks
    future_to_host = {
      executor.submit(rsync, host, destination_path): host 
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
  """Main entry point for the rsync script."""
  parser = argparse.ArgumentParser(
    description="Sync project files to remote hosts and set up uv environment"
  )
  parser.add_argument(
    "--hosts-file",
    default="hosts.json",
    help="Path to the hosts JSON file (default: hosts.json)"
  )
  parser.add_argument(
    "--destination",
    default="/Users/Shared/mlx-train",
    help="Destination path on remote hosts (default: /Users/Shared/mlx-train)"
  )
  parser.add_argument(
    "--max-workers",
    type=int,
    default=5,
    help="Maximum number of parallel sync operations (default: 5)"
  )
  
  args = parser.parse_args()
  
  # Check rsync is available
  check_rsync()
  
  # Load hosts
  hosts = load_hosts(args.hosts_file)
  
  # Expand destination path
  destination = str(Path(args.destination).expanduser())
  
  print(f"Syncing to {len(hosts)} hosts...")
  print(f"Destination: {destination}")
  print(f"Using uv for dependency management")
  print("-" * 40)
  
  # Sync to all hosts
  results = sync_all_hosts(hosts, destination, args.max_workers)
  
  # Summary
  successful = sum(1 for r in results if r["success"])
  failed = len(results) - successful
  
  print("-" * 40)
  print(f"Summary: {successful} successful, {failed} failed")
  
  # Exit with error if any failed
  if failed > 0:
    sys.exit(1)


if __name__ == "__main__":
  main()