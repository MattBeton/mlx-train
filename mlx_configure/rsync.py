#!/usr/bin/env python3

import json
import subprocess
import os
import sys
import shutil
import argparse
from pathlib import Path
import concurrent.futures
import re

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

def get_custom_packages(pyproject_path):
  """Extract custom package names from [tool.uv.sources] in pyproject.toml."""
  if not os.path.exists(pyproject_path):
    return []
  
  try:
    with open(pyproject_path, 'r') as f:
      content = f.read()
    
    # Find the [tool.uv.sources] section
    sources_match = re.search(r'\[tool\.uv\.sources\](.*?)(?:\n\[|\Z)', content, re.DOTALL)
    if not sources_match:
      return []
    
    sources_section = sources_match.group(1)
    
    # Extract package names from lines like: package-name = { git = "..." }
    packages = []
    for line in sources_section.split('\n'):
      line = line.strip()
      if '=' in line and '{' in line:
        package_name = line.split('=')[0].strip()
        if package_name:
          packages.append(package_name)
    
    return packages
  except Exception as e:
    print(f"Warning: Could not parse custom packages from pyproject.toml: {e}", file=sys.stderr)
    return []

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


def rsync(host, destination_path_str: str, source_path_str: str = "./", setup_uv: bool = True):
  """Sync files to a single host using rsync and optionally set up environment with uv. Returns a dict with status."""
  
  # Handle both string hosts and dictionary hosts from hosts.json
  if isinstance(host, dict):
    ssh_host = host.get('ssh', host)
  else:
    ssh_host = host

  # Ensure paths are strings
  source_path_str = str(source_path_str)
  destination_path_str = str(destination_path_str)

  # Ensure proper trailing slash for directories
  if os.path.isdir(source_path_str):
    if not source_path_str.endswith('/'):
      source_path_str += '/'
  
  # Add trailing slash to remote dest if source is a directory
  if source_path_str.endswith('/'):
    if not destination_path_str.endswith('/'):
      destination_path_str += '/'
  
  remote_dest = f"{ssh_host}:{destination_path_str}"
  
  # Create parent directory on remote if it doesn't exist
  # Use shell expansion for paths starting with ~
  parent_dir = os.path.dirname(destination_path_str.rstrip('/'))
  if parent_dir and parent_dir != '/':
    # Use bash -c to ensure proper tilde expansion on remote
    mkdir_cmd = ["ssh", ssh_host, "bash", "-c", f"mkdir -p {parent_dir}"]
    try:
      subprocess.run(mkdir_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
    except subprocess.CalledProcessError:
      # Directory might already exist or we don't have permissions, continue anyway
      pass
  
  cmd_rsync = [
    "rsync",
    "-az",  # Archive, compress. (No -v for less rsync verbosity)
    "--delete",
  ]
  
  # Only apply project-specific excludes when syncing current directory
  if source_path_str in ["./", "."]:
    cmd_rsync.extend([
      "--exclude=.git",
      "--exclude=bench*/",
      "--exclude=cache*/",
    ])
    
    if os.path.exists(GITIGNORE_FILE):
      cmd_rsync.append(f"--exclude-from={GITIGNORE_FILE}")
  
  cmd_rsync.extend([source_path_str, remote_dest])

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

  # After successful sync, optionally set up environment with uv on remote host
  if not setup_uv:
    return {
      "host": ssh_host,
      "success": True,
      "details": "Successfully synced files.",
    }
  
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
    except subprocess.CalledProcessError:
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


def sync_all_hosts(hosts: list, destination_path: str, source_path: str = "./", 
                   setup_uv: bool = True, max_workers: int = 5):
  """Sync to all hosts in parallel."""
  print(f"DEBUG sync_all_hosts: dest={destination_path}, src={source_path}, setup_uv={setup_uv}, max_workers={max_workers}")
  results = []
  
  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all rsync tasks
    future_to_host = {
      executor.submit(rsync, host, destination_path, source_path, setup_uv): host 
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
    description="Sync files to remote hosts and optionally set up uv environment"
  )
  parser.add_argument(
    "--hosts-file",
    default="hosts.json",
    help="Path to the hosts JSON file (default: hosts.json)"
  )
  parser.add_argument(
    "--source",
    default="./",
    help="Source path to sync from (default: ./ - current directory)"
  )
  parser.add_argument(
    "--destination",
    default="/Users/Shared/mlx-train",
    help="Destination path on remote hosts (default: /Users/Shared/mlx-train)"
  )
  parser.add_argument(
    "--no-uv",
    action="store_true",
    help="Skip uv environment setup (useful for non-Python projects or general file sync)"
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
  
  # Expand source path locally only
  source = str(Path(args.source).expanduser())
  
  # For destination: if it starts with ~, keep it for remote expansion
  # If it's already expanded to a user home path, convert it back to ~
  if args.destination.startswith('~'):
    destination = args.destination
  elif args.destination.startswith(str(Path.home())):
    # Convert absolute home path back to ~ for remote expansion
    destination = '~' + args.destination[len(str(Path.home())):]
  else:
    # Keep absolute paths as-is
    destination = args.destination
  setup_uv = not args.no_uv
  
  print(f"Syncing to {len(hosts)} hosts...")
  print(f"Source: {source}")
  print(f"Destination: {destination}")
  if setup_uv:
    print("Using uv for dependency management")
  else:
    print("Skipping uv setup (general file sync)")
  print("-" * 40)
  
  # Sync to all hosts
  results = sync_all_hosts(hosts, destination, source, setup_uv, args.max_workers)
  
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