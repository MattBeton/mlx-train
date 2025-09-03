#!/usr/bin/env python3

import subprocess
import os
import sys
import getpass
from pathlib import Path
from typing import List
from shared.config import load_config


def check_sshpass() -> bool:
    """Check if sshpass is installed on the system."""
    try:
        subprocess.run(["which", "sshpass"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        print("sshpass is not installed. Please install it first.")
        print("On macOS, you can install it using: brew install sshpass")
        print("On Ubuntu/Debian: sudo apt-get install sshpass")
        return False


def get_common_password() -> str:
    """Get the common password for all hosts from environment variable or user input."""
    # Try to get password from environment variable first
    password = os.getenv("EXO_DEVICE_PASSWORD")
    
    if password:
        print("Using password from EXO_DEVICE_PASSWORD environment variable.")
        return password
    
    # Fall back to user input if environment variable is not set
    print("EXO_DEVICE_PASSWORD not found in environment.")
    password = getpass.getpass("Enter the common password for all hosts: ")
    return password


def get_devices(config_file: str = "config.yaml") -> List[str]:
    """Read device names from the YAML configuration file using mlx_configure."""
    config = load_config(config_file)
    
    devices = config['topology']['hosts']
    
    if not devices:
        print("Error: No hosts defined in topology.hosts.")
        sys.exit(1)
    
    return devices


def setup_ssh_config(devices: List[str]) -> None:
    """Add host entries to SSH config file if they don't exist."""
    ssh_dir = Path.home() / ".ssh"
    ssh_config_file = ssh_dir / "config"
    
    # Ensure the .ssh directory exists
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    
    # Ensure the config file exists
    ssh_config_file.touch(mode=0o600, exist_ok=True)
    
    print("Checking SSH config for hosts listed in devices.txt...")
    
    existing_config = ssh_config_file.read_text() if ssh_config_file.exists() else ""
    
    for device_name in devices:
        if f"Host {device_name}" in existing_config:
            print(f"Host entry for '{device_name}' already exists in {ssh_config_file}. Skipping.")
        else:
            print(f"Adding Host entry for '{device_name}' to {ssh_config_file}...")
            
            # Append the new Host entry
            with open(ssh_config_file, 'a') as f:
                f.write("\n")
                f.write(f"Host {device_name}\n")
                f.write(f"    User {device_name}\n")
            
            print(f"Host entry for '{device_name}' added.")
    
    print("SSH config check complete.")


def generate_ssh_key() -> str:
    """Generate SSH key pair and add to ssh-agent."""
    # Get identifier from user
    key_identifier = input("Enter your name to identify SSH key (e.g., 'sethhowes' or 'satoshinakamoto'): ")
    
    ssh_key_path = Path.home() / ".ssh" / "exo"
    
    # Generate key pair
    print("Generating SSH key pair...")
    subprocess.run([
        "ssh-keygen",
        "-t", "ed25519",
        "-C", key_identifier,
        "-f", str(ssh_key_path),
        "-N", ""  # Empty passphrase
    ], check=True)
    
    # Start ssh-agent if needed and add key
    print("Adding key to ssh-agent...")
    try:
        # Start ssh-agent
        result = subprocess.run(
            ["ssh-agent", "-s"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output to get environment variables
        for line in result.stdout.split('\n'):
            if line.startswith('SSH_AUTH_SOCK='):
                os.environ['SSH_AUTH_SOCK'] = line.split('=')[1].split(';')[0]
            elif line.startswith('SSH_AGENT_PID='):
                os.environ['SSH_AGENT_PID'] = line.split('=')[1].split(';')[0]
        
        # Add the key
        subprocess.run(["ssh-add", str(ssh_key_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not add key to ssh-agent: {e}")
    
    return key_identifier


def distribute_public_key(devices: List[str], password: str) -> None:
    """Distribute public key to all hosts using sshpass."""
    ssh_key_pub = Path.home() / ".ssh" / "exo.pub"
    
    if not ssh_key_pub.exists():
        print(f"Error: Public key {ssh_key_pub} not found.")
        sys.exit(1)
    
    for device_name in devices:
        print(f"Adding public key to {device_name}...")
        
        try:
            # Use sshpass with ssh-copy-id
            subprocess.run([
                "sshpass", "-p", password,
                "ssh-copy-id",
                "-o", "StrictHostKeyChecking=no",
                "-i", str(ssh_key_pub),
                device_name
            ], check=True, capture_output=True, text=True)
            
            print(f"Successfully added public key to {device_name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to add public key to {device_name}: {e}")


def add_ssh_key_locally() -> None:
    """Add SSH key to local ssh-agent."""
    ssh_key_path = Path.home() / ".ssh" / "exo"
    
    print("Adding ssh keys to local machine.")
    try:
        subprocess.run(["ssh-add", str(ssh_key_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not add key to ssh-agent: {e}")


def main():
    """Main execution function."""
    # Check if sshpass is installed
    if not check_sshpass():
        sys.exit(1)
    
    # Read devices from file
    devices = get_devices()
    
    # Get common password
    password = get_common_password()
    
    # Setup SSH config
    print("\nAdding hosts to config file.")
    setup_ssh_config(devices)
    
    # Generate SSH key pair
    print("\nGenerating SSH key pair.")
    generate_ssh_key()
    
    # Distribute public key to hosts
    print("\nDistributing public key to hosts.")
    distribute_public_key(devices, password)
    
    # Add SSH key to local machine
    add_ssh_key_locally()
    
    print("\nSSH key setup complete!")


if __name__ == "__main__":
    main()