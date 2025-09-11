import asyncio
import json
from typing import List


async def distributed_config_2(hosts: List[str], output_hostfile: str = 'hosts.json'):
    """
    Configure distributed MLX setup for 2 hosts by getting their bridge0 IPs.
    
    Args:
        hosts: List of hostnames to configure (expects exactly 2)
        output_hostfile: Path to output JSON file
    """
    if len(hosts) != 2:
        print(f"distributed_config_2 expects exactly 2 hosts, got {len(hosts)}")
        return 1
    
    host_configs = []
    
    for host in hosts:
        # Execute ifconfig command on remote host to get bridge0 IP
        cmd = f"ssh {host} \"ifconfig bridge0 | awk '/inet / {{print \\$2}}'\""
        print(f"Getting bridge0 IP from {host}: {cmd}")
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            print(f"Error getting IP from {host}: {stderr.decode()}")
            return 1
        
        ip = stdout.decode().strip()
        if not ip:
            print(f"No IP found for {host}")
            return 1
            
        print(f"Got IP {ip} for {host}")
        
        host_config = {
            "ssh": host,
            "ips": [ip]
        }
        host_configs.append(host_config)
    
    # Write the configuration to the output file
    with open(output_hostfile, 'w') as f:
        json.dump(host_configs, f, indent=4)
    
    print(f"Wrote configuration to {output_hostfile}")
    return 0


async def distributed_config(hosts: List[str], output_hostfile: str = 'hosts.json'):
    """
    Configure distributed MLX setup for the given hosts.
    
    Args:
        hosts: List of hostnames to configure
    """
    if len(hosts) == 2:
        return await distributed_config_2(hosts, output_hostfile)

    prefix_cmd = 'uv run mlx.distributed_config --auto-setup'
    
    cmd = f'{prefix_cmd} --output-hostfile {output_hostfile} --hosts {",".join(hosts)}'
    
    print(f"Configuring distributed setup: {cmd}")
    
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if stdout:
        print(stdout.decode())
    if stderr:
        print(stderr.decode())
    
    return process.returncode

