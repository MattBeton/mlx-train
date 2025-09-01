import asyncio
import subprocess
from typing import List


async def distributed_config(hosts: List[str], output_hostfile: str = 'hosts.json'):
    """
    Configure distributed MLX setup for the given hosts.
    
    Args:
        hosts: List of hostnames to configure
    """
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

