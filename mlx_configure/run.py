
import asyncio
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict


async def run(python_cmd: str, hostfile: Optional[str] = None):
    """
    Run distributed MLX training using mlx.launch.
    
    Args:
        python_cmd: The Python command/script to run
        hostfile: Path to hostfile, defaults to hosts.json
    """
    prefix_cmd = 'uv run mlx.launch'
    
    if hostfile is None:
        hostfile = 'hosts.json'
    
    cmd = f'{prefix_cmd} --hostfile {hostfile} {python_cmd}'
    
    print(f"Executing: {cmd}")
    
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Stream output in real-time
    async def stream_output(stream, prefix):
        async for line in stream:
            print(f"{prefix}: {line.decode().rstrip()}")
    
    # Create tasks for streaming both stdout and stderr
    stdout_task = asyncio.create_task(stream_output(process.stdout, "OUT"))
    stderr_task = asyncio.create_task(stream_output(process.stderr, "ERR"))
    
    # Wait for process to complete
    await process.wait()
    
    # Ensure streaming tasks complete
    await stdout_task
    await stderr_task
    
    return process.returncode