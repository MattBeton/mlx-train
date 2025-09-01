import subprocess
import asyncio

async def check_reachability(current_host_ssh, next_host_ip, verbose=False):
    """
    SSH into the current host and pings the next host's IP.
    Returns (bool_success, message_string).
    """
    command = f"ssh {current_host_ssh} ping -c 1 -W 1 {next_host_ip}"
    try:
        if verbose:
            print(f"DEBUG: Attempting: {command}")  # Optional: Keep for debugging
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        
        if process.returncode != 0:
            error_message = stderr.decode().strip() if stderr else "No stderr"
            return (
                False,
                f"Failed to ping {next_host_ip} (CalledProcessError: {error_message})",
            )
        
        if verbose:
            print(
                f"DEBUG: Output from {current_host_ssh} pinging {next_host_ip}:\\n{stdout.decode()}"
            )  # Optional
        return True, f"Successfully pinged {next_host_ip}"
    except asyncio.TimeoutError:
        return (
            False,
            f"Timeout (10s) while attempting to ping {next_host_ip} from {current_host_ssh}",
        )
    except Exception as e:
        return (
            False,
            f"Unexpected error pinging {next_host_ip} from {current_host_ssh}: {str(e)}",
        )