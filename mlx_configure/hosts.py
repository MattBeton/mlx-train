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

async def check_ring(hostfile: str = 'hosts.json'):
    """
    Read hosts and ips from hosts.json. 
    From each node, ping its predecessor only
    Return true if all pings are successful (ring is complete)
    """
    import json
    
    with open(hostfile, 'r') as f:
        hosts = json.load(f)
    
    if len(hosts) < 2:
        print("Need at least 2 hosts for a ring")
        return False
    
    tasks = []
    checks_info = []
    
    for i, host in enumerate(hosts):
        current_ssh = host['ssh']
        current_ip = host['ips'][0] if host['ips'] else None
        
        if not current_ip:
            print(f"Host {current_ssh} has no IP address")
            return False
        
        prev_index = (i - 1) % len(hosts)
        prev_ip = hosts[prev_index]['ips'][0]
        
        tasks.append(check_reachability(current_ssh, prev_ip))
        checks_info.append(f"{current_ssh} -> {hosts[prev_index]['ssh']} ({prev_ip})")
    
    results = await asyncio.gather(*tasks)
    
    all_success = True
    for i, (success, message) in enumerate(results):
        if success:
            print(f"✓ {checks_info[i]}: {message}")
        else:
            print(f"✗ {checks_info[i]}: {message}")
            all_success = False
    
    if all_success:
        print("\n✓ Ring connectivity check PASSED - all nodes can reach their predecessors")
    else:
        print("\n✗ Ring connectivity check FAILED - some connections are broken")
    
    return all_success