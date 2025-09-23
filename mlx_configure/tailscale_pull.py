#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Usage:
#   TAG=tag:applepark ./export_tailscale_tagged.py
# or:
#   ./export_tailscale_tagged.py tag:applepark
#
# Output: JSON array to stdout, e.g.:
# [
#   {"name":"a1s-Mac-Studio","username":"a1","ips":["100.x.y.z","fd7a:..."]},
#   ...
# ]
#
# Also updates ~/.ssh/config with Host entries for each device

def update_ssh_config(devices):
    """Update SSH config with entries for each device."""
    ssh_config_path = Path.home() / ".ssh" / "config"

    # Create .ssh directory if it doesn't exist
    ssh_dir = ssh_config_path.parent
    ssh_dir.mkdir(mode=0o700, exist_ok=True)

    # Read existing config
    existing_config = ""
    if ssh_config_path.exists():
        existing_config = ssh_config_path.read_text()

    # Find and remove existing managed block
    start_marker = "# === TAILSCALE MANAGED START ==="
    end_marker = "# === TAILSCALE MANAGED END ==="

    if start_marker in existing_config and end_marker in existing_config:
        before = existing_config.split(start_marker)[0]
        after_parts = existing_config.split(end_marker)
        if len(after_parts) > 1:
            after = after_parts[1]
        else:
            after = ""
        existing_config = before.rstrip() + "\n" + after.lstrip()

    # Generate new config entries
    config_entries = [f"\n{start_marker}"]
    for device in devices:
        username = device["username"]
        # Use IPv4 address (first IP)
        if device["ips"]:
            ip = device["ips"][0]
            config_entries.append(f"""
Host {username}
    HostName {ip}
    User {username}
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null""")

    config_entries.append(f"{end_marker}\n")

    # Write updated config
    new_config = existing_config.rstrip() + "\n" + "\n".join(config_entries)
    ssh_config_path.write_text(new_config)
    ssh_config_path.chmod(0o600)

    print(f"Updated SSH config at {ssh_config_path}", file=sys.stderr)
    print(f"Added {len(devices)} host entries", file=sys.stderr)

def main():
    tag = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TAG"))
    if not tag:
        tag = 'tag:applepark'
        # print("ERROR: provide a tag as ARG or TAG env (e.g. tag:applepark)", file=sys.stderr)
        # sys.exit(2)

    # Call `tailscale status --json`
    try:
        out = subprocess.check_output(["tailscale", "status", "--json"])
    except FileNotFoundError:
        print("ERROR: tailscale CLI not found in PATH", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: tailscale status failed: {e}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(out)

    # Remote devices live under "Peer" (map keyed by PublicKey)
    peers = data.get("Peer") or {}
    results = []

    # case-insensitive match for a{i}s-Mac-Studio
    # capture just the digits (i) so username becomes "a{i}"
    host_rx = re.compile(r"^a(?P<i>\d+)s-?mac-?studio$", re.IGNORECASE)

    for peer_obj in peers.values():
        tags = peer_obj.get("Tags") or []
        if tag not in tags:
            continue

        # Prefer HostName; fall back to first label of DNSName
        name = peer_obj.get("HostName")
        if not name:
            dns = (peer_obj.get("DNSName") or "").split(".")[0]
            name = dns

        if not name:
            continue  # skip if we still don't have a name

        m = host_rx.match(name)
        if not m:
            continue

        username = f"a{m.group('i')}"
        ips = peer_obj.get("TailscaleIPs") or []

        results.append({
            "name": name,
            "username": username,
            "ips": ips,
        })

    print(json.dumps(results, indent=2))

    # Update SSH config
    if results:
        update_ssh_config(results)

    # with open('tailscale.json', 'w') as f:
    #     f.write(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

