#!/usr/bin/env python3
"""Read-only Linux smoke gate for a loopback-only single-user Telly host."""
from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen


def listener_hosts(ss_output: str, port: int) -> list[str]:
    hosts: list[str] = []
    for line in ss_output.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0] != "LISTEN":
            continue
        local = fields[3]
        host, separator, found_port = local.rpartition(":")
        if separator and found_port == str(port):
            hosts.append(host.strip("[]"))
    return hosts


def loopback_only(hosts: list[str]) -> bool:
    if not hosts:
        return False
    try:
        return all(ipaddress.ip_address(host).is_loopback for host in hosts)
    except ValueError:
        return False


def check_storage(path: Path, checkout: Path) -> bool:
    if path.is_symlink() or not path.is_dir() or path.stat().st_mode & 0o077:
        return False
    try:
        path.resolve().relative_to(checkout.resolve())
    except ValueError:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")

    try:
        output = subprocess.run(
            ["ss", "-H", "-ltn"], capture_output=True, text=True, check=True, timeout=5
        ).stdout
        hosts = listener_hosts(output, args.port)
        listener_ok = loopback_only(hosts)
    except (OSError, subprocess.SubprocessError):
        hosts, listener_ok = [], False
    try:
        with urlopen(f"http://127.0.0.1:{args.port}/_stcore/health", timeout=5) as response:
            health_ok = response.status == 200 and response.read(64).strip() == b"ok"
    except (OSError, ValueError):
        health_ok = False
    storage_ok = check_storage(args.storage, args.checkout)
    result = {
        "ready": listener_ok and health_ok and storage_ok,
        "checks": {
            "loopback_listener": listener_ok,
            "health": health_ok,
            "persistent_storage": storage_ok,
        },
        "listener_hosts": hosts,
        "scope": "host-local checks only; SSH account restriction and external denial require separate verification",
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
