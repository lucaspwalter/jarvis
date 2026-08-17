#!/usr/bin/env python3
"""Type a voice payload into an already open Codex Kitty window."""
from __future__ import annotations

import json
import os
import subprocess
import sys


def descendants(root: int) -> list[int]:
    found = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            stat = open(os.path.join(entry.path, "stat"), encoding="ascii").read().split()
            if int(stat[3]) == root:
                child = int(entry.name)
                found.append(child)
                found.extend(descendants(child))
        except (OSError, ValueError, IndexError):
            pass
    return found


def command_line(pid: int) -> str:
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def main() -> int:
    payload = " ".join(sys.argv[1:]).strip()
    if not payload:
        return 2
    clients = json.loads(subprocess.check_output(["hyprctl", "clients", "-j"], text=True))
    for client in clients:
        if client.get("class", "").lower() != "kitty":
            continue
        pids = [int(client["pid"])] + descendants(int(client["pid"]))
        if not any("codex" in command_line(pid).lower() for pid in pids):
            continue
        address = client["address"]
        # Hyprland Lua dispatcher requires the address matcher prefix.
        subprocess.run(
            ["hyprctl", "dispatch", f'hl.dsp.focus({{ window = "address:{address}" }})'],
            check=False,
        )
        subprocess.run(["wtype", payload], check=True)
        subprocess.run(["wtype", "\n"], check=True)
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
