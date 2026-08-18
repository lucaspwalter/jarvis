#!/usr/bin/env python3
"""Terminal confirmation for public Jarvis mode."""
from __future__ import annotations

import json
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 3:
        return 2
    try:
        action = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        return 2
    if not isinstance(action, list) or not all(isinstance(item, str) for item in action):
        return 2
    description = sys.argv[2]
    print("Jarvis solicita autorização:")
    print(f"  {description}")
    print("  Ação: " + " ".join(action))
    answer = input("Permitir? [y/N] ").strip().lower()
    if answer not in {"y", "yes", "s", "sim"}:
        print("Ação cancelada.")
        return 1
    try:
        subprocess.Popen(action, start_new_session=True)
    except OSError as error:
        print(f"Falha ao executar: {error}")
        input("Enter para fechar...")
        return 3
    print("Ação autorizada.")
    input("Enter para fechar...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
