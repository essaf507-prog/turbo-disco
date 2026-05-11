"""Lightweight code review runner for this repository."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, required: bool = True) -> bool:
    print(f"\n$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode == 0:
        print("PASS")
        return True
    label = "FAIL" if required else "WARN"
    print(f"{label}: command exited with {completed.returncode}")
    return not required


def main() -> int:
    checks = [
        run([sys.executable, "-m", "compileall", "paper_style_pipeline", "scripts", "tests"]),
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests"]),
    ]
    if shutil.which("ruff"):
        checks.append(run(["ruff", "check", "."]))
    else:
        print("\nWARN: ruff is not installed; skipping lint check.")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
