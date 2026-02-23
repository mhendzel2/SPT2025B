#!/usr/bin/env python3
"""Run the Streamlit app via the package entry path."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [sys.executable, "-m", "streamlit", "run", "spt2025b/ui/app.py", *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
