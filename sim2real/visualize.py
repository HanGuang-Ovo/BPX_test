#!/usr/bin/env python3
"""只读可视化入口：python sim2real/visualize.py [--demo]。"""

import sys
from pathlib import Path

_tools = Path(__file__).resolve().parent / "tools"
sys.path.insert(0, str(_tools))

from bpx_sim2real_tools.monitor import main

if __name__ == "__main__":
    raise SystemExit(main())
