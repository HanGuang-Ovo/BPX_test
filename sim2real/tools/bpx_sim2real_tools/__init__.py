"""独立的只读窗口、诊断、录制与离线回放工具。"""

import sys
from pathlib import Path

# 支持直接从源码运行 tools/run.py，以及已安装的包。
_sim2real = Path(__file__).resolve().parents[2]
if (_sim2real / "bpx_sim2real").is_dir() and str(_sim2real) not in sys.path:
    sys.path.insert(0, str(_sim2real))
