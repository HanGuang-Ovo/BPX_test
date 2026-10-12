"""BPX 实机只读对齐、回放和策略控制。SDK 仅在硬件入口中加载。"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)
