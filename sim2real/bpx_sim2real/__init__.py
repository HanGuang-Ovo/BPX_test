"""BPX 实机部署：SDK、坐标转换、起身和策略控制。"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)
