"""BPX 从 Isaac Lab 到 MuJoCo 的 sim2sim 工具。"""

# 保留 python sim2sim/run_mujoco.py 等原有直接运行入口。
import sys
from pathlib import Path

_repository_root = str(Path(__file__).resolve().parents[2])
if _repository_root not in sys.path:
    sys.path.insert(0, _repository_root)

from .config import Sim2SimConfig, load_config

__all__ = ["Sim2SimConfig", "load_config"]
