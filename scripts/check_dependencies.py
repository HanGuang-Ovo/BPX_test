#!/usr/bin/env python3
"""检查当前 Python 环境是否满足本工程的依赖与资产要求。

用途：把工程移植到新机器（或新建环境）后，先用本脚本快速定位缺什么，再按
README 的安装步骤补齐。不依赖任何第三方库，任何 Python 均可直接运行。

检查分两套：

- training（默认）：Isaac Lab 训练环境。涵盖 Python 版本、pip 包（isaaclab、
  torch、rsl-rl、tensorboard 等）、本工程的可编辑安装，以及被 .gitignore
  忽略、必须手动携带的 USD 机器人资产。
- sim2sim：MuJoCo 推理环境。涵盖 Python 3.11+（配置读取用标准库 tomllib）、
  numpy / mujoco / onnxruntime 与 MJCF 模型。

退出码：全部必需项通过为 0，否则为 1，便于在 shell 脚本中串联。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 训练环境的 pip 依赖：显示名 -> pip 包名。
TRAINING_PACKAGES = [
    ("Isaac Lab", "isaaclab"),
    ("PyTorch", "torch"),
    ("RSL-RL", "rsl-rl-lib"),
    ("Gymnasium", "gymnasium"),
    ("TensorBoard", "tensorboard"),
    ("psutil", "psutil"),
    ("toml (setup.py 构建用)", "toml"),
    ("prettytable (list_envs 用)", "prettytable"),
]

# sim2sim 环境的 pip 依赖（见 sim2sim/requirements.txt）。
SIM2SIM_PACKAGES = [
    ("NumPy", "numpy"),
    ("MuJoCo", "mujoco"),
    ("ONNX Runtime", "onnxruntime"),
]

# 机器人 USD 资产：被 .gitignore 忽略（**/*.usd），git clone 后必须手动携带。
USD_ASSETS = [
    Path("source/BPX_test/BPX_test/BPX_structure/usd/bpx.usd"),
    Path("source/BPX_test/BPX_test/BPX_structure/usd/configuration/bpx_base.usd"),
    Path("source/BPX_test/BPX_test/BPX_structure/usd/configuration/bpx_physics.usd"),
    Path("source/BPX_test/BPX_test/BPX_structure/usd/configuration/bpx_robot.usd"),
    Path("source/BPX_test/BPX_test/BPX_structure/usd/configuration/bpx_sensor.usd"),
]

MJCF_ASSET = Path("source/BPX_test/BPX_test/BPX_structure/mjcf/bpx.xml")
URDF_ASSET = Path("source/BPX_test/BPX_test/BPX_structure/bpx/urdf/bpx.urdf")
CHECKPOINTS_DIR = Path("logs/rsl_rl")


class Check:
    """单项检查结果。required=False 的项缺失只提示不算失败。"""

    def __init__(self, name: str, ok: bool, detail: str, required: bool = True, fix: str = ""):
        self.name = name
        self.ok = ok
        self.detail = detail
        self.required = required
        self.fix = fix


def display_width(text: str) -> int:
    """终端显示宽度：CJK 字符按 2 列计，用于对齐输出。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(width - display_width(text), 0)


def check_python(min_version: tuple[int, int], note: str) -> Check:
    if sys.version_info >= min_version:
        return Check("Python 版本", True, f"{sys.version.split()[0]}（{note}）")
    return Check(
        "Python 版本", False, f"{sys.version.split()[0]}，需要 >={'.'.join(map(str, min_version))}（{note}）",
        fix=f"换用 Python {'.'.join(map(str, min_version))}+ 的环境",
    )


def check_package(display: str, pip_name: str, fix: str | None = None) -> Check:
    try:
        version = importlib.metadata.version(pip_name)
        return Check(display, True, version)
    except importlib.metadata.PackageNotFoundError:
        pass
    # pip 元数据缺失不等于没装：回退到实际导入（PYTHONPATH 注入的包无元数据）。
    module_name = IMPORT_NAMES.get(pip_name, pip_name.replace("-", "_"))
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - ImportError 及被注入包自身的加载错误都按未安装处理
        return Check(display, False, "未安装", fix=fix or f"python -m pip install {pip_name}")
    version = getattr(module, "__version__", "未知版本")
    return Check(display, True, f"{version}（PYTHONPATH 注入，无 pip 元数据）")


# 这些包不应直接 pip 安装，而是随 Isaac Lab 环境一起装，修复建议单独给出。
ISAACLAB_FIX = "安装 Isaac Lab v2.2.1 并执行 ./isaaclab.sh -i（见 README「环境准备」），勿直接 pip install"
TORCH_FIX = "torch 随 Isaac Sim / Isaac Lab 环境自带，请先确认用的是该环境的 Python"

# pip 包名 -> import 模块名：Isaac Sim 的 torch/isaacsim 经 conda 激活脚本以
# PYTHONPATH 注入（pip_prebundle 目录），isaacsim 没有 pip 元数据，需实际导入验证。
IMPORT_NAMES = {"rsl-rl-lib": "rsl_rl", "onnxruntime": "onnxruntime"}


def check_bpx_install() -> Check:
    """确认 BPX_test 已可编辑安装，且指向当前仓库（而不是旧机器上的残留路径）。"""
    spec = importlib.util.find_spec("BPX_test")
    if spec is None:
        return Check(
            "BPX_test 本体", False, "未安装（未注册到当前 Python 环境）",
            fix=f"cd {PROJECT_ROOT} && python -m pip install -e source/BPX_test",
        )
    locations = " ".join(str(p) for p in (spec.submodule_search_locations or []))
    source_dir = (PROJECT_ROOT / "source" / "BPX_test").resolve()
    if source_dir.as_posix() not in Path(locations).as_posix():
        return Check(
            "BPX_test 本体", False, f"已安装但指向别处：{locations}",
            fix="在当前仓库目录重新执行 python -m pip install -e source/BPX_test",
        )
    return Check("BPX_test 本体", True, f"可编辑安装，指向本仓库（{locations}）")


def check_assets(assets: list[Path], label: str, fix: str) -> list[Check]:
    checks = []
    for rel in assets:
        path = PROJECT_ROOT / rel
        if path.is_file():
            size_mb = path.stat().st_size / 2**20
            checks.append(Check(f"{label}：{rel.name}", True, f"{size_mb:.1f} MB"))
        else:
            checks.append(Check(f"{label}：{rel.name}", False, "文件缺失", fix=fix))
    return checks


def check_gpu() -> Check:
    """Isaac Sim 训练必须有 NVIDIA GPU；缺失时给出提示（sim2sim 不受影响）。"""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return Check(
            "NVIDIA GPU / 驱动", False, "找不到 nvidia-smi 命令",
            fix="安装 NVIDIA 显卡驱动（Isaac Sim 训练必需；纯 MuJoCo 推理可忽略）",
        )
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        info = result.stdout.strip().splitlines()[0] if result.stdout.strip() else "未知型号"
    except (subprocess.SubprocessError, IndexError):
        info = "无法查询（nvidia-smi 存在但调用失败）"
    return Check("NVIDIA GPU / 驱动", True, info, required=False)


def check_torch_cuda() -> Check:
    """Torch 已装时顺带报告 CUDA 可用性；只提示，不算失败。"""
    try:
        import torch  # noqa: PLC0415
        return Check(
            "PyTorch CUDA", torch.cuda.is_available(),
            f"device_count={torch.cuda.device_count()}",
            required=False,
            fix="torch.cuda.is_available() 为 False 时，请检查显卡驱动与 CUDA 版本",
        )
    except Exception:  # noqa: BLE001 - torch 未装时由包检查项负责报错
        return Check("PyTorch CUDA", False, "跳过（torch 未安装）", required=False)


def check_tomllib() -> Check:
    if sys.version_info >= (3, 11):
        return Check("tomllib（标准库）", True, "可用")
    return Check(
        "tomllib（标准库）", False, "Python < 3.11，无 tomllib",
        fix="sim2sim 配置读取需要 Python 3.11+，请重建该 conda 环境",
    )


def run_checks(mode: str) -> int:
    print(f"工程根目录：{PROJECT_ROOT}")
    print(f"Python    ：{sys.executable}（{sys.version.split()[0]}）")
    print(f"检查模式  ：{mode}\n")

    checks: list[Check] = []
    if mode in ("training", "all"):
        checks.append(check_python((3, 10), "本工程要求 >= 3.10"))
        checks += [
            check_package(d, p, ISAACLAB_FIX if p in ("isaaclab", "rsl-rl-lib") else TORCH_FIX if p == "torch" else None)
            for d, p in TRAINING_PACKAGES
        ]
        checks.append(check_bpx_install())
        checks += check_assets(
            USD_ASSETS, "USD 资产",
            "USD 被 .gitignore 忽略：从旧机器拷贝 source/BPX_test/BPX_test/BPX_structure/usd/，"
            "或按 README 用 URDF 转换工具重新生成",
        )
        checks.append(check_gpu())
        isaacsim = check_package(
            "Isaac Sim（pip 名 isaacsim）", "isaacsim",
            "随 Isaac Lab 安装（./isaaclab.sh -i 或 setup_conda.sh）；Omniverse 独立安装且训练正常时缺失属正常",
        )
        isaacsim.required = False  # Omniverse 路线下无 pip 元数据但可正常运行，只提示
        checks.append(isaacsim)
        checks.append(check_torch_cuda())
        checkpoints = sorted(p.name for p in (PROJECT_ROOT / CHECKPOINTS_DIR).glob("*/")) if (PROJECT_ROOT / CHECKPOINTS_DIR).is_dir() else []
        checks.append(Check(
            "训练检查点（可选）", bool(checkpoints), f"logs/rsl_rl 下 {len(checkpoints)} 个实验目录",
            required=False, fix="如需在新机器续训/回放，从旧机器拷贝 logs/rsl_rl/",
        ))
    if mode in ("sim2sim", "all"):
        checks.append(check_python((3, 11), "sim2sim 配置读取用 tomllib"))
        checks.append(check_tomllib())
        checks += [check_package(d, p) for d, p in SIM2SIM_PACKAGES]
        checks += check_assets([MJCF_ASSET, URDF_ASSET], "MJCF/URDF", "该文件由 git 跟踪，请检查仓库完整性")
        torch_optional = check_package("PyTorch（仅 TorchScript 需要）", "torch")
        torch_optional.required = False  # ONNX 后端不需要，README 已注明
        checks.append(torch_optional)

    width = max(display_width(c.name) for c in checks) + 2
    failed_required: list[Check] = []
    for c in checks:
        mark = "✓" if c.ok else ("!" if not c.required else "✗")
        print(f"  [{mark}] {pad(c.name, width)}{c.detail}")
        if not c.ok and c.required:
            failed_required.append(c)

    print()
    if failed_required:
        print("缺失的必需项及修复方法：")
        for c in failed_required:
            print(f"  - {c.name}：{c.fix or '见 README「环境准备」一节'}")
        prefix = Path(sys.prefix)
        if "envs" in prefix.parts and os.environ.get("CONDA_PREFIX") != str(prefix):
            print(f"\n  提示：当前解释器属于 conda 环境「{prefix.name}」，但并未通过 conda activate 激活。")
            print("  torch/isaacsim 实际位于 Isaac Sim 目录，由激活脚本经 PYTHONPATH 注入——")
            print(f"  请先 conda activate {prefix.name} 再运行本脚本，否则结果会误报缺失。")
        print(f"\n结论：{len(failed_required)} 项必需依赖缺失（! 为可选项，不影响结论）。")
        return 1
    print("结论：必需依赖全部就绪（! 为可选项）。")
    if mode in ("training", "all"):
        print("下一步验证：python scripts/list_envs.py")
    if mode in ("sim2sim", "all"):
        print("下一步验证：python sim2sim/validate_setup.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode", choices=("training", "sim2sim", "all"), default="training",
        help="检查哪套环境：training=Isaac Lab 训练环境（默认），sim2sim=MuJoCo 推理环境，all=两者",
    )
    args = parser.parse_args()
    return run_checks(args.mode)


if __name__ == "__main__":
    sys.exit(main())
