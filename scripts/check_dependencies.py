#!/usr/bin/env python3
"""检查当前 Python 环境是否满足本工程的依赖与资产要求。

用途：把工程移植到新机器（或新建环境）后，先用本脚本快速定位缺什么，再按
README 的安装步骤补齐。检查器只用标准库，训练库导入和 CUDA 运算在子进程中验证。

检查分两套：

- training（默认）：Isaac Lab 训练环境。涵盖 Python 版本、pip 包（isaaclab、
  torch、rsl-rl、tensorboard 等）、本工程的可编辑安装，以及被 .gitignore
  忽略、必须手动携带的 USD 机器人资产。
- sim2sim：MuJoCo 推理环境。涵盖 Python 3.11+（配置读取用标准库 tomllib）、
  numpy / mujoco / onnxruntime 与 MJCF 模型。

--smoke-test：检查通过后启动选定任务，用 4 个环境完成两轮 PPO 并检查模型保存。
退出码：所选检查全部通过为 0，失败为 1，参数错误为 2。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 训练环境的 pip 依赖：显示名 -> pip 包名。
TRAINING_PACKAGES = [
    ("Isaac Lab", "isaaclab"),
    ("Isaac Lab Assets", "isaaclab_assets"),
    ("Isaac Lab Tasks", "isaaclab_tasks"),
    ("Isaac Lab RL", "isaaclab_rl"),
    ("PyTorch", "torch"),
    ("RSL-RL", "rsl-rl-lib"),
    ("Gymnasium", "gymnasium"),
    ("TensorBoard", "tensorboard"),
    ("psutil", "psutil"),
    ("toml (setup.py 构建用)", "toml"),
    ("prettytable (list_envs 用)", "prettytable"),
    ("NumPy", "numpy"),
    ("Hydra", "hydra-core"),
    ("packaging", "packaging"),
    ("PyYAML", "PyYAML"),
    ("trimesh", "trimesh"),
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
    module_name = IMPORT_NAMES.get(pip_name, pip_name.replace("-", "_"))
    try:
        version = importlib.metadata.version(pip_name)
    except importlib.metadata.PackageNotFoundError:
        version = "无 pip 元数据（可能由 PYTHONPATH 注入）"
    # 只查顶层模块位置。不能在 AppLauncher 之前导入 BPX/Isaac 包，
    # 也不能触发 isaacsim 的交互式 EULA 提示。
    try:
        spec = importlib.util.find_spec(module_name)
        # scripts/rsl_rl/ 自身会被解释器当作同名 namespace package，
        # 不能因此把未安装的 RSL-RL 判为存在。这里检查的依赖都是常规包。
        available = spec is not None and spec.origin is not None
    except (ImportError, ValueError, AttributeError):
        available = False
    return Check(display, available, version if available else "模块不可发现（未安装或安装路径失效）",
                 fix=fix or "bash scripts/install_training_deps.sh")


# 这些包不应直接 pip 安装，而是随 Isaac Lab 环境一起装，修复建议单独给出。
ISAACLAB_FIX = "激活 Isaac Lab 环境；从零安装可运行 bash scripts/install_training_deps.sh --full"
TORCH_FIX = "torch 随 Isaac Sim / Isaac Lab 环境自带，请先确认用的是该环境的 Python"

# pip 包名 -> import 模块名：Isaac Sim 的 torch/isaacsim 经 conda 激活脚本以
# PYTHONPATH 注入（pip_prebundle 目录）；快速检查只查顶层模块位置。
IMPORT_NAMES = {"rsl-rl-lib": "rsl_rl", "hydra-core": "hydra", "PyYAML": "yaml"}


def check_bpx_install() -> Check:
    """确认 BPX_test 已可编辑安装，且指向当前仓库（而不是旧机器上的残留路径）。"""
    spec = importlib.util.find_spec("BPX_test")
    if spec is None:
        return Check(
            "BPX_test 本体", False, "未安装（未注册到当前 Python 环境）",
            fix=f"cd {PROJECT_ROOT} && python -m pip install -e source/BPX_test",
        )
    paths = [Path(p).resolve() for p in (spec.submodule_search_locations or [])]
    locations = " ".join(map(str, paths))
    source_dir = (PROJECT_ROOT / "source" / "BPX_test" / "BPX_test").resolve()
    if source_dir not in paths:
        return Check(
            "BPX_test 本体", False, f"已安装但指向别处：{locations}",
            fix="在当前仓库目录重新执行 python -m pip install -e source/BPX_test",
        )
    return Check("BPX_test 本体", True, f"可编辑安装，指向本仓库（{locations}）")


def check_assets(assets: list[Path], label: str, fix: str) -> list[Check]:
    checks = []
    for rel in assets:
        path = PROJECT_ROOT / rel
        if path.is_file() and path.stat().st_size > 0:
            size_mb = path.stat().st_size / 2**20
            checks.append(Check(f"{label}：{rel.name}", True, f"{size_mb:.1f} MB"))
        else:
            checks.append(Check(f"{label}：{rel.name}", False, "文件缺失或为空", fix=fix))
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
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        ok = result.returncode == 0 and bool(result.stdout.strip())
        info = result.stdout.strip() if ok else (result.stderr.strip() or result.stdout.strip() or "查询无输出")
    except (OSError, subprocess.SubprocessError) as exc:
        ok, info = False, str(exc)
    return Check("NVIDIA GPU / 驱动", ok, info,
                 fix="在宿主机终端运行 nvidia-smi；检查驱动或容器/会话的 GPU 透传权限")


def check_training_runtime(device: str) -> Check:
    """在独立进程里验证原生库、RSL 导入和实际 CUDA 运算，防止挂死检查器。"""
    code = """
import sys
import torch
import gymnasium, hydra, numpy, psutil, toml, prettytable, packaging, yaml, trimesh
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.runners import OnPolicyRunner
device = sys.argv[1]
if not torch.cuda.is_available():
    raise RuntimeError('torch.cuda.is_available() = False')
x = torch.ones((32, 32), device=device)
assert (x @ x).sum().item() == 32768
torch.cuda.synchronize(device)
print(f'torch={torch.__version__}, CUDA={torch.version.cuda}, {device}: {torch.cuda.get_device_name(device)}')
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, device], capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=60,
        )
        ok = result.returncode == 0
        detail = result.stdout.strip() if ok else (result.stderr.strip() or result.stdout.strip())[-2500:]
    except (OSError, subprocess.SubprocessError) as exc:
        ok, detail = False, str(exc)
    return Check("训练库导入 / CUDA 运算", ok, detail,
                 fix="先激活 Isaac Lab 环境；检查上方导入错误、CUDA 版 PyTorch、驱动和 --device 编号")


def check_training_versions() -> list[Check]:
    checks = [Check("训练 Python", sys.version_info[:2] in ((3, 10), (3, 11)),
                    f"{sys.version.split()[0]}（Sim 4.5 用 3.10；Sim 5.0 用 3.11）",
                    fix="conda activate isaaclab；从零安装使用 Python 3.11 独立环境")]
    try:
        version = importlib.metadata.version("rsl-rl-lib")
        parts = tuple(int(p) for p in version.split(".")[:3])
        checks.append(Check("RSL-RL 接口版本", (2, 3, 1) <= parts < (2, 4, 0), version,
                            fix="安装项目基线 rsl-rl-lib==2.3.3（当前支持 2.3.1–2.3.x 接口）"))
    except (importlib.metadata.PackageNotFoundError, ValueError):
        checks.append(Check("RSL-RL 接口版本", False, "无法确认版本", fix="安装 rsl-rl-lib==2.3.3"))
    # 二进制安装没有 pip 元数据时，VERSION 也可用于检查 Python ABI。
    try:
        sim_version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        version_file = Path(os.environ.get("ISAAC_PATH", "/nonexistent")) / "VERSION"
        sim_version = version_file.read_text().strip() if version_file.is_file() else ""
    if sim_version:
        expected = (3, 10) if sim_version.startswith("4.5.") else (3, 11)
        supported = sim_version.startswith(("4.5.", "5.0."))
        checks.append(Check("Isaac Sim / Python 配套", supported and sys.version_info[:2] == expected,
                            f"Sim {sim_version} / Python {sys.version.split()[0]}",
                            fix="使用 Sim 5.0 + Python 3.11，或现有 Sim 4.5 + Python 3.10"))
    return checks


def smoke_test(task: str, device: str, num_envs: int, timeout: int) -> int:
    """使用真实训练入口在独立输出目录执行两轮 PPO，检查模型保存。"""
    output_root = PROJECT_ROOT / "outputs" / "training_env_check"
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="check_", dir=output_root))
    command = [sys.executable, str(PROJECT_ROOT / "scripts/rsl_rl/train.py"),
               "--task", task, "--num_envs", str(num_envs), "--max_iterations", "2",
               "--headless", "--device", device, "--logger", "tensorboard", f"agent.device={device}"]
    log_path = output / "training.log"
    print(f"\n真实训练验证：{shlex.join(command)}", flush=True)
    print(f"输出（日志与 checkpoint）：{output}", flush=True)
    # cwd 隔离，train.py 的相对日志目录不会污染正式训练目录。
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            print(f"验证超时或被中断（上限 {timeout}s），已停止验证进程；见 {log_path}")
            return 1
    # RSL-RL v2.3.x 迭代从 0 开始，两轮更新后的最终文件为 model_1.pt。
    checkpoints = list(output.glob("logs/rsl_rl/*/*/model_1.pt"))
    if code != 0 or not checkpoints:
        print(f"结论：真实训练验证失败（退出码 {code}，最终 checkpoint={bool(checkpoints)}）。")
        print(log_path.read_text(errors="replace")[-6000:])
        print(f"完整日志：{log_path}")
        return 1
    print(f"结论：可启动训练，已完成两轮 PPO 并保存 {checkpoints[0]}。")
    print("这只验证小规模训练链路；正式训练的并行环境数仍受显存限制。")
    return 0


def check_tomllib() -> Check:
    if sys.version_info >= (3, 11):
        return Check("tomllib（标准库）", True, "可用")
    return Check(
        "tomllib（标准库）", False, "Python < 3.11，无 tomllib",
        fix="sim2sim 配置读取需要 Python 3.11+，请重建该 conda 环境",
    )


def run_checks(mode: str, device: str = "cuda:0") -> int:
    print(f"工程根目录：{PROJECT_ROOT}")
    print(f"Python    ：{sys.executable}（{sys.version.split()[0]}）")
    print(f"检查模式  ：{mode}\n")

    checks: list[Check] = []
    if mode in ("training", "all"):
        checks += check_training_versions()
        checks += [
            check_package(d, p, ISAACLAB_FIX if p.startswith("isaaclab") else TORCH_FIX if p == "torch" else None)
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
            "激活 Isaac Lab 环境；二进制安装需要 source /path/to/isaac_sim/setup_conda_env.sh",
        )
        checks.append(isaacsim)
        checks.append(check_training_runtime(device))
        checkpoints = sorted(p.name for p in (PROJECT_ROOT / CHECKPOINTS_DIR).glob("*/")) if (PROJECT_ROOT / CHECKPOINTS_DIR).is_dir() else []
        checks.append(Check(
            "训练检查点（可选）", bool(checkpoints), f"logs/rsl_rl 下 {len(checkpoints)} 个实验目录",
            required=False, fix="如需在新机器续训/回放，从旧机器拷贝 logs/rsl_rl/",
        ))
    if mode in ("sim2sim", "all"):
        checks.append(check_python((3, 11), "sim2sim 配置读取用 tomllib"))
        checks.append(check_tomllib())
        checks += [check_package(d, p, "python -m pip install -r sim2sim/requirements.txt")
                   for d, p in SIM2SIM_PACKAGES]
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
        print("未通过的必需项及修复方法：")
        for c in failed_required:
            print(f"  - {c.name}：{c.fix or '见 README「环境准备」一节'}")
        prefix = Path(sys.prefix)
        if "envs" in prefix.parts and os.environ.get("CONDA_PREFIX") != str(prefix):
            print(f"\n  提示：当前解释器属于 conda 环境「{prefix.name}」，但并未通过 conda activate 激活。")
            print("  torch/isaacsim 实际位于 Isaac Sim 目录，由激活脚本经 PYTHONPATH 注入——")
            print(f"  请先 conda activate {prefix.name} 再运行本脚本，否则结果会误报缺失。")
        print(f"\n结论：当前环境不能确认可启动训练。{len(failed_required)} 项必需检查未通过。"
              if mode in ("training", "all") else f"\n结论：{len(failed_required)} 项必需检查未通过。")
        return 1
    print("结论：静态依赖与运行库检查通过（! 为可选项）。")
    if mode in ("training", "all"):
        print("尚未启动仿真；确认训练链路请运行 python scripts/check_dependencies.py --smoke-test")
    if mode in ("sim2sim", "all"):
        print("下一步验证：python sim2sim/validate_setup.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode", choices=("training", "sim2sim", "all"), default="training",
        help="检查哪套环境：training=Isaac Lab 训练环境（默认），sim2sim=MuJoCo 推理环境，all=两者",
    )
    parser.add_argument("--smoke-test", action="store_true", help="检查通过后执行真实仿真与两轮 PPO")
    parser.add_argument("--task", default="BPX-Locomotion-Stairs-History-v0", help="真实验证使用的 Gym 任务")
    parser.add_argument("--device", default="cuda:0", help="CUDA 设备，默认 cuda:0")
    parser.add_argument("--num-envs", type=int, default=4, help="真实验证环境数，默认 4")
    parser.add_argument("--timeout", type=int, default=900, help="真实验证超时秒数，默认 900")
    args = parser.parse_args()
    if args.smoke_test and args.mode == "sim2sim":
        parser.error("--smoke-test 仅适用于 training / all")
    if args.num_envs < 1 or args.timeout < 1:
        parser.error("--num-envs 和 --timeout 必须为正整数")
    if not (args.device == "cuda" or (args.device.startswith("cuda:") and args.device[5:].isdigit())):
        parser.error("训练检查 --device 必须是 cuda 或 cuda:N（Isaac Sim 本身仍需要 NVIDIA GPU）")
    result = run_checks(args.mode, args.device)
    if result == 0 and args.smoke_test:
        return smoke_test(args.task, args.device, args.num_envs, args.timeout)
    return result


if __name__ == "__main__":
    sys.exit(main())
