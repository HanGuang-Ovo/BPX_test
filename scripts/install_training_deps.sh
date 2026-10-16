#!/usr/bin/env bash
# 在选定 Python 环境安装 BPX 训练依赖；不安装系统驱动、不启动正式训练。
set -euo pipefail

BPX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BPX_PYTHON="${PYTHON:-python}"
BPX_LAB="${ISAACLAB_PATH:-$BPX_ROOT/.deps/IsaacLab-2.2.1}"
BPX_FULL=0
BPX_DRY_RUN=0
BPX_SMOKE=0
BPX_TASK="BPX-Locomotion-Stairs-History-v0"
BPX_DEVICE="cuda:0"

usage() {
    cat <<'EOF'
用法：bash scripts/install_training_deps.sh [选项]

默认：在已激活的 Isaac Lab 环境补齐训练工具、RSL-RL 2.3.3，并可编辑安装 BPX。
--full                 从零安装 Sim 5.0.0、CUDA PyTorch、Isaac Lab v2.2.1（需要 Python 3.11）
--python PATH          安装目标解释器或 Isaac Sim python.sh，默认当前 python
--isaaclab PATH        --full 使用的 Isaac Lab 源码目录，默认 $ISAACLAB_PATH 或 .deps/IsaacLab-2.2.1
                       目录不存在时克隆固定 tag；存在时验证 VERSION，不切分支或覆盖源码
--smoke-test           安装后额外启动 4 个环境，执行两轮 PPO
--task ID              验证任务，默认 BPX-Locomotion-Stairs-History-v0
--device cuda:N        检查/验证设备，默认 cuda:0
--dry-run              只打印安装命令，不下载、不安装、不创建目录
-h, --help             显示帮助

示例：
  conda activate isaaclab
  bash scripts/install_training_deps.sh
  bash scripts/install_training_deps.sh --full --python /path/to/python3.11 --dry-run

--full 下载量和磁盘占用较大。建议使用独立的 Conda/venv 环境。
系统 NVIDIA 驱动与机器人 USD 资产需另行准备；最后的检查失败会返回非零退出码。
EOF
}

die() { echo "[错误] $*" >&2; exit 1; }
value_required() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 缺少参数"; }
run() {
    printf '[执行]'; printf ' %q' "$@"; printf '\n'
    if (( ! BPX_DRY_RUN )); then "$@"; fi
}
trap 'echo "[错误] 安装或验证失败（行 $LINENO）。修复上方错误后可重新运行；已有成功步骤会保留。" >&2' ERR

while (( $# )); do
    case "$1" in
        --full) BPX_FULL=1; shift ;;
        --dry-run) BPX_DRY_RUN=1; shift ;;
        --smoke-test) BPX_SMOKE=1; shift ;;
        --python) value_required "$@"; BPX_PYTHON="$2"; shift 2 ;;
        --isaaclab) value_required "$@"; BPX_LAB="$2"; shift 2 ;;
        --task) value_required "$@"; BPX_TASK="$2"; shift 2 ;;
        --device) value_required "$@"; BPX_DEVICE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "未知选项：$1（使用 --help 查看用法）" ;;
    esac
done
[[ "$BPX_DEVICE" =~ ^cuda(:[0-9]+)?$ ]] || die "--device 必须为 cuda 或 cuda:N"
command -v "$BPX_PYTHON" >/dev/null || die "找不到解释器：$BPX_PYTHON"
# 统一成绝对路径，避免后续相对路径或工作目录改变导致装到其他环境。
BPX_PYTHON="$(command -v "$BPX_PYTHON")"
[[ "$BPX_PYTHON" = /* ]] || BPX_PYTHON="$PWD/$BPX_PYTHON"
[[ "$BPX_LAB" = /* ]] || BPX_LAB="$PWD/$BPX_LAB"
echo "项目：$BPX_ROOT"
echo "目标解释器：$BPX_PYTHON"

if (( ! BPX_DRY_RUN )); then
    "$BPX_PYTHON" - "$BPX_FULL" <<'PY'
import importlib.util
import sys

full = sys.argv[1] == "1"
allowed = ((3, 11),) if full else ((3, 10), (3, 11))
if sys.version_info[:2] not in allowed:
    sys.exit(f"Python {sys.version.split()[0]} 不支持本项目训练基线。"
             "先激活 Isaac Lab 环境；--full 需要 Python 3.11。")
if not full:
    missing = [name for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl",
                                 "isaacsim", "torch", "torchvision")
               if importlib.util.find_spec(name) is None]
    if missing:
        sys.exit("现有基础环境不完整：" + ", ".join(missing)
                 + "。先激活正确的 Isaac Lab 环境，或在 Python 3.11 独立环境使用 --full。")
print(f"Python {sys.version.split()[0]} / {sys.prefix}")
PY
fi

if (( BPX_FULL )); then
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "--full 仅支持 Linux x86_64"
    command -v git >/dev/null || die "--full 需要 git"
    # 先验证本地源码，再进行任何 pip 修改。
    if [[ -e "$BPX_LAB" ]]; then
        [[ -f "$BPX_LAB/VERSION" ]] || die "$BPX_LAB 不是 Isaac Lab 源码目录（缺少 VERSION）"
        [[ "$(tr -d '[:space:]' < "$BPX_LAB/VERSION")" == 2.2.1 ]] || die "源码版本必须是 2.2.1"
        for package in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
            [[ -f "$BPX_LAB/source/$package/setup.py" ]] || die "Isaac Lab 源码不完整：$package"
        done
    else
        run mkdir -p -- "$(dirname -- "$BPX_LAB")"
        run git clone --depth 1 --branch v2.2.1 https://github.com/isaac-sim/IsaacLab.git "$BPX_LAB"
    fi
    # 官方 v2.2.1 pip 安装基线，不跟随 latest：
    # https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/pip_installation.html
    run "$BPX_PYTHON" -m pip install --upgrade pip
    run "$BPX_PYTHON" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    run "$BPX_PYTHON" -m pip install 'isaacsim[all,extscache]==5.0.0' --extra-index-url https://pypi.nvidia.com
    # 通过指定解释器直接安装扩展，避免 isaaclab.sh 选到另一个 Conda 环境。
    run "$BPX_PYTHON" -m pip install \
        -e "$BPX_LAB/source/isaaclab" \
        -e "$BPX_LAB/source/isaaclab_assets" \
        -e "$BPX_LAB/source/isaaclab_tasks" \
        -e "$BPX_LAB/source/isaaclab_rl[rsl-rl]" \
        'torch==2.7.0' 'torchvision==0.22.0' 'rsl-rl-lib==2.3.3'
else
    # 二进制 Isaac Sim 的 torch 可能仅通过 PYTHONPATH 注入，无 pip 元数据。
    # --no-deps 避免 pip 因此重装 torch；RSL-RL 的其余依赖在下一步显式安装。
    run "$BPX_PYTHON" -m pip install --no-deps rsl-rl-lib==2.3.3
fi

run "$BPX_PYTHON" -m pip install 'numpy>=1.24,<2' psutil toml prettytable \
    hydra-core tensorboard packaging PyYAML trimesh h5py GitPython 'onnx>=1.18.0'
run "$BPX_PYTHON" -m pip install --no-deps -e "$BPX_ROOT/source/BPX_test"

BPX_CHECK=("$BPX_PYTHON" "$BPX_ROOT/scripts/check_dependencies.py" --mode training --device "$BPX_DEVICE")
if (( BPX_SMOKE )); then BPX_CHECK+=(--smoke-test --task "$BPX_TASK"); fi
echo "下一步：检查驱动、CUDA、包路径和 USD 资产。"
run "${BPX_CHECK[@]}"
if (( BPX_DRY_RUN )); then
    echo "预览完成：未执行安装或检查。"
else
    echo "安装与所选检查通过。训练仍需在同一 Python 环境中启动。"
fi
