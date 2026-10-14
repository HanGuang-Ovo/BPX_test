"""实机部署命令行；旧调试子命令仅按需转发到独立工具包。"""

import argparse
from contextlib import ExitStack
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from bpx_runtime.joints import SDK_JOINT_NAMES
from bpx_runtime.gamepad import GamepadCommandSource, select_gamepad, BTN_TL, BTN_TR
from .config import DEFAULT_CONFIG, load_config, positive
from .runner import run_live
from .safety import require_ready

TOOL_COMMANDS = ("monitor", "inspect", "analyze", "replay")


def check(config):
    models = {}
    for mode in ("walk", "stand"):
        p = config.model_path(mode)
        models[mode] = {
            "path": str(p),
            "exists": p.is_file(),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
        }
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "onnxruntime", "bpx_sdk")}
    return {
        "config": str(config.path),
        "calibration": str(config.calibration_path),
        "dependencies": dependencies,
        "models": models,
        "observe_blockers": [] if dependencies["bpx_sdk"] else ["当前 Python 未安装 bpx_sdk"],
        "replay_blockers": [] if dependencies["onnxruntime"] else ["当前 Python 未安装 onnxruntime"],
        "deploy_blockers": config.blockers("deploy"),
        "mapping": [
            {
                "policy_index": i,
                "name": j.name,
                "sdk_index": j.sdk_index,
                "sdk_documented_name": SDK_JOINT_NAMES[j.sdk_index],
                "sign": j.sign,
                "offset": j.offset,
                "alignment_verified": config.alignment_verified,
            }
            for i, j in enumerate(config.joints)
        ],
    }


def parser():
    p = argparse.ArgumentParser(description="BPX 实机部署：起身及 Stand/Walk 策略控制")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="离线检查配置、模型和依赖；不连接机器人")
    # 兼容原有启动命令；参数解析和实现都交给工具包。
    for command in TOOL_COMMANDS:
        sub.add_parser(command, add_help=False, help=f"独立调试工具 {command}（转发）")
    run = sub.add_parser("run", help="默认仅检查运行资源；--live 执行实机控制")
    run.add_argument("--live", action="store_true")
    run.add_argument("--duration", type=float, help="可选运行秒数；省略则运行至 Ctrl+C 或故障")
    run.add_argument("--output", type=Path, help="日志路径；省略则自动创建带时间戳的 JSONL")
    run.add_argument("--vx", type=float, default=0.0)
    run.add_argument("--vy", type=float, default=0.0)
    run.add_argument("--wz", type=float, default=0.0)
    run.add_argument("--command-source", choices=("gamepad", "constant"), default="gamepad",
                     help="默认手柄：RB 起身，摇杆行走；constant 仅提供固定速度，无起身请求")
    device = run.add_mutually_exclusive_group()
    device.add_argument("--gamepad-device", type=Path)
    device.add_argument("--gamepad-index", type=int)
    run.add_argument("--gamepad-deadman", action="store_true", help="按住 LB 才接受摇杆速度，RB 独立生效")
    return p


def run_tool(command, config_path, arguments):
    # 源码入口可用；安装后由 pyproject.toml 提供独立工具包。
    tool_path = Path(__file__).resolve().parents[1] / "tools"
    if tool_path.is_dir() and str(tool_path) not in sys.path:
        sys.path.insert(0, str(tool_path))
    from bpx_sim2real_tools.cli import main as tools_main

    return tools_main(["--config", str(config_path), command, *arguments])


def main(argv=None):
    cli = parser()
    args, tool_arguments = cli.parse_known_args(argv)
    if args.command in TOOL_COMMANDS:
        return run_tool(args.command, args.config, tool_arguments)
    if tool_arguments:
        cli.error("unrecognized arguments: " + " ".join(tool_arguments))
    try:
        config = load_config(args.config)
        if args.command == "check":
            result = check(config)
        elif args.command == "run":
            if not args.live:
                result = {
                    "mode": "preflight",
                    "blockers": config.blockers("deploy"),
                    "note": "未创建 SDK 对象；--live 才进入关节控制；仅检查有限数与四元数模长，退出尝试阻尼",
                }
            else:
                if args.output is None:
                    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
                    args.output = Path("outputs/sim2real") / f"live-{stamp}.jsonl"
                require_ready(config, "deploy")
                duration = positive(args.duration, "duration") if args.duration is not None else None
                with ExitStack() as stack:
                    gamepad = None
                    if args.command_source == "gamepad":
                        if any((args.vx, args.vy, args.wz)):
                            raise ValueError("手柄模式不接受固定速度 --vx/--vy/--wz")
                        device = select_gamepad(args.gamepad_device, args.gamepad_index)
                        gamepad = stack.enter_context(GamepadCommandSource(
                            device, tuple(config.control["command_limit"]), init_button=BTN_TR,
                            deadman_button=BTN_TL if args.gamepad_deadman else None))
                        print("手柄已连接：等待 RB 起身；摇杆控制速度。退出尝试 kp=0、kd=1 阻尼。", file=sys.stderr)
                    elif args.gamepad_device is not None or args.gamepad_index is not None or args.gamepad_deadman:
                        raise ValueError("constant 模式不能指定手柄参数")
                    count = run_live(
                        config, args.output, (args.vx, args.vy, args.wz), duration,
                        command_source=gamepad.read if gamepad else None,
                        init_request_source=gamepad.consume_init_request if gamepad else None,
                        status_callback=lambda mode, reason: print(f"状态：{mode} ({reason})", file=sys.stderr),
                    )
                result = {"policy_steps": count, "output": str(args.output)}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("操作已中断，SDK 上下文执行退出处理。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
