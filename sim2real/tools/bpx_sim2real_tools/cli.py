"""调试工具命令行；部署入口仅在选择工具命令时转发到这里。"""

import argparse
import json
from pathlib import Path
import sys

from bpx_sim2real.config import DEFAULT_CONFIG, load_config


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def add_gamepad_arguments(parser):
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--gamepad-device", type=Path, help="只读手柄设备，例如 /dev/input/eventN")
    selector.add_argument("--gamepad-index", type=int, help="多个手柄时指定列表索引")
    parser.add_argument("--list-gamepads", action="store_true", help="列出电脑手柄，不连接机器人或打开窗口")


def list_gamepads():
    from bpx_runtime.gamepad import enumerate_gamepads

    devices = enumerate_gamepads()
    for index, device in enumerate(devices):
        print(f"[{index}] {device.label}")
    if not devices:
        print("没有找到电脑 USB/蓝牙手柄")
    return 0


def parser():
    p = argparse.ArgumentParser(description="BPX 独立调试工具：窗口、采集、分析和离线回放")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="command", required=True)
    monitor = sub.add_parser("monitor", help="只读反馈、策略预览与三维姿态窗口")
    monitor.add_argument("--demo", action="store_true", help="模拟反馈，不连接 SDK")
    monitor.add_argument("--recording", type=Path, help="回放 inspect 录制，不连接 SDK")
    monitor.add_argument("--no-policy", action="store_true", help="只看反馈，不加载模型")
    monitor.add_argument("--output", type=Path, help="保存原始反馈 JSONL")
    monitor.add_argument("--duration", type=float, help="窗口自动关闭时间，秒")
    add_gamepad_arguments(monitor)
    observe = sub.add_parser("inspect", help="只读状态录制；不发送运动命令")
    observe.add_argument("--output", type=Path, required=True)
    observe.add_argument("--duration", type=float, default=10.0)
    observe.add_argument("--poll-hz", type=float, default=100.0)
    analyze = sub.add_parser("analyze", help="分析只读录制")
    analyze.add_argument("recording", type=Path)
    analyze.add_argument("--output", type=Path)
    off = sub.add_parser("replay", help="离线策略推理；不连接 SDK")
    off.add_argument("recording", type=Path)
    off.add_argument("--output", type=Path, required=True)
    off.add_argument("--vx", type=float, default=0.0)
    off.add_argument("--vy", type=float, default=0.0)
    off.add_argument("--wz", type=float, default=0.0)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "monitor":
            if args.list_gamepads:
                return list_gamepads()
            from .monitor import launch

            return launch(
                config,
                demo=args.demo,
                recording=args.recording,
                no_policy=args.no_policy,
                output=args.output,
                duration=args.duration,
                gamepad_device=args.gamepad_device,
                gamepad_index=args.gamepad_index,
            )
        if args.command == "inspect":
            from .capture import capture

            result = capture(config, args.output, args.duration, args.poll_hz)
        elif args.command == "analyze":
            from .recording import load_samples, summarize

            _, samples = load_samples(args.recording)
            result = summarize(samples)
            if args.output:
                write_json(args.output, result)
        elif args.command == "replay":
            from .replay import replay

            count = replay(config, args.recording, args.output, (args.vx, args.vy, args.wz))
            result = {"policy_steps": count, "output": str(args.output), "mode": "offline"}
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
