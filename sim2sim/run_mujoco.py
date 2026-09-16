#!/usr/bin/env python3
"""在 MuJoCo 中运行 BPX 的 Isaac Lab 策略。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from bpx_sim2sim.config import load_config
from bpx_sim2sim.behaviors import BehaviorPolicies
from bpx_sim2sim.gamepad import (
    BTN_TL,
    BTN_TR,
    GamepadCommandSource,
    GamepadMapping,
    enumerate_gamepads,
    select_gamepad,
)
from bpx_sim2sim.policy import OnnxPolicy, create_policy
from bpx_sim2sim.runner import Sim2SimRunner
from bpx_sim2sim.supervisor import BehaviorSupervisor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config" / "bpx_flat.toml")
    parser.add_argument("--backend", choices=("torchscript", "onnx", "zero"), default="onnx")
    parser.add_argument("--device", default="cpu", help="TorchScript 设备，例如 cpu 或 cuda:0")
    parser.add_argument("--vx", type=float, default=None, help="机体 x 方向目标线速度，m/s")
    parser.add_argument("--vy", type=float, default=None, help="机体 y 方向目标线速度，m/s")
    parser.add_argument("--wz", type=float, default=None, help="目标偏航角速度，rad/s")
    parser.add_argument("--duration", type=float, default=None, help="仿真时长，秒")
    parser.add_argument("--viewer", action="store_true", help="打开 MuJoCo 可视化窗口")
    parser.add_argument("--no-realtime", action="store_true", help="关闭实时限速，尽快完成仿真")
    parser.add_argument("--gamepad", action="store_true", help="用 Linux 游戏手柄实时生成速度指令")
    parser.add_argument("--list-gamepads", action="store_true", help="列出检测到的游戏手柄后退出")
    gamepad_selection = parser.add_mutually_exclusive_group()
    gamepad_selection.add_argument("--gamepad-device", type=Path, help="显式指定 /dev/input/eventN")
    gamepad_selection.add_argument("--gamepad-index", type=int, help="选择 --list-gamepads 输出的设备序号")
    parser.add_argument("--gamepad-deadzone", type=float, default=0.1, help="摇杆死区（默认 0.1）")
    parser.add_argument("--gamepad-max-vx", type=float, default=1.0, help="前后最大速度 m/s")
    parser.add_argument("--gamepad-max-vy", type=float, default=0.5, help="横向最大速度 m/s")
    parser.add_argument("--gamepad-max-wz", type=float, default=1.0, help="最大偏航角速度 rad/s")
    parser.add_argument("--gamepad-vx-axis", type=int, default=1, help="vx 对应的 Linux ABS 轴（默认 1/ABS_Y）")
    parser.add_argument("--gamepad-vy-axis", type=int, default=0, help="vy 对应的 Linux ABS 轴（默认 0/ABS_X）")
    parser.add_argument("--gamepad-wz-axis", type=int, default=3, help="wz 对应的 Linux ABS 轴（默认 3/ABS_RX）")
    parser.add_argument(
        "--gamepad-vx-sign", type=float, choices=(-1.0, 1.0), default=-1.0, help="vx 轴方向（默认 -1）"
    )
    parser.add_argument(
        "--gamepad-vy-sign", type=float, choices=(-1.0, 1.0), default=-1.0, help="vy 轴方向（默认 -1）"
    )
    parser.add_argument(
        "--gamepad-wz-sign", type=float, choices=(-1.0, 1.0), default=-1.0, help="wz 轴方向（默认 -1）"
    )
    parser.add_argument(
        "--gamepad-deadman",
        action="store_true",
        help=f"仅按住 LB（Linux 按钮码 {BTN_TL}）时接受非零指令",
    )
    parser.add_argument(
        "--supervisor",
        action="store_true",
        help="启用 WAITING_INIT/INIT/STAND/WALK/STOPPING/DISABLED 上层状态机",
    )
    parser.add_argument(
        "--stand-policy",
        type=Path,
        help="覆盖 TOML 中的站立 ONNX 策略路径（同为 48 输入、12 输出）",
    )
    init_policy_group = parser.add_mutually_exclusive_group()
    init_policy_group.add_argument(
        "--init-policy",
        type=Path,
        help="覆盖 TOML 中的 Init 起身 ONNX 策略路径（同为 48 输入、12 输出）",
    )
    init_policy_group.add_argument(
        "--recovery-policy",
        dest="init_policy",
        type=Path,
        help=argparse.SUPPRESS,
    )
    termination_group = parser.add_mutually_exclusive_group()
    termination_group.add_argument(
        "--terminate-on-fall",
        action="store_true",
        help="机器人高度过低、倾角过大或躯干触地时提前结束（默认不提前结束）",
    )
    termination_group.add_argument(
        "--no-terminate",
        dest="terminate_on_fall",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(terminate_on_fall=False)
    parser.add_argument("--log", type=Path, default=None, help="保存策略频率的 CSV 日志")
    return parser.parse_args()


def load_optional_policy(
    name: str,
    path: Path | None,
    observation_dimension: int,
    action_dimension: int,
) -> OnnxPolicy | None:
    """加载 Supervisor 的可选 ONNX 策略，路径为空或文件缺失时安全回退。

    locomotion 是主策略，缺少时无法继续运行；stand/init 则是增量加入的专家策略。
    因此可选策略缺失只给出醒目的 warning，由 ``BehaviorPolicies`` 和 Supervisor 分别
    采用 locomotion 站立回退或跌倒后 DISABLED 的既有行为。
    """

    if path is None:
        print(f"[WARNING] TOML 中未配置 {name} policy；将继续使用现有可用策略。", file=sys.stderr)
        return None
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        print(
            f"[WARNING] 未找到 {name} policy：{resolved_path}；将继续使用现有可用策略。",
            file=sys.stderr,
        )
        return None
    return OnnxPolicy(resolved_path, observation_dimension, action_dimension)


def main() -> int:
    args = parse_args()
    if args.list_gamepads:
        devices = enumerate_gamepads()
        if not devices:
            print("没有找到 Linux input 游戏手柄。")
            return 1
        for index, device in enumerate(devices):
            print(f"[{index}] {device.label}")
        return 0

    config = load_config(args.config)
    for path_name, path in (
        ("MJCF", config.paths.mjcf),
        ("TorchScript", config.paths.torchscript_policy),
        ("ONNX", config.paths.locomotion_policy),
    ):
        if path_name == "MJCF" or args.backend in path_name.lower():
            if not path.is_file():
                raise FileNotFoundError(f"{path_name} 文件不存在：{path}")

    command_defaults = config.command.as_tuple()
    command = (
        command_defaults[0] if args.vx is None else args.vx,
        command_defaults[1] if args.vy is None else args.vy,
        command_defaults[2] if args.wz is None else args.wz,
    )
    use_gamepad = args.gamepad or args.gamepad_device is not None or args.gamepad_index is not None
    policy = create_policy(
        backend=args.backend,
        torchscript_path=config.paths.torchscript_policy,
        onnx_path=config.paths.locomotion_policy,
        observation_dimension=config.observation.dimension,
        action_dimension=config.observation.action_dimension,
        device=args.device,
    )
    # 指定专家策略或使用已配置 Init 的手柄流程时隐式启用 Supervisor。
    use_supervisor = (
        args.supervisor
        or args.stand_policy is not None
        or args.init_policy is not None
        or (use_gamepad and config.paths.init_policy is not None)
    )
    stand_policy = None
    init_policy = None
    if use_supervisor:
        # 命令行参数只用于临时覆盖；通常直接使用 TOML 中记录的专家策略路径。
        stand_path = args.stand_policy if args.stand_policy is not None else config.paths.stand_policy
        init_path = args.init_policy if args.init_policy is not None else config.paths.init_policy
        stand_policy = load_optional_policy(
            "stand",
            stand_path,
            config.observation.dimension,
            config.observation.action_dimension,
        )
        init_policy = load_optional_policy(
            "init",
            init_path,
            config.observation.dimension,
            config.observation.action_dimension,
        )
    # init_available 必须反映真实模型是否加载；否则未站立时会路由到空策略。
    supervisor = (
        BehaviorSupervisor(config.supervisor, init_available=init_policy is not None)
        if use_supervisor
        else None
    )
    behavior_policies = (
        BehaviorPolicies(locomotion=policy, stand=stand_policy, init=init_policy)
        if use_supervisor
        else None
    )
    if use_supervisor and stand_policy is None:
        # 这是有意的兼容回退，不代表 locomotion policy 已具备严格静止能力。
        print("提示：未加载到 stand policy，STAND 暂时复用零 command 的 locomotion policy。")
    gamepad_context = nullcontext(None)
    if use_gamepad:
        try:
            device = select_gamepad(args.gamepad_device, args.gamepad_index)
        except (RuntimeError, ValueError) as exc:
            print(f"手柄错误：{exc}", file=sys.stderr)
            return 1
        mapping = GamepadMapping(
            vx_axis=args.gamepad_vx_axis,
            vy_axis=args.gamepad_vy_axis,
            wz_axis=args.gamepad_wz_axis,
            vx_sign=args.gamepad_vx_sign,
            vy_sign=args.gamepad_vy_sign,
            wz_sign=args.gamepad_wz_sign,
        )
        gamepad_context = GamepadCommandSource(
            device=device,
            maximum_command=(args.gamepad_max_vx, args.gamepad_max_vy, args.gamepad_max_wz),
            deadzone=args.gamepad_deadzone,
            mapping=mapping,
            deadman_button=BTN_TL if args.gamepad_deadman else None,
            init_button=BTN_TR if use_supervisor else None,
        )

    runner = Sim2SimRunner(
        config,
        policy,
        supervisor=supervisor,
        behavior_policies=behavior_policies,
    )
    try:
        with gamepad_context as gamepad:
            source_label = f"gamepad ({gamepad.device.label})" if gamepad is not None else "fixed"
            print(
                f"BPX sim2sim: backend={args.backend}, source={source_label}, "
                f"command=[{command[0]}, {command[1]}, {command[2]}], "
                f"supervisor={'on' if use_supervisor else 'off'}, "
                f"physics={1.0 / config.simulation.timestep:.1f}Hz, "
                f"policy={1.0 / config.simulation.policy_dt:.1f}Hz"
            )
            if gamepad is not None and use_supervisor:
                print(f"控制流程：机器人以趴姿等待；按下 RB（Linux 按钮码 {BTN_TR}）启动 Init。")
            result = runner.run(
                command=command,
                command_source=gamepad.read if gamepad is not None else None,
                init_request_source=(
                    gamepad.consume_init_request if gamepad is not None and use_supervisor else None
                ),
                duration=args.duration,
                viewer=args.viewer,
                realtime=False if args.no_realtime else None,
                log_path=args.log,
                terminate_on_fall=args.terminate_on_fall,
            )
    except (OSError, ValueError) as exc:
        if not use_gamepad:
            raise
        print(f"手柄错误：{exc}", file=sys.stderr)
        print("请检查设备节点、轴编号以及 /dev/input/event* 的读取权限。", file=sys.stderr)
        return 1
    print(
        f"结束：simulated={result.simulated_seconds:.3f}s, policy_steps={result.policy_steps}, "
        f"reason={result.termination_reason}"
    )
    return 0 if result.termination_reason in ("duration", "viewer_closed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
