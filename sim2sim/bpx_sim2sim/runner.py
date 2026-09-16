"""BPX MuJoCo 闭环运行器。

运行器是各层的汇合点，但不负责制定状态转换规则：

- 200 Hz：更新 PD 力矩、推进 MuJoCo、执行基础跌倒检查；
- 50 Hz：读取 command、更新 Supervisor、选择策略、构造观测并推理；
- 策略切换：使用 ActionBlender 保证实际 action 连续；
- 记录：把原始/过滤 command、行为模式、状态、动作和力矩写入 CSV。

不传 ``supervisor`` 时仍走原来的单策略路径，保证旧命令和验证脚本行为不变。
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
import numpy.typing as npt

from .config import Sim2SimConfig
from .behaviors import BehaviorPolicies
from .policy import ConstantPolicy, Policy
from .robot import BpxMujocoRobot
from .supervisor import BehaviorMode, BehaviorSupervisor, SupervisorState
from .transition import ActionBlender


@dataclass(frozen=True)
class RunResult:
    simulated_seconds: float
    policy_steps: int
    termination_reason: str


class CsvLogger:
    """在策略频率记录闭环数据，便于重放、画图和分析策略切换。"""

    def __init__(self, path: Path | None, joint_names: tuple[str, ...]):
        self.path = path
        self.joint_names = joint_names
        self.stream = None
        self.writer = None

    def __enter__(self) -> "CsvLogger":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open("w", encoding="utf-8", newline="")
            fieldnames = [
                "time",
                # 以下三列用于回答“何时切换、实际运行哪个策略、切换完成了多少”。
                "behavior_mode",
                "behavior_policy",
                "transition_alpha",
                "base_x",
                "base_y",
                "base_z",
                "body_vx",
                "body_vy",
                "body_vz",
                "body_wx",
                "body_wy",
                "body_wz",
                "gravity_x",
                "gravity_y",
                "gravity_z",
                "tilt",
                # raw_command 来自手柄/命令行，command 是 Supervisor 过滤后的实际策略输入。
                "raw_command_vx",
                "raw_command_vy",
                "raw_command_wz",
                "command_vx",
                "command_vy",
                "command_wz",
            ]
            for prefix in ("joint_pos", "joint_vel", "joint_target", "action", "torque"):
                fieldnames.extend(f"{prefix}_{name}" for name in self.joint_names)
            self.writer = csv.DictWriter(self.stream, fieldnames=fieldnames)
            self.writer.writeheader()
        return self

    def write(
        self,
        robot: BpxMujocoRobot,
        command: npt.NDArray[np.float32],
        action: npt.NDArray[np.float32],
        target: npt.NDArray[np.floating],
        torque: npt.NDArray[np.floating],
        behavior_mode: str = "direct",
        behavior_policy: str = "default",
        transition_alpha: float = 1.0,
        raw_command: npt.NDArray[np.float32] | None = None,
    ) -> None:
        if self.writer is None:
            return
        state = robot.state()
        base_position = robot.data.xpos[robot.base_body_id]
        raw = command if raw_command is None else raw_command
        row: dict[str, float | str] = {
            "time": float(robot.data.time),
            "behavior_mode": behavior_mode,
            "behavior_policy": behavior_policy,
            "transition_alpha": transition_alpha,
            "base_x": float(base_position[0]),
            "base_y": float(base_position[1]),
            "base_z": float(base_position[2]),
            "body_vx": float(state.base_linear_velocity[0]),
            "body_vy": float(state.base_linear_velocity[1]),
            "body_vz": float(state.base_linear_velocity[2]),
            "body_wx": float(state.base_angular_velocity[0]),
            "body_wy": float(state.base_angular_velocity[1]),
            "body_wz": float(state.base_angular_velocity[2]),
            "gravity_x": float(state.projected_gravity[0]),
            "gravity_y": float(state.projected_gravity[1]),
            "gravity_z": float(state.projected_gravity[2]),
            "tilt": robot.tilt_angle(),
            "raw_command_vx": float(raw[0]),
            "raw_command_vy": float(raw[1]),
            "raw_command_wz": float(raw[2]),
            "command_vx": float(command[0]),
            "command_vy": float(command[1]),
            "command_wz": float(command[2]),
        }
        values = {
            "joint_pos": state.joint_position,
            "joint_vel": state.joint_velocity,
            "joint_target": target,
            "action": action,
            "torque": torque,
        }
        for prefix, array in values.items():
            for name, value in zip(self.joint_names, array, strict=True):
                row[f"{prefix}_{name}"] = float(value)
        self.writer.writerow(row)

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.stream is not None:
            self.stream.close()


class Sim2SimRunner:
    """协调 MuJoCo、底层策略和可选的上层行为状态机。"""

    def __init__(
        self,
        config: Sim2SimConfig,
        policy: Policy,
        supervisor: BehaviorSupervisor | None = None,
        behavior_policies: BehaviorPolicies | None = None,
    ):
        self.config = config
        self.policy = policy
        self.supervisor = supervisor
        # 策略路由只有在 Supervisor 存在时才有意义，提前拒绝不完整的接线。
        if behavior_policies is not None and supervisor is None:
            raise ValueError("配置 behavior_policies 时必须同时配置 supervisor")
        policies = behavior_policies or BehaviorPolicies(locomotion=policy)
        if supervisor is not None and policies.waiting is None:
            # joint_position 与策略 action 零点刻意分离；固定 action 会把初始趴姿
            # 原样保持到 RB 触发，避免等待阶段被站姿零 action 拉起。
            reset_action = (
                np.asarray(config.initial_state.joint_position, dtype=np.float32)
                - np.asarray(config.initial_state.default_joint_position, dtype=np.float32)
            ) / config.control.action_scale
            policies = BehaviorPolicies(
                locomotion=policies.locomotion,
                stand=policies.stand,
                init=policies.init,
                waiting=ConstantPolicy(reset_action),
            )
        self.behavior_policies = policies
        self.model = mujoco.MjModel.from_xml_path(str(config.paths.mjcf))
        self.data = mujoco.MjData(self.model)
        self.robot = BpxMujocoRobot(self.model, self.data, config)

    def run(
        self,
        command: tuple[float, float, float],
        command_source: Callable[[], npt.ArrayLike] | None = None,
        init_request_source: Callable[[], bool] | None = None,
        duration: float | None = None,
        viewer: bool = False,
        realtime: bool | None = None,
        log_path: Path | None = None,
        print_interval: float = 1.0,
        terminate_on_fall: bool = False,
    ) -> RunResult:
        """运行一次闭环仿真。

        Args:
            command: 启动时的原始 ``[vx, vy, wz]``；没有动态来源时会一直使用它。
            command_source: 可选动态指令源，例如手柄。每个策略周期调用一次。
            init_request_source: 可选 Init 触发源；返回 True 的周期进入 INIT。
            duration: 仿真时长；为 None 时使用 TOML 默认值。
            viewer: 是否打开 MuJoCo 被动 viewer。
            realtime: 是否按物理步长进行墙钟限速。
            log_path: 可选 CSV 输出路径。
            print_interval: 终端状态打印周期。
            terminate_on_fall: 是否启用原有的物理步级跌倒终止逻辑。

        Supervisor 启用时，``command``/``command_source`` 都被视为 raw command；真正进入
        观测的是 SupervisorDecision.command。
        """

        raw_command_array = np.asarray(command, dtype=np.float32)
        if raw_command_array.shape != (3,):
            raise ValueError("速度指令必须为 [vx, vy, wz]")
        if not np.all(np.isfinite(raw_command_array)):
            raise FloatingPointError("速度指令中出现 NaN 或 Inf")
        # raw 与 filtered 必须分开保存，否则 CSV 无法判断变化来自手柄还是状态机。
        command_array = raw_command_array.copy()
        run_duration = self.config.simulation.duration if duration is None else duration
        use_realtime = self.config.simulation.realtime if realtime is None else realtime
        self.robot.reset()
        if self.supervisor is None:
            action = np.zeros(self.config.observation.action_dimension, dtype=np.float32)
        else:
            action = (
                np.asarray(self.config.initial_state.joint_position, dtype=np.float32)
                - np.asarray(self.config.initial_state.default_joint_position, dtype=np.float32)
            ) / self.config.control.action_scale
        target = self.robot.desired_joint_position(action)
        torque = np.zeros_like(target)
        policy_steps = 0
        termination_reason = "duration"
        next_print_time = 0.0
        behavior_mode = "direct"
        behavior_policy = "default"
        transition_alpha = 1.0
        # Supervisor 模式下第一帧解析实际行为，并从复位姿态对应的 action 平滑接入。
        active_policy: Policy | None = self.policy if self.supervisor is None else None
        action_blender = ActionBlender(
            self.config.observation.action_dimension,
            self.config.supervisor.action_blend_duration,
        )
        if self.supervisor is not None:
            self.supervisor.reset()

        viewer_context = _viewer_context(self.model, self.data, viewer)
        with viewer_context as active_viewer, CsvLogger(log_path, self.config.joint_names) as logger:
            while self.data.time < run_duration:
                wall_step_start = time.perf_counter()
                # policy_steps 与仿真时间比较，避免墙钟抖动改变 50 Hz 策略时序。
                if policy_steps == 0 or policy_steps * self.config.simulation.policy_dt <= self.data.time + 1.0e-9:
                    if command_source is not None:
                        new_command = np.asarray(command_source(), dtype=np.float32)
                        if new_command.shape != (3,):
                            raise ValueError("动态速度指令必须为 [vx, vy, wz]")
                        if not np.all(np.isfinite(new_command)):
                            raise FloatingPointError("动态速度指令中出现 NaN 或 Inf")
                        raw_command_array[:] = new_command
                    if self.supervisor is None:
                        # 兼容原来的单策略模式：raw command 不经过任何上层修改。
                        command_array[:] = raw_command_array
                    else:
                        init_requested = (
                            bool(init_request_source()) if init_request_source is not None else False
                        )
                        # Supervisor 只接收与仿真器无关的紧凑状态，便于未来复用到实机。
                        state = self.robot.state()
                        decision = self.supervisor.update(
                            raw_command_array,
                            SupervisorState(
                                base_height=self.robot.base_height(),
                                tilt=self.robot.tilt_angle(),
                                planar_speed=float(np.linalg.norm(state.base_linear_velocity[:2])),
                                angular_speed=float(np.linalg.norm(state.base_angular_velocity)),
                                feet_in_contact=self.robot.feet_in_contact(),
                            ),
                            self.config.simulation.policy_dt,
                            init_requested=init_requested,
                        )
                        behavior_mode = decision.mode.value
                        if decision.mode == BehaviorMode.DISABLED:
                            # DISABLED 不对应任何会继续驱动关节的策略，立即安全结束本次仿真。
                            termination_reason = f"supervisor_{decision.reason}"
                            break
                        command_array[:] = decision.command
                        if decision.mode in (BehaviorMode.WAITING_INIT, BehaviorMode.INIT):
                            # 等待和 Init 与训练侧一致，不接收速度或起身时序指令。
                            observation_command = np.zeros(3, dtype=np.float32)
                        else:
                            observation_command = command_array
                        selected_policy = self.behavior_policies.resolve(decision.mode)
                        behavior_policy = self.behavior_policies.policy_name(decision.mode)
                        if selected_policy is not active_policy:
                            # 策略对象真正变化时才重新开始混合；WALK→STOPPING 仍使用 locomotion，
                            # 因此只平滑 command，不会无意义地重复启动 action 混合。
                            action_blender.start(action)
                            active_policy = selected_policy
                    observation = self.robot.observation(
                        observation_command if self.supervisor is not None else command_array,
                        action,
                    )
                    if active_policy is None:
                        raise RuntimeError("没有可执行的 behavior policy")
                    policy_action = np.asarray(active_policy(observation), dtype=np.float32)
                    if self.supervisor is None:
                        action = policy_action
                    else:
                        # action 是“上一帧实际施加值”，下一帧也会作为 last_action 写入观测。
                        action = action_blender.update(policy_action, self.config.simulation.policy_dt)
                        transition_alpha = action_blender.alpha
                    target, torque = self.robot.apply_pd(action)
                    logger.write(
                        self.robot,
                        command_array,
                        action,
                        target,
                        torque,
                        behavior_mode=behavior_mode,
                        behavior_policy=behavior_policy,
                        transition_alpha=transition_alpha,
                        raw_command=raw_command_array,
                    )
                    policy_steps += 1

                # 在一个策略周期内保持目标角，但每个物理步重新计算 PD 力矩。
                target, torque = self.robot.apply_pd(action)
                mujoco.mj_step(self.model, self.data)

                if active_viewer is not None:
                    active_viewer.sync()
                    if not active_viewer.is_running():
                        termination_reason = "viewer_closed"
                        break

                if self.data.time >= next_print_time:
                    print(
                        f"t={self.data.time:6.2f}s  "
                        f"height={self.robot.base_height():.3f}m  "
                        f"tilt={self.robot.tilt_angle():.3f}rad  "
                        f"command=[{command_array[0]:+.2f}, {command_array[1]:+.2f}, "
                        f"{command_array[2]:+.2f}]  "
                        f"mode={behavior_mode}  "
                        f"|action|max={np.max(np.abs(action)):.3f}  "
                        f"|torque|max={np.max(np.abs(torque)):.3f}Nm"
                    )
                    next_print_time += print_interval

                if terminate_on_fall and behavior_mode not in (
                    BehaviorMode.WAITING_INIT.value,
                    BehaviorMode.INIT.value,
                ):
                    reason = self._fall_reason()
                    if reason is not None:
                        termination_reason = reason
                        break

                if use_realtime:
                    remaining = self.config.simulation.timestep - (time.perf_counter() - wall_step_start)
                    if remaining > 0.0:
                        time.sleep(remaining)

        return RunResult(float(self.data.time), policy_steps, termination_reason)

    def _fall_reason(self) -> str | None:
        if self.robot.base_height() < self.config.termination.minimum_base_height:
            return "base_height"
        if self.robot.tilt_angle() > self.config.termination.maximum_tilt:
            return "bad_orientation"
        if self.robot.torso_touches_floor():
            return "torso_contact"
        return None


class _NullViewerContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


class _SafePassiveViewerContext:
    """关闭被动 viewer 后等待其渲染线程结束。

    MuJoCo 的 Linux ``launch_passive`` 使用后台守护线程。部分 MuJoCo/GLFW
    版本中的 ``Handle.close`` 只发送退出请求；如果 Python 随后立即结束，
    原生 OpenGL 资源可能仍在清理，从而在解释器退出阶段触发段错误。
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.handle = None
        self.viewer_thread: threading.Thread | None = None

    def __enter__(self):
        import mujoco.viewer

        threads_before = set(threading.enumerate())
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        new_threads = set(threading.enumerate()) - threads_before
        self.viewer_thread = next(
            (thread for thread in new_threads if "_launch_internal" in thread.name),
            None,
        )
        return self.handle

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.handle is not None:
            self.handle.close()
        if self.viewer_thread is not None:
            self.viewer_thread.join(timeout=5.0)
        else:
            # 兼容未来可能改变线程命名的 MuJoCo 版本，给原生资源清理留出时间。
            time.sleep(0.25)


def _viewer_context(model: mujoco.MjModel, data: mujoco.MjData, enabled: bool):
    if not enabled:
        return _NullViewerContext()
    return _SafePassiveViewerContext(model, data)
