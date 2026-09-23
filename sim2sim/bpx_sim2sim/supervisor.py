"""Deterministic high-level behavior state machine for BPX."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from .config import SupervisorConfig
from .transition import SlewRateLimiter


class BehaviorMode(str, Enum):
    WAITING_INIT = "waiting_init"
    STAND = "stand"
    WALK = "walk"
    STOPPING = "stopping"
    INIT = "init"
    # 旧枚举名作为别名保留；新 CSV 和状态判断统一使用 INIT / "init"。
    RECOVERY = "init"
    DISABLED = "disabled"


@dataclass(frozen=True)
class SupervisorState:
    base_height: float
    tilt: float
    planar_speed: float
    angular_speed: float
    feet_in_contact: int = 4


@dataclass(frozen=True)
class SupervisorDecision:
    mode: BehaviorMode
    command: npt.NDArray[np.float32]
    mode_changed: bool
    reason: str


class BehaviorSupervisor:
    """使用 RB 触发、稳定确认、迟滞和驻留时间选择起身/站立/行走行为。"""

    def __init__(
        self,
        config: SupervisorConfig,
        init_available: bool = False,
        *,
        recovery_available: bool | None = None,
    ):
        self.config = config
        # recovery_available 是策略改名前的兼容关键字；新代码使用 init_available。
        self.init_available = init_available if recovery_available is None else recovery_available
        self.command_scale = np.asarray(config.command_scale, dtype=np.float32)
        self.command_limiter = SlewRateLimiter(config.command_slew_rate)
        self.mode = BehaviorMode.STAND
        self.mode_time = 0.0
        self.condition_time = 0.0
        self.condition_name: str | None = None
        self.reason = "initialized"

    def reset(
        self
    ) -> None:
        self.command_limiter.reset()
        self.mode = BehaviorMode.STAND
        self.mode_time = 0.0
        self.condition_time = 0.0
        self.condition_name = None
        self.reason = "reset"

    # 判断当前命令的活动程度，返回 [0, 1] 范围的归一化值
    def command_activity(
        self, 
        command: npt.ArrayLike
    ) -> float:
        command_array = np.asarray(command, dtype=np.float32)
        if command_array.shape != (3,) or not np.all(np.isfinite(command_array)):
            raise ValueError("supervisor command 必须为 [vx, vy, wz] 三个有限数")
        return float(np.max(np.abs(command_array) / self.command_scale))

    # 执行状态切换，参数 mode 为目标模式，reason 为切换原因。
    def _transition(
        self, 
        mode: BehaviorMode, 
        reason: str
    ) -> bool:
        # 如果当前模式与目标模式相同，则不切换，返回 False
        if mode == self.mode:
            return False
        # 否则切换模式，重置计时器和命令限制器，返回 True
        self.mode = mode
        self.mode_time = 0.0
        self.condition_time = 0.0
        self.condition_name = None
        self.reason = reason
        if mode in (
            BehaviorMode.WAITING_INIT,
            BehaviorMode.STAND,
            BehaviorMode.INIT,
            BehaviorMode.DISABLED,
        ):
            self.command_limiter.reset()
        return True

    # 滤波器，防止单帧命令波动导致频繁切换模式，使用条件计时器判断是否满足持续时间要求。
    def _condition_duration(
        self, 
        name: str, 
        active: bool, 
        dt: float
    ) -> float:
        if not active:
            self.condition_time = 0.0
            self.condition_name = None
            return 0.0
        if self.condition_name != name:
            self.condition_time = 0.0
            self.condition_name = name
        self.condition_time += dt
        return self.condition_time

    """
    状态机核心逻辑：
    - 根据当前状态、命令活动程度和传感器状态判断是否需要切换模式。
    - 在 WALK 模式下，使用命令滤波器限制速度变化，防止频繁切换。
    - 在 INIT 模式下，确保机器人起身过程中保持稳定，避免倾倒。
    - 在 WAITING_INIT 模式下，等待用户请求起身，或者根据传感器状态判断是否可以直接进入 STAND 模式。
    参数：
    - raw_command: 用户输入的速度命令 [vx, vy, wz]。
    - state: 当前机器人状态，包括基座高度、倾角、平面速度、角速度和足端接触数量。
    - dt: 时间步长，必须大于零。
    - init_requested: 是否请求起身，默认为 False。
    返回：
    - SupervisorDecision: 包含当前模式、滤波后的命令、是否切换模式以及切换原因。
    """
    def update(
        self,
        raw_command: npt.ArrayLike,
        state: SupervisorState,
        dt: float,
        init_requested: bool = False,
    ) -> SupervisorDecision:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("supervisor dt 必须大于零")
        command = np.asarray(raw_command, dtype=np.float32)

        # 计算当前命令的活动程度，范围 [0, 1]，用于判断是否进入 WALK 模式或退出 WALK 模式。
        activity = self.command_activity(command)
        self.mode_time += dt
        mode_changed = False

        # 检查状态是否有效
        state_values = (state.base_height, state.tilt, state.planar_speed, state.angular_speed)
        if not np.all(np.isfinite(state_values)):
            mode_changed = self._transition(BehaviorMode.DISABLED, "non_finite_state")
        if state.planar_speed < 0.0 or state.angular_speed < 0.0:
            raise ValueError("supervisor 速度模长不能小于零")
        if not 0 <= state.feet_in_contact <= 4:
            raise ValueError("supervisor 足端接触数量必须位于 [0, 4]")

        # Init 只覆盖低高度且机身接近水平的腹部朝地趴卧状态。侧翻/仰翻超出了当前
        # 训练分布，即使模型存在也进入 DISABLED，避免错误动作让机器人进一步碰撞。
        low_height = state.base_height < self.config.fall_base_height
        excessive_tilt = state.tilt > self.config.fall_tilt
        not_standing = low_height or excessive_tilt
        prone_for_init = low_height and state.tilt < self.config.init_start_max_tilt

        # 如果机器人不处于站立状态，且当前模式不是 WAITING_INIT、INIT 或 DISABLED，则根据是否支持起身和是否请求起身来决定切换到 WAITING_INIT 或 DISABLED。
        if self.mode not in (
            BehaviorMode.WAITING_INIT,
            BehaviorMode.INIT,
            BehaviorMode.DISABLED,
        ) and not_standing:
            if self.init_available and prone_for_init:
                target = BehaviorMode.WAITING_INIT
                reason = "awaiting_init_request"
            else:
                target = BehaviorMode.DISABLED
                reason = "unsupported_init_pose" if self.init_available else "not_standing_without_init"
            mode_changed = self._transition(target, reason)

        # 处于 WATING_INIT 模式，且没有别的问题，按下按键触发INIT模式；如果机器人已经站立，直接切换到 STAND 模式；如果机器人不支持起身，切换到 DISABLED 模式。  
        if self.mode == BehaviorMode.WAITING_INIT:
            if not self.init_available:
                mode_changed = self._transition(
                    BehaviorMode.DISABLED, "not_standing_without_init"
                ) or mode_changed
            elif not low_height and state.tilt < self.config.fall_tilt:
                mode_changed = self._transition(BehaviorMode.STAND, "already_standing") or mode_changed
            elif not prone_for_init:
                mode_changed = self._transition(
                    BehaviorMode.DISABLED, "unsupported_init_pose"
                ) or mode_changed
            elif init_requested:
                mode_changed = self._transition(BehaviorMode.INIT, "init_requested") or mode_changed

        # 如果机器人处于 STAND 模式，且命令活动程度超过阈值，则开始计时，如果持续时间超过延迟，则切换到 WALK 模式。
        elif self.mode == BehaviorMode.STAND:
            duration = self._condition_duration(
                "walk_enter", activity > self.config.walk_enter_threshold, dt
            )
            if duration >= self.config.walk_enter_delay:
                mode_changed = self._transition(BehaviorMode.WALK, "command_enter") or mode_changed

        # 如果机器人处于 WALK 模式，且命令活动程度低于阈值，则开始计时，如果持续时间超过延迟，则切换到 STOPPING 模式。
        elif self.mode == BehaviorMode.WALK:
            can_stop = self.mode_time >= self.config.minimum_walk_time
            duration = self._condition_duration(
                "walk_exit", can_stop and activity < self.config.walk_exit_threshold, dt
            )
            if duration >= self.config.walk_exit_delay:
                mode_changed = self._transition(BehaviorMode.STOPPING, "command_exit") or mode_changed

        # 如果机器人处于 STOPPING 模式，且命令活动程度超过阈值，则开始计时，如果持续时间超过延迟，则切换到 WALK 模式；否则，如果机器人速度低于阈值，且命令活动程度低于阈值，则开始计时，如果持续时间超过确认时间，则切换到 STAND 模式；否则，如果 STOPPING 模式持续时间超过超时，则切换到 STAND 模式。
        elif self.mode == BehaviorMode.STOPPING:
            if activity > self.config.walk_enter_threshold:
                duration = self._condition_duration("walk_resume", True, dt)
                if duration >= self.config.walk_enter_delay:
                    mode_changed = self._transition(BehaviorMode.WALK, "command_resumed") or mode_changed
            else:
                stable = (
                    state.planar_speed < self.config.stop_linear_speed
                    and state.angular_speed < self.config.stop_angular_speed
                    and self.command_activity(self.command_limiter.value) < self.config.walk_exit_threshold
                )
                duration = self._condition_duration("stand_confirm", stable, dt)
                if duration >= self.config.stand_confirm_time:
                    mode_changed = self._transition(BehaviorMode.STAND, "robot_stopped") or mode_changed
                elif self.mode_time >= self.config.stopping_timeout:
                    mode_changed = self._transition(BehaviorMode.STAND, "stopping_timeout") or mode_changed

        # 如果机器人处于 INIT 模式，需要检查是否满足初始化条件，如果倾角超过阈值，则切换到 DISABLED 模式；如果满足初始化条件，则开始计时，如果持续时间超过确认时间，则切换到 STAND 模式。
        elif self.mode == BehaviorMode.INIT:
            # Init 只针对接近水平的腹部趴卧状态。若起身过程中倾角持续增大，说明策略
            # 已偏离训练目标；立即停止策略驱动，避免继续输出力矩直至完全翻倒。
            if state.tilt >= self.config.init_abort_tilt:
                mode_changed = self._transition(
                    BehaviorMode.DISABLED, "init_tilt_abort"
                ) or mode_changed

            # 起身完成必须连续确认 0.5 秒，保证交给 stand policy 时已经基本稳定。
            initialized = self.mode == BehaviorMode.INIT and (
                state.base_height > self.config.init_base_height
                and state.tilt < self.config.init_tilt
                and state.planar_speed < self.config.stop_linear_speed
                and state.angular_speed < self.config.stop_angular_speed
                and state.feet_in_contact == 4
            )
            duration = self._condition_duration("init_confirm", initialized, dt)
            if duration >= self.config.init_confirm_time:
                mode_changed = self._transition(BehaviorMode.STAND, "init_complete") or mode_changed

        # target_command 是根据当前模式和用户命令计算出的目标速度命令，如果当前模式是 WALK，则使用用户命令，否则使用零向量。
        # filtered_command 是经过命令滤波器处理后的速度命令，如果当前模式是 WAITING_INIT、STAND、INIT 或 DISABLED，则使用零向量，否则使用滤波后的目标命令。
        target_command = command if self.mode == BehaviorMode.WALK else np.zeros(3, dtype=np.float32)
        if self.mode in (
            BehaviorMode.WAITING_INIT,
            BehaviorMode.STAND,
            BehaviorMode.INIT,
            BehaviorMode.DISABLED,
        ):
            filtered_command = np.zeros(3, dtype=np.float32)
        else:
            filtered_command = self.command_limiter.update(target_command, dt)
        return SupervisorDecision(
            mode=self.mode,
            command=filtered_command,
            mode_changed=mode_changed,
            reason=self.reason,
        )
