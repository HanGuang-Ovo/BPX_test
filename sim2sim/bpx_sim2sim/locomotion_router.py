"""Heuristic expert selection from IMU gravity and joint tracking error only."""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ProprioceptionRoutingConfig:
    filter_time: float = 0.15
    enter_tilt: float = 0.14
    exit_tilt: float = 0.12
    enter_joint_error: float = 0.40
    exit_joint_error: float = 0.22
    enter_duration: float = 0.20
    exit_duration: float = 2.0
    exit_clear_fraction: float = 0.85
    minimum_stairs_time: float = 3.0
    startup_grace: float = 0.8

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("本体感知路由参数须为有限正数")
        if self.exit_clear_fraction > 1:
            raise ValueError("exit_clear_fraction 必须位于 (0, 1]")
        if self.exit_tilt >= self.enter_tilt or self.exit_joint_error >= self.enter_joint_error:
            raise ValueError("本体感知路由退出阈值必须小于进入阈值")


class StairPolicyRouter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.on_stairs = False
        self.tilt = self.joint_error = self.joint_error_rms = 0.0
        self.clear_window = deque()
        self.window_time = self.clear_fraction = 0.0
        self.enter_time = self.clear_time = self.hold_time = self.walk_time = 0.0
        self.reason = "flat_initial"

    def update(self, gravity, joint_error, dt, moving=True):
        # Freeze expert choice while standing/stopping; Init must never trigger it.
        if not moving:
            self.enter_time = self.clear_time = self.walk_time = 0.0
            self.tilt = self.joint_error = self.joint_error_rms = 0.0
            self.clear_window.clear()
            self.window_time = self.clear_fraction = 0.0
            return self.on_stairs
        self.walk_time += dt
        alpha = 1 - math.exp(-dt / self.cfg.filter_time)
        tilt = math.atan2(float(np.linalg.norm(gravity[:2])), float(-gravity[2]))
        error = float(np.max(np.abs(joint_error)))
        error_rms = float(np.sqrt(np.mean(np.square(joint_error))))
        if not math.isfinite(tilt) or not math.isfinite(error):
            raise ValueError("本体感知路由输入包含非有限数")
        self.tilt += alpha * (tilt - self.tilt)
        self.joint_error += alpha * (error - self.joint_error)
        self.joint_error_rms += alpha * (error_rms - self.joint_error_rms)
        if self.walk_time < self.cfg.startup_grace:
            return self.on_stairs
        difficult = self.tilt >= self.cfg.enter_tilt or self.joint_error >= self.cfg.enter_joint_error
        # Normal stair-policy gait has a larger peak error even on flat ground.
        # RMS measures whole-body tracking; the entry peak limit still blocks a
        # persistent single-joint obstruction. Brief gait peaks may occupy 15%.
        clear = (self.tilt <= self.cfg.exit_tilt
                 and self.joint_error_rms <= self.cfg.exit_joint_error
                 and self.joint_error < self.cfg.enter_joint_error)
        if not self.on_stairs:
            self.enter_time = self.enter_time + dt if difficult else 0.0
            if self.enter_time >= self.cfg.enter_duration:
                self.on_stairs = True
                self.hold_time = self.clear_time = self.enter_time = 0.0
                self.clear_window.clear()
                self.window_time = self.clear_fraction = 0.0
                self.reason = "imu_tilt" if self.tilt >= self.cfg.enter_tilt else "joint_tracking_error"
        else:
            self.hold_time += dt
            self.clear_window.append((dt, clear))
            self.window_time += dt
            self.clear_time += dt if clear else 0.0
            while self.window_time > self.cfg.exit_duration + 1.e-9:
                length, was_clear = self.clear_window.popleft()
                removed = min(length, self.window_time - self.cfg.exit_duration)
                self.window_time -= removed
                self.clear_time -= removed if was_clear else 0.0
                if length > removed:
                    self.clear_window.appendleft((length - removed, was_clear))
            self.clear_fraction = self.clear_time / self.window_time
            if (self.hold_time >= self.cfg.minimum_stairs_time
                    and self.window_time >= self.cfg.exit_duration - 1.e-9
                    and self.clear_fraction >= self.cfg.exit_clear_fraction
                    and clear):
                self.on_stairs = False
                self.clear_time = 0.0
                self.reason = "stable_proprioception"
        return self.on_stairs
