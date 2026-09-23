"""水平趴卧起身：髋坐标系足端五次插值、解析 IK 和关节 PD。

四腿先同步收脚到低位支撑，再沿竖直方向伸展。toe link 原点是运动学端点，
并非足底接触点；standing_depth 不等于机身离地高度。此轨迹面向平地趴卧，
不包含跨台阶落足规划或翻身恢复。所有几何尺寸、轴向和限位从 MJCF 读取。
"""

from __future__ import annotations

import mujoco
import numpy as np

from .robot import BpxMujocoRobot


def quintic(progress: float) -> float:
    """零起止速度、零起止加速度的五次时间标度。"""
    s = float(np.clip(progress, 0.0, 1.0))
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


class LegKinematics:
    """BPX 的 x 轴髋横滚 + y 轴髋俯仰/膝结构，固定膝向后弯曲分支。"""

    def __init__(self, robot: BpxMujocoRobot):
        model = robot.model
        names = robot.config.joint_names
        self.indices = np.array([
            [names.index(f"{leg}_{joint}_joint") for joint in ("hip_roll", "hip_pitch", "knee")]
            for leg in ("fl", "fr", "hl", "hr")
        ])
        offsets, lengths = [], []
        for leg in ("fl", "fr", "hl", "hr"):
            bodies = [model.body(f"{leg}_{part}_link").id for part in ("hip", "thigh", "calf", "toe")]
            hip, thigh, calf, toe = bodies
            if list(model.body_parentid[bodies]) != [robot.base_body_id, hip, thigh, calf]:
                raise ValueError("Init IK 不支持该腿部父子结构")
            if not np.allclose(model.body_quat[bodies], [1, 0, 0, 0]):
                raise ValueError("Init IK 要求腿部零位坐标系方向一致")
            for part, body, axis in zip(("hip_roll", "hip_pitch", "knee"), bodies, np.eye(3)[[0, 1, 1]]):
                joint = model.joint(f"{leg}_{part}_joint").id
                if (model.jnt_bodyid[joint] != body or not np.allclose(model.jnt_axis[joint], axis)
                        or not np.allclose(model.jnt_pos[joint], 0)
                        or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE):
                    raise ValueError("Init IK 不支持该关节轴或关节原点")
            offset = model.body_pos[thigh]
            upper, lower = model.body_pos[calf], model.body_pos[toe]
            if (not np.allclose(offset[[0, 2]], 0) or not np.allclose(upper[:2], 0)
                    or not np.allclose(lower[:2], 0) or upper[2] >= 0 or lower[2] >= 0):
                raise ValueError("Init IK 不支持该腿部连杆偏移")
            offsets.append(offset[1])
            lengths.append([-upper[2], -lower[2]])
        self.offsets = np.asarray(offsets)
        self.upper, self.lower = np.asarray(lengths).T
        self.limits = model.jnt_range[robot.joint_ids].copy()

    def forward(self, joint_position: np.ndarray) -> np.ndarray:
        a, b, c = np.asarray(joint_position)[self.indices].T
        x = -self.upper * np.sin(b) - self.lower * np.sin(b + c)
        z = -self.upper * np.cos(b) - self.lower * np.cos(b + c)
        return np.column_stack((x, self.offsets * np.cos(a) - z * np.sin(a),
                                self.offsets * np.sin(a) + z * np.cos(a)))

    def inverse(self, feet: np.ndarray) -> np.ndarray:
        feet = np.asarray(feet, dtype=float)
        if feet.shape != (4, 3) or not np.all(np.isfinite(feet)):
            raise ValueError("Init 足端目标必须为 4x3 有限数组")
        x, y, z = feet.T
        radial_squared = y * y + z * z - self.offsets ** 2
        if np.any(radial_squared <= 0) or np.any(z >= 0):
            raise ValueError("Init 足端目标超出向下伸腿的可达空间")
        planar_z = -np.sqrt(radial_squared)
        roll = (np.arctan2(z, y) - np.arctan2(planar_z, self.offsets) + np.pi) % (2 * np.pi) - np.pi
        cosine = (x * x + planar_z ** 2 - self.upper ** 2 - self.lower ** 2) / (2 * self.upper * self.lower)
        if np.any(np.abs(cosine) > 1.0 + 1e-9):
            raise ValueError("Init 足端目标超出连杆长度允许的可达空间")
        knee = -np.arccos(np.clip(cosine, -1, 1))
        pitch = np.arctan2(-x, -planar_z) - np.arctan2(self.lower * np.sin(knee),
                                                    self.upper + self.lower * np.cos(knee))
        result = np.empty(12)
        result[self.indices] = np.column_stack((roll, pitch, knee))
        if np.any(result < self.limits[:, 0] - 1e-7) or np.any(result > self.limits[:, 1] + 1e-7):
            raise ValueError("Init IK 目标超过关节限位")
        return result


class FootTrajectoryInit:
    """以物理步频率求轨迹；每次进入 INIT 捕获实际姿态并重新计时。"""

    label = "foot trajectory + PD (quintic)"
    behavior_name = "init_pd"

    def __init__(self, robot: BpxMujocoRobot):
        self.robot = robot
        self.config = robot.config.init_controller
        self.kinematics = LegKinematics(robot)
        self.progress = 0.0
        self.started_at: float | None = None
        self.last_time = 0.0
        self.phase = "idle"
        self.target = robot.default_joint_position.copy()
        self.support_feet = np.column_stack((np.zeros(4), self.kinematics.offsets,
                                            np.full(4, -self.config.support_depth)))
        self.final_feet = self.support_feet.copy()
        self.final_feet[:, 2] = -self.config.standing_depth
        self.final_target = self.kinematics.inverse(self.final_feet)
        # Validate every nominal support/lift sample, including limits, before simulation.
        for s in np.linspace(0, 1, 101):
            self.kinematics.inverse(self.support_feet + s * (self.final_feet - self.support_feet))

    def start(self) -> None:
        self.target = self.robot.joint_position().copy()
        self.start_feet = self.kinematics.forward(self.target)
        for s in np.linspace(0, 1, 101):
            self.kinematics.inverse(self.start_feet + s * (self.support_feet - self.start_feet))
        self.started_at = self.last_time = float(self.robot.data.time)
        self.progress = 0.0
        self.phase = "support"

    @property
    def complete(self) -> bool:
        return self.progress >= self.config.duration and np.max(np.abs(self.target - self.final_target)) < 1e-5

    @property
    def timed_out(self) -> bool:
        return self.started_at is not None and self.robot.data.time - self.started_at >= self.config.timeout

    def __call__(self, observation: np.ndarray | None = None) -> np.ndarray:
        del observation
        if self.started_at is None:
            raise RuntimeError("必须先 start() 再执行 Init")
        now = float(self.robot.data.time)
        dt = max(0.0, now - self.last_time)
        self.last_time = now
        error = float(np.max(np.abs(self.target - self.robot.joint_position())))
        supported = self.robot.feet_in_contact() == 4
        if error <= self.config.tracking_error_limit:
            next_progress = min(self.progress + dt, self.config.duration)
            # At the support waypoint, wait for four feet before lifting further.
            if not supported and next_progress > self.config.support_duration:
                next_progress = max(self.progress, self.config.support_duration)
            self.progress = next_progress
        if self.progress <= self.config.support_duration:
            alpha = quintic(self.progress / self.config.support_duration)
            feet = self.start_feet + alpha * (self.support_feet - self.start_feet)
            self.phase = "support"
        else:
            alpha = quintic((self.progress - self.config.support_duration)
                            / (self.config.duration - self.config.support_duration))
            feet = self.support_feet + alpha * (self.final_feet - self.support_feet)
            self.phase = "lift"
        desired = self.kinematics.inverse(feet)
        self.target += np.clip(desired - self.target, -self.config.joint_speed_limit * dt,
                               self.config.joint_speed_limit * dt)
        if self.complete:
            self.phase = "settle"
        elif error > self.config.tracking_error_limit or (self.progress >= self.config.support_duration and not supported):
            self.phase = "paused"
        return ((self.target - self.robot.default_joint_position) / self.robot.config.control.action_scale).astype(np.float32)
