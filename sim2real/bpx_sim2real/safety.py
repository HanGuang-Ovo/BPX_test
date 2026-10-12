"""阶段就绪、反馈新鲜度和关节命令约束。机器人限矩仍由固件保障。"""

import math

import numpy as np

from .config import vector


def require_ready(config, stage):
    reasons = config.blockers(stage)
    if reasons:
        raise RuntimeError("当前阶段尚未就绪：\n  - " + "\n  - ".join(reasons))


class StateGuard:
    def __init__(self, config):
        self.config = config
        self.session = None

    def check(self, sample, now=None, *, require_imu=True, hardware=False):
        now = sample.host_time if now is None else now
        if not sample.connected:
            raise RuntimeError("状态连接断开；需要重新使能")
        if not math.isfinite(sample.host_time) or not math.isfinite(now) or now < sample.host_time - 1e-6:
            raise RuntimeError("状态主机时间无效")
        if self.session is None:
            self.session = sample.session
        elif sample.session != self.session:
            raise RuntimeError("SDK 会话变化；禁止自动恢复控制")
        vector(sample.q, 12, "joint position")
        vector(sample.dq, 12, "joint velocity")
        extra_age = max(0., now - sample.host_time)
        groups = ["joint"]
        if require_imu:
            groups.append("imu")
            vector(sample.quat_xyzw, 4, "IMU quat")
            vector(sample.omega, 3, "IMU omega")
        if require_imu and self.config.policy["include_base_lin_vel"]:
            groups.append("odom")
            vector(sample.velocity_body, 3, "body velocity")
            if self.config.calibration["odometry"].get("verified") is not True and hardware:
                raise RuntimeError("48 维 Actor 所需里程计尚未确认")
        ages = []
        for group in groups:
            age = getattr(sample, f"{group}_age_s")
            token = getattr(sample, f"{group}_token")
            if token is None or age is None or not math.isfinite(age) or age < 0:
                raise RuntimeError(f"缺少有效 {group} 更新信息")
            age += extra_age
            if age > self.config.safety["max_state_age_s"]:
                raise RuntimeError(f"{group} 数据已过期：{age:.4f}s")
            ages.append(age)
        if max(ages) - min(ages) > self.config.safety["max_group_skew_s"]:
            raise RuntimeError("反馈组的主机观测更新时间差过大")
        if hardware:
            if sample.source != "high_rate":
                raise RuntimeError("实机关节控制要求 high_rate 状态")
            expected = self.config.calibration["identity"]
            for key in ("serial_number", "robot_model", "firmware_version"):
                if sample.identity.get(key) != expected.get(key):
                    raise RuntimeError(f"实机身份 {key} 与标定文件不匹配")
            if sample.identity.get("sdk_version") != self.config.firmware["expected_sdk_version"]:
                raise RuntimeError("SDK 版本与确认的固件组合不匹配")


class TargetGuard:
    def __init__(self, config, initial_position):
        self.config = config
        self.last = vector(initial_position, 12, "initial position").copy()
        self.low = np.array([j.minimum for j in config.joints], dtype=float)
        self.high = np.array([j.maximum for j in config.joints], dtype=float)
        if not np.isfinite(self.low).all() or not np.isfinite(self.high).all():
            raise RuntimeError("目标保护需要全部已填写的工作限位")
        if np.any(self.last < self.low) or np.any(self.last > self.high):
            raise RuntimeError("接管初始姿态超出已标定工作限位")
        self.speed = vector(config.safety["target_speed_rad_s"], 12, "target speed", positive=True)
        self.feedback_speed = vector(config.safety["joint_speed_rad_s"], 12, "feedback speed", positive=True)
        self.tracking = vector(config.safety["tracking_error_rad"], 12, "tracking error", positive=True)
        self.torque_limit = vector(config.firmware["robot_torque_limit_nm"], 12, "robot torque limit", positive=True)

    def preview(self, target, q, dq, dt):
        q = vector(q, 12, "position")
        dq = vector(dq, 12, "velocity")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("目标处理 dt 无效")
        if np.any(q < self.low) or np.any(q > self.high):
            raise RuntimeError("实际关节角超出工作限位")
        if np.any(np.abs(dq) > self.feedback_speed):
            raise RuntimeError("实际关节速度超限")
        if np.any(np.abs(self.last - q) > self.tracking):
            raise RuntimeError("实际关节对上一目标的跟踪误差超限")
        desired = np.clip(vector(target, 12, "target"), self.low, self.high)
        processed = self.last + np.clip(desired - self.last, -self.speed * dt, self.speed * dt)
        estimated_torque = np.asarray(self.config.control["kp"]) * (processed - q) - np.asarray(self.config.control["kd"]) * dq
        if np.any(np.abs(estimated_torque) > self.torque_limit):
            raise RuntimeError("按配置 PD 估算的力矩超限；此检查不替代固件限矩")
        return processed, estimated_torque, bool(np.any(np.abs(processed - target) > 1e-7))

    def commit(self, target):
        self.last = np.asarray(target, dtype=float).copy()
