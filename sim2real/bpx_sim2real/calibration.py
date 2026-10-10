"""双向关节转换与 IMU 坐标转换；未确认参数也可用于离线诊断。"""

import numpy as np

from bpx_runtime.math import rotation_body_to_world
from .config import vector


class Calibration:
    def __init__(self, config):
        self.config = config
        self.indices = np.array([j.sdk_index for j in config.joints], dtype=int)
        self.sign = np.array([j.sign for j in config.joints], dtype=float)
        self.offset = np.array([j.offset for j in config.joints], dtype=float)
        self.default = np.array(config.policy["default_joint_position"], dtype=float)

    def position_from_sdk(self, values):
        return self.sign * vector(values, 12, "SDK position")[self.indices] + self.offset

    def velocity_from_sdk(self, values):
        return self.sign * vector(values, 12, "SDK velocity")[self.indices]

    def position_to_sdk(self, values):
        result = np.empty(12)
        result[self.indices] = self.sign * (vector(values, 12, "model target") - self.offset)
        return result

    def gains_to_sdk(self, values):
        result = np.empty(12)
        result[self.indices] = vector(values, 12, "gains")
        return result

    def imu(self, quat_xyzw, omega):
        settings = self.config.calibration["imu"]
        q = vector(quat_xyzw, 4, "IMU quaternion")
        if not 0.9 <= np.linalg.norm(q) <= 1.1:
            raise ValueError("IMU 四元数模长不在 [0.9,1.1]")
        r_world_from_reported = rotation_body_to_world(q[[3, 0, 1, 2]])
        r_body_from_sensor = np.asarray(settings["sensor_to_body"]).reshape(3, 3)
        r_world_from_body = (r_world_from_reported @ r_body_from_sensor.T
                             if settings["quaternion_frame"] == "sensor_to_world"
                             else r_world_from_reported)
        gravity = r_world_from_body.T @ np.array([0., 0., -1.])
        angular = r_body_from_sensor @ (vector(omega, 3, "IMU omega") - settings["gyro_bias"])
        return gravity, angular
