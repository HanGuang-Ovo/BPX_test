"""官方 SDK 坐标默认值、可选覆盖与实机运动就绪检查。"""

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import tomllib

import numpy as np

from bpx_runtime.joints import POLICY_JOINT_NAMES, SDK_JOINT_NAMES, DEFAULT_JOINT_POSITION
from bpx_runtime.supervisor import SupervisorConfig
from bpx_runtime.init_controller import InitControllerConfig

# 官方 BPX/bpx/urdf/bpx.urdf 的模型范围；不代表固件限矩或机械验收。
OFFICIAL_JOINT_LIMITS = ((-.8727, .8727),) * 4 + ((-.7854, 2.60661),) * 4 + ((-2.7531, -.5539),) * 4


def vector(value, size, name, *, positive=False):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} 要求 {size} 个有限数")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} 必须全部大于零")
    return array


def positive(value, name):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} 必须为有限正数")
    return result


@dataclass(frozen=True)
class JointCalibration:
    name: str
    sdk_index: int
    sign: int
    offset: float
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class Config:
    path: Path
    calibration_path: Path
    runtime: dict
    calibration: dict
    joints: tuple[JointCalibration, ...]

    @property
    def network(self):
        return self.runtime["network"]

    @property
    def policy(self):
        return self.runtime["policy"]

    @property
    def control(self):
        return self.runtime["control"]

    @property
    def safety(self):
        return self.runtime["safety"]

    @property
    def firmware(self):
        return self.runtime["firmware"]

    @property
    def init_controller(self):
        return InitControllerConfig(**{
            k: v for k, v in self.runtime["init_controller"].items()
            if k in InitControllerConfig.__dataclass_fields__
        })

    @property
    def supervisor(self):
        control = self.control
        return SupervisorConfig(
            command_scale=tuple(control["command_limit"]),
            command_slew_rate=tuple(control["command_slew_rate"]),
            walk_enter_threshold=control["walk_enter_threshold"],
            walk_exit_threshold=control["walk_exit_threshold"],
            walk_enter_delay=control["walk_enter_delay_s"],
            walk_exit_delay=control["walk_exit_delay_s"],
            minimum_walk_time=control["minimum_walk_time_s"],
            stopping_timeout=control["stopping_duration_s"],
            action_blend_duration=control["action_blend_duration_s"],
            **{k: self.runtime["supervisor"][k] for k in (
                "stop_linear_speed", "stop_angular_speed", "stand_confirm_time",
                "fall_base_height", "fall_tilt", "init_start_max_tilt", "init_abort_tilt",
                "init_base_height", "init_tilt", "init_confirm_time")},
        )

    @property
    def period(self):
        return 1.0 / self.control["policy_rate_hz"]

    def resolve(self, value):
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (self.path.parent / p).resolve()

    def model_path(self, mode):
        return self.resolve(self.policy[f"{mode}_model"])

    @property
    def alignment_verified(self):
        return self.calibration.get("alignment", {}).get("verified", False)

    def blockers(self, stage="deploy"):
        """读取和预览无需确认；运动前整体核对 SDK 与训练模型。"""
        if stage in ("observe", "replay"):
            return []
        reasons = []
        identity = self.calibration["identity"]
        for key in ("serial_number", "robot_model", "firmware_version"):
            if not identity.get(key):
                reasons.append(f"calibration.identity.{key} 未填写")
        if identity.get("robot_model") != "BPX":
            reasons.append("当前实机 Init 几何仅支持已核对的 BPX 型号")
        fw = self.firmware
        for key in ("joint_control_verified", "watchdog_verified", "shutdown_verified",
                    "gain_units_verified", "robot_limits_verified"):
            if fw.get(key) is not True:
                reasons.append(f"firmware.{key} 未确认")
        if fw.get("shutdown_action") != "zero_joint_command":
            reasons.append("仅支持已实测的 shutdown_action=zero_joint_command")
        if not isinstance(fw.get("expected_sdk_version"), str) or not fw["expected_sdk_version"]:
            reasons.append("firmware.expected_sdk_version 未填写")
        watchdog = fw.get("watchdog_timeout_s", 0)
        if not math.isfinite(watchdog) or watchdog <= 0:
            reasons.append("firmware.watchdog_timeout_s 未实测")
        if not fw.get("robot_torque_limit_nm", []):
            reasons.append("firmware.robot_torque_limit_nm 未填写")
        if self.safety.get("allow_motion") is not True:
            reasons.append("safety.allow_motion=false")
        if not self.alignment_verified:
            reasons.append("alignment.verified=false；请在只读窗口核对关节与 IMU 坐标")
        if self.policy["include_base_lin_vel"] and self.calibration["odometry"].get("verified") is not True:
            reasons.append("48 维 Actor 所需 calibration.odometry.verified=false")
        if not self.control.get("gains_verified", False):
            reasons.append("control.gains_verified=false")
        if not self.control.get("standing_handover_verified", False):
            reasons.append("control.standing_handover_verified=false")
        if not self.control.get("prone_handover_verified", False):
            reasons.append("control.prone_handover_verified=false")
        if not self.runtime["init_controller"]["verified"]:
            reasons.append("init_controller.verified=false；需验证实机起身轨迹与增益")
        if not self.runtime["supervisor"]["feedback_verified"]:
            reasons.append("supervisor.feedback_verified=false；需验证高度与接地估计")
        if self.calibration["odometry"].get("verified") is not True:
            reasons.append("calibration.odometry.verified=false；实机状态机需要里程计")
        for mode in ("walk", "stand"):
            if not self.model_path(mode).is_file():
                reasons.append(f"policy.{mode}_model 不存在")
        for key in ("target_speed_rad_s", "joint_speed_rad_s", "tracking_error_rad", "takeover_position_error_rad"):
            if not self.safety.get(key):
                reasons.append(f"safety.{key} 未填写")
        for j in self.joints:
            if j.minimum is None or j.maximum is None:
                reasons.append(f"{j.name} 缺少 minimum/maximum")
        return reasons


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    calibration_path = Path(raw["calibration_file"]).expanduser()
    if not calibration_path.is_absolute():
        calibration_path = path.parent / calibration_path
    with calibration_path.open("rb") as stream:
        calibration = tomllib.load(stream)
    calibration.setdefault("identity", {"serial_number": "", "robot_model": "", "firmware_version": ""})
    calibration.setdefault("alignment", {"verified": False})
    calibration.setdefault("imu", {})
    for key, value in {"sensor_to_body": [1,0,0,0,1,0,0,0,1], "gyro_bias": [0,0,0],
                       "quaternion_frame": "body_to_world"}.items():
        calibration["imu"].setdefault(key, value)
    calibration.setdefault("odometry", {"verified": False})
    net = raw["network"]
    if not isinstance(net["robot_ip"], str) or not net["robot_ip"]:
        raise ValueError("network.robot_ip 不能为空")
    for key in ("state_port", "joint_state_port", "tcp_local_port"):
        value = net[key]
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 65535:
            raise ValueError(f"network.{key} 端口无效")
    if not net["state_port"] or not net["joint_state_port"] or net["state_port"] == net["joint_state_port"]:
        raise ValueError("两个 UDP 端口必须非零且不同")
    if not isinstance(net["state_rate_hz"], int) or not 1 <= net["state_rate_hz"] <= 200:
        raise ValueError("network.state_rate_hz 要求 1..200")
    positive(net["connect_timeout_s"], "network.connect_timeout_s")
    policy = raw["policy"]
    if policy["history_length"] not in (1, 10) or not isinstance(policy["include_base_lin_vel"], bool):
        raise ValueError("支持 H=1/10 的 45/48 维单帧 Actor")
    dimension = (48 if policy["include_base_lin_vel"] else 45) * policy["history_length"]
    if policy["observation_dimension"] != dimension:
        raise ValueError("policy.observation_dimension 与历史配置不一致")
    if tuple(policy["joint_names"]) != POLICY_JOINT_NAMES:
        raise ValueError("policy.joint_names 必须使用已训练策略的顺序")
    vector(policy["default_joint_position"], 12, "default_joint_position")
    if not np.array_equal(policy["default_joint_position"], DEFAULT_JOINT_POSITION):
        raise ValueError("当前模型的默认关节角必须为 roll=0/pitch=0.7/knee=-1.4")
    if policy["action_scale"] != 0.5:
        raise ValueError("当前模型的 action_scale 必须为 0.5")
    control = raw["control"]
    control.setdefault("prone_handover_verified", False)
    control.setdefault("prone_joint_position", [0.] * 4 + [.98] * 4 + [-2.62] * 4)
    init = raw.setdefault("init_controller", {})
    for key, value in asdict(InitControllerConfig(mode="pd")).items():
        init.setdefault(key, value)
    init.setdefault("verified", False)
    behavior = raw.setdefault("supervisor", {})
    defaults = dict(stop_linear_speed=.05, stop_angular_speed=.10, stand_confirm_time=.20,
                    fall_base_height=.20, fall_tilt=.70, init_start_max_tilt=.35,
                    init_abort_tilt=.55, init_base_height=.36, init_tilt=.25, init_confirm_time=.50,
                    feedback_verified=False, ground_z=0., toe_radius=.027,
                    toe_center_depth=.012, toe_half_width=.0179,
                    contact_height_tolerance=.025, contact_force_threshold_n=5.,
                    contact_jacobian_condition_max=100., torque_bias_nm=[0.] * 12)
    for key, value in defaults.items():
        behavior.setdefault(key, value)
    vector(control["prone_joint_position"], 12, "control.prone_joint_position")
    if init["mode"] != "pd":
        raise ValueError("实机 Init 仅支持 mode=pd")
    for key in InitControllerConfig.__dataclass_fields__:
        if key != "mode":
            positive(init[key], f"init_controller.{key}")
    if not init["support_duration"] < init["duration"] < init["timeout"]:
        raise ValueError("Init 要求 support_duration < duration < timeout")
    if init["support_depth"] >= init["standing_depth"]:
        raise ValueError("Init 要求 support_depth < standing_depth")
    for key in defaults:
        if key not in ("feedback_verified", "ground_z", "torque_bias_nm"):
            positive(behavior[key], f"supervisor.{key}")
    if not math.isfinite(behavior["ground_z"]):
        raise ValueError("supervisor.ground_z 必须有限")
    vector(behavior["torque_bias_nm"], 12, "supervisor.torque_bias_nm")
    if not (behavior["init_tilt"] < behavior["init_start_max_tilt"]
            < behavior["init_abort_tilt"] <= behavior["fall_tilt"] < np.pi / 2):
        raise ValueError("Init/跌倒倾角阈值顺序无效")
    if behavior["init_base_height"] <= behavior["fall_base_height"]:
        raise ValueError("Init 完成高度必须高于跌倒高度")
    if control["policy_rate_hz"] != 50:
        raise ValueError("当前策略及历史必须按 50 Hz 更新")
    positive(control["send_rate_hz"], "control.send_rate_hz")
    if control["send_rate_hz"] < control["policy_rate_hz"]:
        raise ValueError("发送频率不得低于策略频率")
    for key in ("kp", "kd"):
        vector(control[key], 12, key, positive=True)
    for key in ("takeover_duration_s", "action_blend_duration_s", "walk_enter_delay_s",
                "walk_exit_delay_s", "minimum_walk_time_s", "stopping_duration_s"):
        positive(control[key], key)
    vector(control["command_limit"], 3, "command_limit", positive=True)
    vector(control["command_slew_rate"], 3, "command_slew_rate", positive=True)
    if not 0 <= control["walk_exit_threshold"] < control["walk_enter_threshold"]:
        raise ValueError("Walk 迟滞阈值无效")
    safety = raw["safety"]
    for key in ("max_state_age_s", "max_group_skew_s", "max_cycle_lateness_s", "max_tilt_rad"):
        positive(safety[key], key)
    if safety["max_tilt_rad"] >= np.pi / 2:
        raise ValueError("max_tilt_rad 必须小于 pi/2")
    for key in ("target_speed_rad_s", "joint_speed_rad_s", "tracking_error_rad", "takeover_position_error_rad"):
        if safety[key]:
            vector(safety[key], 12, key, positive=True)
    fw = raw["firmware"]
    booleans = [(safety, "allow_motion"), (control, "gains_verified"),
                (control, "standing_handover_verified"), (calibration["alignment"], "verified"),
                (calibration["odometry"], "verified"), (control, "prone_handover_verified"),
                (init, "verified"), (behavior, "feedback_verified")]
    booleans += [(fw, k) for k in ("joint_control_verified", "watchdog_verified", "shutdown_verified",
                                  "gain_units_verified", "robot_limits_verified")]
    if any(not isinstance(section[key], bool) for section, key in booleans):
        raise ValueError("所有 verified/allow_motion/confirmed 字段必须为 bool")
    if not isinstance(fw["watchdog_timeout_s"], (int, float)) or not math.isfinite(fw["watchdog_timeout_s"]) or fw["watchdog_timeout_s"] < 0:
        raise ValueError("watchdog_timeout_s 必须为非负有限数")
    if fw["robot_torque_limit_nm"]:
        vector(fw["robot_torque_limit_nm"], 12, "robot_torque_limit_nm", positive=True)
    joints = []
    entries = calibration.get("joints", [])
    if any(j.get("name") not in POLICY_JOINT_NAMES for j in entries):
        raise ValueError("关节覆盖项包含未知名称")
    for i, name in enumerate(POLICY_JOINT_NAMES):
        matches = [j for j in entries if j["name"] == name]
        if len(matches) > 1:
            raise ValueError(f"重复关节 {name}")
        lo, hi = OFFICIAL_JOINT_LIMITS[i]
        item = {"sdk_index": SDK_JOINT_NAMES.index(name), "sign": 1, "offset": 0.,
                "minimum": lo, "maximum": hi, **(matches[0] if matches else {})}
        index = item["sdk_index"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 12:
            raise ValueError("sdk_index 要求 0..11")
        if item["sign"] not in (-1, 1) or isinstance(item["sign"], bool):
            raise ValueError("sign 必须为 +1/-1")
        offset = float(item["offset"])
        lo, hi = item.get("minimum"), item.get("maximum")
        if not math.isfinite(offset) or any(v is not None and not math.isfinite(v) for v in (lo, hi)):
            raise ValueError("关节标定值必须有限")
        if (lo is None) != (hi is None) or (lo is not None and lo >= hi):
            raise ValueError("minimum/maximum 必须同时填写且 minimum < maximum")
        joints.append(JointCalibration(name, index, item["sign"], offset, lo, hi))
    if sorted(j.sdk_index for j in joints) != list(range(12)):
        raise ValueError("SDK 索引必须为 0..11 的一一映射")
    imu = calibration["imu"]
    matrix = vector(imu["sensor_to_body"], 9, "imu.sensor_to_body").reshape(3, 3)
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(matrix), 1):
        raise ValueError("imu.sensor_to_body 必须为旋转矩阵，det=+1")
    vector(imu["gyro_bias"], 3, "imu.gyro_bias")
    if imu["quaternion_frame"] not in ("sensor_to_world", "body_to_world"):
        raise ValueError("imu.quaternion_frame 要求 sensor_to_world/body_to_world")
    # 录制快照保存解析后的全部默认/覆盖项，避免重放依赖未来版本默认值。
    calibration["joints"] = [asdict(joint) for joint in joints]
    return Config(path, calibration_path.resolve(), raw, calibration, tuple(joints))
