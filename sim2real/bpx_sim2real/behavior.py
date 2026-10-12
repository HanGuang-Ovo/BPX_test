"""实机反馈适配；行为状态机和起身轨迹与 sim2sim 共用。

SDK 没有足端接触传感器。平地接地估计使用已校验的里程计高度、
运动学足端高度及反馈力矩反解的向上支撑力，不把估计记作测量。
"""

import numpy as np

from bpx_runtime.init_controller import FootTrajectory, LegKinematics, quintic
from bpx_runtime.observation import actor_frame
from bpx_runtime.supervisor import BehaviorMode, BehaviorSupervisor, SupervisorState
from .controller import ActorController
from .diagnostics import attitude
from .config import vector


class HardwareFeedback:
    def __init__(self, config, calibration):
        self.config, self.cal = config, calibration
        # 官方 BPX URDF；与 monitor 的骨架尺寸相同。实机仅支持该几何。
        self.kinematics = LegKinematics(
            list(config.policy["joint_names"]), [.11522, -.11522, .11522, -.11522],
            .23, .2315648, [[j.minimum, j.maximum] for j in config.joints],
        )
        self.anchors = np.array([[.2386, .056, 0], [.2386, -.056, 0],
                                 [-.2386, .056, 0], [-.2386, -.056, 0]])

    def state(self, sample, trajectory_complete=True):
        settings = self.config.runtime["supervisor"]
        q = self.cal.position_from_sdk(sample.q)
        rotation = np.asarray(attitude(self.config, sample)["body_rotation"])
        gravity, angular = self.cal.imu(sample.quat_xyzw, sample.omega)
        tilt = float(np.arccos(np.clip(-gravity[2], -1, 1)))
        height = float(vector(sample.odom_position, 3, "odometry position")[2] - settings["ground_z"])
        if height < 0:
            raise RuntimeError("里程计高度低于配置地面；需重新校验 ground_z")
        feet = self.kinematics.forward(q)
        # 官方 toe collision 为沿脚部 y 轴的圆柱：中心 z=-0.012、半径 0.027。
        # 跟随机体及关节旋转计算圆柱最低点，避免把 toe link 原点当作足底。
        roll, pitch, knee = q[self.kinematics.indices].T
        sole_z_axis = np.column_stack((np.sin(pitch + knee),
                                       -np.sin(roll) * np.cos(pitch + knee),
                                       np.cos(roll) * np.cos(pitch + knee))) @ rotation.T
        cylinder_axis = np.column_stack((np.zeros(4), np.cos(roll), np.sin(roll))) @ rotation.T
        extent = (settings["toe_half_width"] * np.abs(cylinder_axis[:, 2])
                  + settings["toe_radius"] * np.sqrt(np.maximum(0., 1. - cylinder_axis[:, 2] ** 2)))
        heights = (height + ((feet + self.anchors) @ rotation.T)[:, 2]
                   - settings["toe_center_depth"] * sole_z_axis[:, 2] - extent)
        torque = self.cal.sign * vector(sample.torque, 12, "joint torque")[self.cal.indices]
        torque -= np.asarray(settings["torque_bias_nm"])
        # 中心差分 FK Jacobian。反馈力矩必须为已确认的 Nm，禁止沿用电流数值。
        jacobians = np.empty((4, 3, 3))
        for joint in range(3):
            perturb = np.zeros(12)
            perturb[self.kinematics.indices[:, joint]] = 1e-5
            jacobians[:, :, joint] = (
                self.kinematics.forward(q + perturb) - self.kinematics.forward(q - perturb)
            ) / 2e-5
        forces = np.zeros(4)
        valid = np.zeros(4, dtype=bool)
        for leg, jacobian in enumerate(jacobians):
            if np.linalg.cond(jacobian) <= settings["contact_jacobian_condition_max"]:
                # 静态近似 tau = -J^T F；忽略腿部重力/惯性，必须实机校验阈值。
                force_body = -np.linalg.solve(jacobian.T, torque[self.kinematics.indices[leg]])
                forces[leg] = (rotation @ force_body)[2]
                valid[leg] = True
        contacts = (valid & (np.abs(heights) <= settings["contact_height_tolerance"])
                    & (forces >= settings["contact_force_threshold_n"]))
        velocity = vector(sample.velocity_body, 3, "body velocity")
        state = SupervisorState(height, tilt, float(np.linalg.norm(velocity[:2])),
                                float(np.linalg.norm(angular)), int(contacts.sum()), trajectory_complete)
        details = dict(source="sdk_odometry+kinematic_torque_estimate", base_height=height,
                       feet_in_contact_estimated=int(contacts.sum()),
                       foot_height_estimated=heights.tolist(), support_force_estimated_n=forces.tolist(),
                       contact_estimate_valid=valid.tolist())
        return state, details


class HardwareController:
    def __init__(self, config, *, policies=None):
        self.config = config
        self.actor = ActorController(config, policies=policies)
        self.calibration = self.actor.calibration
        self.feedback = HardwareFeedback(config, self.calibration)
        self.trajectory = FootTrajectory(config.init_controller, self.feedback.kinematics)
        # 在创建 SDK 前验证配置中的趴姿及整段收脚轨迹。
        self.trajectory.start(0., config.control["prone_joint_position"])
        self.trajectory.started_at = None
        self.supervisor = BehaviorSupervisor(config.supervisor, init_available=True)
        self.mode = "stand"
        self.waiting_target = None
        self.target = None
        self.result = None
        self.kp = np.asarray(config.control["kp"], dtype=float)
        self.kd = np.asarray(config.control["kd"], dtype=float)
        self.gain_start = (self.kp.copy(), self.kd.copy())
        self.gain_started_at = None

    def initialize(self, sample):
        state, _ = self.feedback.state(sample)
        q = self.calibration.position_from_sdk(sample.q)
        prone = state.base_height < self.config.supervisor.fall_base_height
        reference = self.config.control["prone_joint_position"] if prone else self.calibration.default
        if np.any(np.abs(q - reference) > self.config.safety["takeover_position_error_rad"]):
            raise RuntimeError("实测姿态不符合已确认的趴姿/站姿接管范围")
        if state.tilt >= (self.config.supervisor.init_start_max_tilt if prone else self.config.supervisor.fall_tilt):
            raise RuntimeError("接管姿态倾角不支持；不执行侧翻/仰翻恢复")
        self.target = q.copy()
        self.actor.commit_target(q)

    def step(self, sample, command, *, now, policy_due, init_requested=False):
        state, details = self.feedback.state(sample, self.trajectory.complete)
        q = self.calibration.position_from_sdk(sample.q)
        # 额外的实机保护保持生效；不会因增加起身而放宽已有安全阈值。
        tilt_limit = min(self.config.safety["max_tilt_rad"],
                         self.config.supervisor.init_abort_tilt if self.mode == "init"
                         else self.config.supervisor.fall_tilt)
        if state.tilt >= tilt_limit:
            self.supervisor._transition(BehaviorMode.DISABLED, "hardware_tilt_abort")
            self.mode = "disabled"
            raise RuntimeError("DISABLED: 实机倾角超限")
        if self.mode == "init" and self.trajectory.timed_out(now):
            self.supervisor._transition(BehaviorMode.DISABLED, "init_pd_timeout")
            self.mode = "disabled"
            raise RuntimeError("DISABLED: init_pd_timeout")
        if policy_due:
            old_mode = self.mode
            decision = self.supervisor.update(command, state, self.config.period, init_requested)
            self.mode = decision.mode.value
            if self.mode == "disabled":
                raise RuntimeError(f"DISABLED: {decision.reason}")
            if decision.mode_changed:
                if self.mode == "waiting_init":
                    if np.any(np.abs(q - self.config.control["prone_joint_position"])
                              > self.config.safety["takeover_position_error_rad"]):
                        raise RuntimeError("DISABLED: unsupported_init_pose，关节不符合已验证趴姿")
                    self.waiting_target = q.copy()
                if self.mode == "init":
                    if np.any(np.abs(q - self.config.control["prone_joint_position"])
                              > self.config.safety["takeover_position_error_rad"]):
                        raise RuntimeError("DISABLED: unsupported_init_pose，关节不符合已验证趴姿")
                    self.trajectory.start(now, q)
                if old_mode in ("init", "waiting_init") and self.mode == "stand":
                    self.actor.started_at = None
                    self.actor.stand_history = None
                    self.actor.mode = "stand"
                if (old_mode == "init") != (self.mode == "init"):
                    self.gain_start = (self.kp.copy(), self.kd.copy())
                    self.gain_started_at = now
            if self.mode in ("stand", "walk", "stopping"):
                self.result = self.actor.step(sample, command, decision=decision)
                self.target = np.asarray(self.result["target_model"])
            else:
                gravity, angular = self.calibration.imu(sample.quat_xyzw, sample.omega)
                frame = actor_frame(angular, gravity, np.zeros(3), q,
                                    self.calibration.velocity_from_sdk(sample.dq), self.actor.last_action,
                                    self.calibration.default,
                                    sample.velocity_body if self.config.policy["include_base_lin_vel"] else None)
                self.actor.walk_history.append(frame)
                self.result = dict(mode=self.mode, command=[0., 0., 0.])
            self.result["reason"] = decision.reason
        if self.mode == "waiting_init":
            self.target = self.waiting_target.copy()
        elif self.mode == "init":
            self.target = self.trajectory.step(now, q, state.feet_in_contact == 4)
        desired_kp = np.full(12, self.config.init_controller.stiffness) if self.mode == "init" else np.asarray(self.config.control["kp"])
        desired_kd = np.full(12, self.config.init_controller.damping) if self.mode == "init" else np.asarray(self.config.control["kd"])
        alpha = (1. if self.gain_started_at is None else
                 quintic((now - self.gain_started_at) / self.config.control["action_blend_duration_s"]))
        self.kp = self.gain_start[0] + alpha * (desired_kp - self.gain_start[0])
        self.kd = self.gain_start[1] + alpha * (desired_kd - self.gain_start[1])
        self.result.update(mode=self.mode, target_model=self.target.tolist(), feedback=details,
                           init_phase=self.trajectory.phase if self.mode == "init" else "",
                           init_progress=self.trajectory.progress if self.mode == "init" else 0.,
                           init_requested=bool(init_requested))
        return self.result

    def commit_target(self, target):
        self.actor.commit_target(target)
        if self.mode == "init":
            # 轨迹完成必须以实际发送的受限目标为准。
            self.trajectory.target = np.asarray(target).copy()
