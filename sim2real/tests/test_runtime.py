"""离线回归：使用模拟 SDK，绝不连接机器人。"""

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from pathlib import Path
import copy
import io
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim2real"))
sys.path.insert(0, str(ROOT))

from bpx_runtime.history import ObservationHistory
from bpx_runtime.joints import DEFAULT_JOINT_POSITION
from bpx_runtime.supervisor import BehaviorMode, SupervisorDecision
from bpx_sim2real.calibration import Calibration
from bpx_sim2real.cli import main
from bpx_sim2real.config import load_config
from bpx_sim2real.controller import ActorController
from bpx_sim2real.recording import JsonlWriter, load_samples, resample
from bpx_sim2real.runner import replay, run_live
from bpx_sim2real.safety import StateGuard, TargetGuard
from bpx_sim2real.sdk_robot import SDKSource
from bpx_sim2real.state import Sample


def config(ready=False):
    cfg = load_config(ROOT / "sim2real/config/runtime.toml")
    if not ready:
        return cfg
    raw, cal = copy.deepcopy(cfg.runtime), copy.deepcopy(cfg.calibration)
    cal["identity"] = {"serial_number": "test-SN", "robot_model": "BPX", "firmware_version": "1.0.15"}
    cal["alignment"]["verified"] = True
    cal["odometry"]["verified"] = True
    raw["safety"].update(
        allow_motion=True,
        target_speed_rad_s=[1.0] * 12,
        joint_speed_rad_s=[10.0] * 12,
        tracking_error_rad=[0.5] * 12,
        takeover_position_error_rad=[0.2] * 12,
    )
    raw["firmware"].update(
        joint_control_verified=True,
        watchdog_verified=True,
        shutdown_verified=True,
        gain_units_verified=True,
        robot_limits_verified=True,
        watchdog_timeout_s=0.2,
        robot_torque_limit_nm=[30.0] * 12,
        shutdown_action="zero_joint_command",
    )
    raw["control"].update(gains_verified=True, standing_handover_verified=True,
                          prone_handover_verified=True, send_rate_hz=100)
    raw["init_controller"]["verified"] = True
    raw["supervisor"]["feedback_verified"] = True
    joints = tuple(replace(j, minimum=-3.0, maximum=3.0) for j in cfg.joints)
    return replace(cfg, runtime=raw, calibration=cal, joints=joints)


def sample(cfg=None, t=0.0, token=1, **changes):
    cfg = cfg or config()
    q = Calibration(cfg).position_to_sdk(DEFAULT_JOINT_POSITION).tolist()
    data = dict(
        host_time=t,
        session=1,
        source="high_rate",
        q=q,
        dq=[0.0] * 12,
        torque=[0.0] * 12,
        quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        omega=[0.0] * 3,
        acceleration=[0.0, 0.0, 9.81],
        joint_token=token,
        imu_token=token,
        joint_age_s=0.0,
        imu_age_s=0.0,
        odom_token=token,
        odom_age_s=0.0,
        velocity_body=[0., 0., 0.],
        odom_position=[0., 0., .37],
        identity={**cfg.calibration["identity"], "sdk_version": "1.0.9"},
    )
    return Sample(**{**data, **changes})


class Policy:
    def __init__(self, value=0.2):
        self.value = value
        self.observations = []

    def __call__(self, obs):
        self.observations.append(obs.copy())
        return np.full(12, self.value, dtype=np.float32)


class Clock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def sleep(self, delta):
        self.value += delta


class Source:
    def __init__(self, cfg, clock, disconnect_at=None):
        self.cfg, self.clock = cfg, clock
        self.q = sample(cfg).q
        self.commands = []
        self.reads = 0
        self.exited = False
        self.disconnect_at = disconnect_at

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exited = True

    def read(self):
        self.reads += 1
        connected = self.disconnect_at is None or self.clock.now() < self.disconnect_at
        return sample(self.cfg, t=self.clock.now(), token=self.reads, q=list(self.q), connected=connected)

    def send(self, kp, target, kd, velocity, ff):
        self.commands.append(
            (self.clock.now(), np.asarray(kp).copy(), np.asarray(target).copy(), np.asarray(kd).copy())
        )
        self.q = list(target)


class RuntimeTest(unittest.TestCase):
    def test_template_refuses_motion_before_sdk_creation(self):
        factory = Mock()
        with self.assertRaisesRegex(RuntimeError, "未确认|未填写"):
            run_live(config(), "unused", (0, 0, 0), 1, armed=True, source_factory=factory)
        factory.assert_not_called()

    def test_preflight_never_construct_sdk(self):
        with patch(
            "bpx_sim2real.sdk_robot.SDKSource.__init__", side_effect=AssertionError("hardware")
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["check"]), 0)
            self.assertEqual(main(["run"]), 0)

    def test_mapping_sign_offset_round_trip(self):
        cfg = config()
        joints = tuple(replace(j, sign=(-1 if i % 2 else 1), offset=0.01 * i) for i, j in enumerate(cfg.joints))
        cal = Calibration(replace(cfg, joints=joints))
        expected = np.arange(12) * 0.1
        sdk = cal.position_to_sdk(expected)
        np.testing.assert_allclose(cal.position_from_sdk(sdk), expected)
        dq = np.arange(12) * 0.2
        sdk_dq = np.empty(12)
        sdk_dq[cal.indices] = cal.sign * dq
        np.testing.assert_allclose(cal.velocity_from_sdk(sdk_dq), dq)
        # 增益只重排，不随方向符号变成负数。
        np.testing.assert_allclose(cal.gains_to_sdk(np.arange(12))[cal.indices], np.arange(12))

    def test_imu_rotation_and_bias(self):
        cfg = config()
        cal = copy.deepcopy(cfg.calibration)
        cal["imu"].update(
            sensor_to_body=[0, -1, 0, 1, 0, 0, 0, 0, 1], gyro_bias=[0.1, 0, 0], quaternion_frame="sensor_to_world"
        )
        gravity, omega = Calibration(replace(cfg, calibration=cal)).imu([0, 0, 0, 1], [0.3, 0, 0])
        np.testing.assert_allclose(gravity, [0, 0, -1])
        np.testing.assert_allclose(omega, [0, 0.2, 0], atol=1e-8)

    def test_state_age_session_and_identity_failures(self):
        cfg = config(True)
        good = sample(cfg)
        guard = StateGuard(cfg)
        guard.check(good, hardware=True)
        for bad in (
            replace(good, connected=False),
            replace(good, session=2),
            replace(good, joint_age_s=0.11),
            replace(good, joint_age_s=None),
            replace(good, identity={}),
            replace(good, source="ordinary"),
            replace(good, q=[float("nan")] * 12),
            replace(good, imu_age_s=0.08),
        ):
            with self.assertRaises((RuntimeError, ValueError)):
                guard.check(bad, hardware=True)
        with self.assertRaisesRegex(RuntimeError, "过期"):
            guard.check(good, now=0.11, hardware=True)

    def test_target_limits_rate_and_tracking(self):
        cfg = config(True)
        q = np.array(DEFAULT_JOINT_POSITION)
        guard = TargetGuard(cfg, q)
        result, _, constrained = guard.preview(q + 0.4, q, np.zeros(12), 0.02)
        np.testing.assert_allclose(result, q + 0.02)
        self.assertTrue(constrained)
        guard.commit(result)
        with self.assertRaisesRegex(RuntimeError, "速度"):
            guard.preview(q, q, np.full(12, 11), 0.02)
        with self.assertRaisesRegex(RuntimeError, "跟踪"):
            guard.preview(q, q + 0.8, np.zeros(12), 0.02)

    def test_controller_first_command_holds_measured_pose(self):
        cfg = config()
        policy = Policy()
        controller = ActorController(cfg, policies={"walk": policy, "stand": policy})
        first = controller.step(sample(cfg), (0, 0, 0))
        np.testing.assert_allclose(first["target_model"], DEFAULT_JOINT_POSITION)
        self.assertEqual(len(first["observation"]), 450)
        self.assertEqual(first["takeover_alpha"], 0.0)

    def test_history_receives_processed_action(self):
        cfg = config()
        policy = Policy()
        controller = ActorController(cfg, policies={"walk": policy, "stand": policy})
        controller.step(sample(cfg), (0, 0, 0))
        target = np.array(DEFAULT_JOINT_POSITION) + 0.03
        controller.commit_target(target)
        controller.step(sample(cfg, t=0.02, token=2), (0, 0, 0))
        np.testing.assert_allclose(policy.observations[-1][-12:], 0.06, atol=1e-7)

    def test_actor_bounds_external_supervisor_command_before_observation(self):
        cfg = config()
        limits = np.asarray(cfg.control["command_limit"])
        for sign in (1., -1.):
            with self.subTest(sign=sign):
                policy = Policy(0.)
                controller = ActorController(cfg, policies={"walk": policy, "stand": policy})
                raw = sign * np.array([5., -2., 3.])
                decision = SupervisorDecision(BehaviorMode.WALK, raw, True, "test")
                result = controller.step(sample(cfg), raw, decision=decision)
                expected = np.clip(raw, -limits, limits)
                np.testing.assert_allclose(result["command"], expected)
                commands = policy.observations[-1][60:90].reshape(10, 3)
                np.testing.assert_allclose(commands, np.tile(expected, (10, 1)))

    def test_actor_rejects_invalid_supervisor_command_before_clipping(self):
        cfg = config()
        for command in ([np.inf, 0., 0.], [np.nan, 0., 0.], [1., 2.]):
            with self.subTest(command=command):
                policy = Policy(0.)
                controller = ActorController(cfg, policies={"walk": policy, "stand": policy})
                decision = SupervisorDecision(BehaviorMode.WALK, np.asarray(command), True, "test")
                with self.assertRaises(ValueError):
                    controller.step(sample(cfg), (0., 0., 0.), decision=decision)
                self.assertEqual(policy.observations, [])

    def test_replay_resampling_has_no_future_state_and_increases_age(self):
        cfg = config()
        samples = [sample(cfg, t=0, telemetry={"motion_age_s": 0.01}), sample(cfg, t=0.04, token=2)]
        result = list(resample(samples, 0.02))
        self.assertEqual([s.joint_token for s in result], [1, 1, 2])
        self.assertAlmostEqual(result[1].joint_age_s, 0.02)
        self.assertAlmostEqual(result[1].telemetry["motion_age_s"], 0.03)

    def test_offline_replay_with_injected_policy(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            path, output = Path(tmp) / "state.jsonl", Path(tmp) / "policy.jsonl"
            with JsonlWriter(path, {"mode": "test"}) as writer:
                for i in range(5):
                    writer.write({"type": "sample", "sample": sample(cfg, t=i * 0.02, token=i + 1).to_dict()})
            policy = Policy()
            with patch("bpx_sim2real.sdk_robot.SDKSource.__init__", side_effect=AssertionError("hardware")):
                count = replay(cfg, path, output, (0, 0, 0), policies={"walk": policy, "stand": policy})
            self.assertEqual(count, 5)
            self.assertTrue(output.is_file())

    def test_replay_skips_incomplete_initial_udp_prefix(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            path, output = Path(tmp) / "state.jsonl", Path(tmp) / "policy.jsonl"
            with JsonlWriter(path, {}) as writer:
                writer.write({"type": "sample", "sample": Sample(0.0, connected=True).to_dict()})
                for i in range(1, 4):
                    writer.write({"type": "sample", "sample": sample(cfg, t=i * 0.02, token=i).to_dict()})
            policy = Policy()
            count = replay(cfg, path, output, (0, 0, 0), policies={"walk": policy, "stand": policy})
            self.assertEqual(count, 3)
            self.assertEqual(json.loads(output.read_text().splitlines()[0])["skipped_initial_samples"], 1)

    def test_recording_never_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.jsonl"
            path.write_text("keep")
            with self.assertRaises(FileExistsError):
                with JsonlWriter(path, {}):
                    pass
            self.assertEqual(path.read_text(), "keep")

    def test_live_schedule_separates_policy_and_send_rates(self):
        cfg, clock = config(True), Clock()
        source = Source(cfg, clock)
        policy = Policy(0.1)
        with tempfile.TemporaryDirectory() as tmp:
            count = run_live(
                cfg,
                Path(tmp) / "log.jsonl",
                (0, 0, 0),
                0.1,
                armed=True,
                policies={"walk": policy, "stand": policy},
                source_factory=lambda *a, **k: source,
                clock=clock.now,
                sleep=clock.sleep,
            )
        self.assertEqual(count, 5)
        self.assertEqual(len(source.commands), 10)
        self.assertTrue(source.exited)
        np.testing.assert_allclose(source.commands[0][2], sample(cfg).q)

    def test_live_disconnect_stops_without_recovery(self):
        cfg, clock = config(True), Clock()
        source = Source(cfg, clock, disconnect_at=0.025)
        policy = Policy()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(RuntimeError, "断开"):
            run_live(
                cfg,
                Path(tmp) / "log.jsonl",
                (0, 0, 0),
                0.2,
                armed=True,
                policies={"walk": policy, "stand": policy},
                source_factory=lambda *a, **k: source,
                clock=clock.now,
                sleep=clock.sleep,
            )
        self.assertLessEqual(len(source.commands), 3)
        self.assertTrue(source.exited)

    def test_live_inference_overrun_does_not_send_late_target(self):
        cfg, clock = config(True), Clock()
        source = Source(cfg, clock)

        def slow(obs):
            clock.sleep(0.03)
            return np.zeros(12)

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(RuntimeError, "超期"):
            run_live(
                cfg,
                Path(tmp) / "log.jsonl",
                (0, 0, 0),
                0.2,
                armed=True,
                policies={"walk": slow, "stand": slow},
                source_factory=lambda *a, **k: source,
                clock=clock.now,
                sleep=clock.sleep,
            )
        self.assertEqual(source.commands, [])

    def test_config_rejects_duplicate_indices_and_false_as_string(self):
        original = (ROOT / "sim2real/config/runtime.toml").read_text()
        calibration = (ROOT / "sim2real/config/joint_calibration.template.toml").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.toml"
            calpath = Path(tmp) / "joint_calibration.template.toml"
            path.write_text(original.replace("allow_motion = false", 'allow_motion = "false"'))
            calpath.write_text(calibration)
            with self.assertRaises(ValueError):
                load_config(path)
            path.write_text(original)
            calpath.write_text(calibration + '\n[[joints]]\nname = "fr_hip_roll_joint"\nsdk_index = 0\n')
            with self.assertRaisesRegex(ValueError, "一一映射"):
                load_config(path)


class SDKTest(unittest.TestCase):
    def robot(self):
        r = Mock()
        r.connect.return_value = True
        r.isConnected.return_value = True
        r.getRobotModel.return_value = 1
        r.getRobotVersion.return_value = (1, 0, 15, 0, 0, 0)
        r.getRobotSerialNumber.return_value = "test-SN"
        r.getJointStateTimestamp.return_value = 10
        r.getImuTimestamp.return_value = 20
        r.getOdometryTimestamp.return_value = 30
        r.getMotionStateTimestamp.return_value = 40
        r.getJointPosition.return_value = [0.0] * 12
        r.getJointVelocity.return_value = [0.0] * 12
        r.getJointTorque.return_value = [0.0] * 12
        r.getImuQuat.return_value = [0, 0, 0, 1]
        r.getImuOmega.return_value = [0.0] * 3
        r.getImuAcc.return_value = [0.0, 0.0, 9.81]
        r.getLegOdom.return_value = {"velocity_body": [0.0] * 3}
        r.getMotorTemperature.return_value = [20.0] * 12
        r.getDriverTemperature.return_value = [20.0] * 12
        r.getCurrentMotionState.return_value = 0
        r.getCurrentGait.return_value = 0
        r.getControlMode.return_value = 1
        r.getBatteryLevel.return_value = 80
        r.setJointCommand.return_value = True
        r.setZeroJointCommand.return_value = True
        return r

    def test_readonly_object_has_no_motor_command_calls(self):
        r, clock = self.robot(), Clock()
        sdk = SimpleNamespace(RequestRobotState=Mock(return_value=r), JointLevelControl=Mock(), __version__="1.0.9")
        with SDKSource(config(), sdk=sdk, clock=clock.now) as source:
            source.read()
            clock.sleep(0.12)
            stale = source.read()
            self.assertAlmostEqual(stale.joint_age_s, 0.12)
            with self.assertRaisesRegex(RuntimeError, "只读"):
                source.send(*[[0.0] * 12] * 5)
        sdk.JointLevelControl.assert_not_called()
        r.setJointCommand.assert_not_called()
        r.setZeroJointCommand.assert_not_called()
        r.disconnect.assert_called_once()

    def test_high_rate_snapshot_retries_on_mid_read_update(self):
        r = self.robot()
        r.getJointStateSeqHighRate.side_effect = [1, 2, 2, 2]
        for name, size in [
            ("JointPosition", 12),
            ("JointVelocity", 12),
            ("JointTorque", 12),
            ("ImuQuat", 4),
            ("ImuOmega", 3),
            ("ImuAcc", 3),
        ]:
            getattr(r, "get" + name + "HighRate").return_value = [0.0] * size
        r.getImuQuatHighRate.return_value = [0, 0, 0, 1]
        r.getJointStateTimestampHighRate.return_value = 1.25
        sdk = SimpleNamespace(JointLevelControl=Mock(return_value=r), __version__="1.0.9")
        with SDKSource(config(True), joint_control=True, sdk=sdk) as source:
            state = source.read()
            self.assertEqual(state.joint_token, 2)
            self.assertEqual(r.getJointPositionHighRate.call_count, 2)

    def test_zero_command_is_attempted_after_send_failure(self):
        r = self.robot()
        r.setJointCommand.return_value = False
        sdk = SimpleNamespace(JointLevelControl=Mock(return_value=r), __version__="1.0.9")
        with self.assertRaisesRegex(RuntimeError, "发送失败"):
            with SDKSource(config(True), joint_control=True, sdk=sdk) as source:
                source.send(*[[0.0] * 12] * 5)
        r.setZeroJointCommand.assert_called_once()
        r.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
