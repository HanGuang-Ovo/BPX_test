"""实机流程回归：模拟反馈和手柄，绝不连接硬件。"""

from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_runtime import config, sample, Clock, Source, Policy, ROOT
from bpx_sim2real.behavior import HardwareController, HardwareFeedback
from bpx_sim2real.calibration import Calibration
from bpx_sim2real.cli import main, parser
from bpx_sim2real.config import load_config
from bpx_sim2real.runner import run_live
from bpx_sim2real.safety import StateGuard, TargetGuard
from bpx_runtime.gamepad import (GamepadCommandSource, InputDevice, BTN_TR, EVENT_STRUCT, EV_KEY)
from bpx_runtime.supervisor import BehaviorMode, BehaviorSupervisor, SupervisorState


def supported_sample(cfg, q, t=0., token=1, *, supported=True):
    """四腿同步水平姿态；独立解析式生成已知 25 N/腿支撑力与足底高度。"""
    q = np.asarray(q, dtype=float)
    roll, pitch, knee = q.reshape(3, 4)
    if not np.allclose(roll, 0., atol=1e-7):
        raise ValueError("测试反馈仅用于 roll=0")
    feet_z = -.23 * np.cos(pitch) - .2315648 * np.cos(pitch + knee)
    feet_x = -.23 * np.sin(pitch) - .2315648 * np.sin(pitch + knee)
    height = -float(feet_z[0]) + .012 * np.cos(pitch[0] + knee[0]) + .027
    force = 25. if supported else 0.
    torque = np.concatenate((-np.array([.11522, -.11522, .11522, -.11522]) * force,
                             feet_x * force, -.2315648 * np.sin(pitch + knee) * force))
    cal = Calibration(cfg)
    sdk_torque = np.empty(12)
    sdk_torque[cal.indices] = cal.sign * torque
    return sample(cfg, t=t, token=token, q=cal.position_to_sdk(q).tolist(),
                  torque=sdk_torque.tolist(), odom_position=[0., 0., height])


class SupportedSource(Source):
    def __init__(self, cfg, clock, *, supported=True):
        super().__init__(cfg, clock)
        self.q = Calibration(cfg).position_to_sdk(cfg.control["prone_joint_position"]).tolist()
        self.supported = supported

    def read(self):
        self.reads += 1
        q = Calibration(self.cfg).position_from_sdk(self.q)
        return supported_sample(self.cfg, q, self.clock.now(), self.reads, supported=self.supported)


class HardwareBehaviorTest(unittest.TestCase):
    def test_live_constant_and_dynamic_commands_cannot_exceed_limits(self):
        for dynamic in (False, True):
            for sign in (1., -1.):
                with self.subTest(dynamic=dynamic, sign=sign), tempfile.TemporaryDirectory() as tmp:
                    cfg, clock = config(True), Clock()
                    source = Source(cfg, clock)
                    policies = {"walk": Policy(0.), "stand": Policy(0.)}
                    raw = sign * np.array([5., -2., 3.])
                    limits = np.asarray(cfg.control["command_limit"])
                    path = Path(tmp) / "live.jsonl"
                    run_live(cfg, path, (0., 0., 0.) if dynamic else raw, 2., armed=True,
                             policies=policies, source_factory=lambda *a, **k: source,
                             clock=clock.now, sleep=clock.sleep,
                             command_source=(lambda: raw) if dynamic else None)
                    self.assertTrue(policies["walk"].observations)
                    for observation in policies["walk"].observations:
                        commands = observation[60:90].reshape(10, 3)
                        self.assertTrue(np.all(np.abs(commands) <= limits + 1e-7))
                    np.testing.assert_allclose(policies["walk"].observations[-1][87:90],
                                               np.clip(raw, -limits, limits), atol=1e-7)
                    rows = [row for line in path.read_text().splitlines()
                            if (row := json.loads(line)).get("type") == "command"]
                    for row in rows:
                        np.testing.assert_allclose(row["raw_command"], raw)
                        self.assertTrue(np.all(np.abs(row["policy"]["command"]) <= limits + 1e-7))
                    self.assertTrue(source.exited)

    def run_source(self, cfg, clock, source, duration, command_source=None, request=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live.jsonl"
            policies = {"walk": Policy(0.), "stand": Policy(0.)}
            run_live(cfg, path, (0, 0, 0), duration, armed=True, policies=policies,
                     source_factory=lambda *a, **k: source, clock=clock.now, sleep=clock.sleep,
                     command_source=command_source, init_request_source=request)
            return [row for line in path.read_text().splitlines()
                    if (row := json.loads(line)).get("type") == "command"], policies

    def test_prone_waiting_never_runs_stand_or_moves_for_stick_input(self):
        cfg, clock = config(True), Clock()
        source = SupportedSource(cfg, clock)
        rows, policies = self.run_source(cfg, clock, source, .5, lambda: [1., .5, 1.])
        self.assertEqual({r["mode"] for r in rows}, {"waiting_init"})
        for _, _, target, _ in source.commands:
            np.testing.assert_allclose(target, source.commands[0][2], atol=1e-12)
        self.assertFalse(policies["stand"].observations)
        self.assertFalse(policies["walk"].observations)

    def test_rb_full_pipeline_to_stand_walk_stopping_and_stand(self):
        cfg, clock = config(True), Clock()
        source = SupportedSource(cfg, clock)
        fired = False

        def request():
            nonlocal fired
            if not fired and clock.now() >= .3:
                fired = True
                return True
            return False

        def command():
            return [.3, 0., 0.] if 5.5 <= clock.now() < 6.5 else [0., 0., 0.]

        rows, policies = self.run_source(cfg, clock, source, 8., command, request)
        modes = [r["mode"] for i, r in enumerate(rows) if i == 0 or r["mode"] != rows[i - 1]["mode"]]
        self.assertEqual(modes, ["waiting_init", "init", "stand", "walk", "stopping", "stand"])
        init = [r for r in rows if r["mode"] == "init"]
        self.assertGreaterEqual(init[0]["host_time"], .3)
        self.assertGreaterEqual(init[-1]["host_time"] - init[0]["host_time"], 4.45)
        for prev, cur in zip(source.commands, source.commands[1:]):
            self.assertLessEqual(np.max(np.abs(cur[2] - prev[2])), .010001)
        self.assertTrue(policies["stand"].observations)
        self.assertTrue(policies["walk"].observations)
        self.assertEqual(init[0]["kp_model"], cfg.control["kp"])
        self.assertAlmostEqual(max(r["kp_model"][0] for r in init), cfg.init_controller.stiffness)

    def test_no_support_pauses_and_times_out_without_stand(self):
        cfg, clock = config(True), Clock()
        source = SupportedSource(cfg, clock, supported=False)
        with self.assertRaisesRegex(RuntimeError, "init_pd_timeout"):
            self.run_source(cfg, clock, source, 10., request=lambda: True)
        self.assertTrue(source.exited)
        q = Calibration(cfg).position_from_sdk(source.q)
        feet = HardwareFeedback(cfg, Calibration(cfg)).kinematics.forward(q)
        np.testing.assert_allclose(feet[:, 2], -.19, atol=1e-5)

    def test_contact_requires_both_ground_proximity_and_upward_force(self):
        cfg = config(True)
        feedback = HardwareFeedback(cfg, Calibration(cfg))
        good = supported_sample(cfg, cfg.policy["default_joint_position"])
        state, details = feedback.state(good)
        self.assertEqual(state.feet_in_contact, 4)
        np.testing.assert_allclose(details["support_force_estimated_n"], 25., atol=1e-6)
        for bad in (replace(good, torque=[0.] * 12),
                    replace(good, odom_position=[0., 0., good.odom_position[2] + .1]),
                    replace(good, torque=(-np.asarray(good.torque)).tolist())):
            self.assertEqual(feedback.state(bad)[0].feet_in_contact, 0)

    def test_singular_contact_estimate_does_not_claim_support(self):
        cfg = config(True)
        q = [0.] * 4 + [.7] * 4 + [0.] * 4
        state, details = HardwareFeedback(cfg, Calibration(cfg)).state(supported_sample(cfg, q))
        self.assertEqual(state.feet_in_contact, 0)
        self.assertEqual(details["contact_estimate_valid"], [False] * 4)

    def test_pose_mismatch_fails_before_first_motor_command(self):
        cfg, clock = config(True), Clock()
        source = SupportedSource(cfg, clock)
        source.q = Calibration(cfg).position_to_sdk([0.] * 4 + [.5] * 4 + [-2.62] * 4).tolist()
        with self.assertRaisesRegex(RuntimeError, "接管范围"):
            self.run_source(cfg, clock, source, .5)
        self.assertEqual(source.commands, [])

    def test_handheld_disconnect_ends_session_without_reconnect(self):
        cfg, clock = config(True), Clock()
        source = SupportedSource(cfg, clock)
        command = Mock(side_effect=[[0., 0., 0.], OSError("手柄已断开")])
        with self.assertRaisesRegex(OSError, "手柄已断开"):
            self.run_source(cfg, clock, source, .5, command)
        self.assertTrue(source.exited)
        self.assertEqual(command.call_count, 2)

    def test_mid_init_tilt_aborts_before_another_motor_command(self):
        cfg, clock = config(True), Clock()
        source = SupportedSource(cfg, clock)
        read = source.read

        def tilted_read():
            result = read()
            if clock.now() >= .1:
                result = replace(result, quat_xyzw=[np.sin(.3), 0., 0., np.cos(.3)])
            return result

        source.read = tilted_read
        with self.assertRaisesRegex(RuntimeError, "DISABLED.*倾角"):
            self.run_source(cfg, clock, source, 1., request=lambda: True)
        self.assertLess(source.commands[-1][0], .1)
        self.assertTrue(source.exited)

    def test_odometry_stale_or_missing_fails_even_for_450_actor(self):
        cfg = config(True)
        good = sample(cfg)
        for bad in (replace(good, odom_age_s=.2), replace(good, odom_token=None),
                    replace(good, odom_position=None), replace(good, velocity_body=None)):
            with self.assertRaises((RuntimeError, ValueError)):
                StateGuard(cfg).check(bad, hardware=True, require_odometry=True)

    def test_init_gain_torque_guard_uses_actual_gains(self):
        cfg = config(True)
        q = np.asarray(cfg.policy["default_joint_position"])
        cfg.safety["target_speed_rad_s"] = [10.] * 12
        cfg.firmware["robot_torque_limit_nm"] = [1.] * 12
        guard = TargetGuard(cfg, q)
        with self.assertRaisesRegex(RuntimeError, "力矩"):
            guard.preview(q + .02, q, np.zeros(12), .01, kp=[80.] * 12, kd=[2.] * 12)

    def test_unbounded_run_keeps_running_until_operator_interrupt(self):
        cfg, clock = config(True), Clock()
        source = SupportedSource(cfg, clock)

        def command():
            if clock.now() > 10.1:
                raise KeyboardInterrupt()
            return [0., 0., 0.]

        with self.assertRaises(KeyboardInterrupt):
            self.run_source(cfg, clock, source, None, command)
        self.assertGreater(source.commands[-1][0], 10.)
        self.assertTrue(source.exited)

    def test_cli_defaults_gamepad_unbounded_and_wires_rb_and_lb(self):
        args = parser().parse_args(["run"])
        self.assertEqual(args.command_source, "gamepad")
        self.assertIsNone(args.duration)
        gamepad = Mock()
        gamepad.__enter__ = Mock(return_value=gamepad)
        gamepad.__exit__ = Mock()
        with patch("bpx_sim2real.cli.load_config", return_value=config(True)), \
             patch("bpx_sim2real.cli.require_ready"), \
             patch("bpx_sim2real.cli.select_gamepad", return_value=Mock()), \
             patch("bpx_sim2real.cli.GamepadCommandSource", return_value=gamepad) as factory, \
             patch("bpx_sim2real.cli.run_live", return_value=1) as run, \
             patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(main(["run", "--live", "--arm", "--output", "unused", "--gamepad-deadman"]), 0)
        self.assertEqual(factory.call_args.kwargs["init_button"], BTN_TR)
        self.assertEqual(factory.call_args.kwargs["deadman_button"], 310)
        self.assertEqual(run.call_args.kwargs["command_source"], gamepad.read)
        self.assertEqual(run.call_args.kwargs["init_request_source"], gamepad.consume_init_request)

    def test_rb_edge_is_consumed_once_and_hold_repeat_does_not_trigger(self):
        source = GamepadCommandSource(InputDevice(Path("/dev/test"), "test", 0, 0, ""), (1., .5, 1.), init_button=BTN_TR)
        source._file_descriptor = 123
        source._buttons[BTN_TR] = False
        event = lambda value: EVENT_STRUCT.pack(0, 0, EV_KEY, BTN_TR, value)
        for value, expected in ((1, True), (2, False), (0, False), (1, True)):
            with patch("bpx_runtime.gamepad.os.read", side_effect=[event(value), BlockingIOError()]):
                source._drain_events()
            self.assertEqual(source.consume_init_request(), expected)
            self.assertFalse(source.consume_init_request())

    def test_dropped_gamepad_events_fail_instead_of_using_old_command(self):
        from bpx_runtime.gamepad import EV_SYN, SYN_DROPPED
        source = GamepadCommandSource(InputDevice(Path("/dev/test"), "test", 0, 0, ""), (1., .5, 1.), init_button=BTN_TR)
        source._file_descriptor = 123
        packet = EVENT_STRUCT.pack(0, 0, EV_SYN, SYN_DROPPED, 0)
        with patch("bpx_runtime.gamepad.os.read", return_value=packet), self.assertRaisesRegex(OSError, "事件丢失"):
            source.read()

    def test_invalid_init_settings_are_rejected_on_load(self):
        text = (ROOT / "sim2real/config/runtime.toml").read_text()
        cal = (ROOT / "sim2real/config/joint_calibration.template.toml").read_text()
        for old, new in (("duration = 4.0", "duration = 0.0"),
                         ("support_duration = 1.0", "support_duration = 5.0"),
                         ("init_base_height = 0.36", "init_base_height = 0.1"),
                         ("feedback_verified = false", 'feedback_verified = "true"')):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "runtime.toml"
                path.write_text(text.replace(old, new))
                (Path(tmp) / "joint_calibration.template.toml").write_text(cal)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_matching_sim2sim_supervisor_and_config(self):
        import sys
        sys.path.insert(0, str(ROOT / "sim2sim"))
        from bpx_sim2sim.config import load_config as sim_config
        from bpx_sim2sim.supervisor import BehaviorSupervisor as SimSupervisor
        real = config()
        sim = sim_config(ROOT / "sim2sim/config/bpx_terrain_history.toml")
        self.assertIs(SimSupervisor, BehaviorSupervisor)
        self.assertEqual(real.supervisor, sim.supervisor)
        self.assertEqual(real.init_controller, sim.init_controller)


if __name__ == "__main__":
    unittest.main()
