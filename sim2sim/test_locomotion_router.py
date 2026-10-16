"""Proprioceptive router unit behavior and the unified walk-actor contract."""
from pathlib import Path
from unittest.mock import Mock, patch
import unittest
import numpy as np
from bpx_sim2sim.config import load_config, TerrainConfig
from bpx_sim2sim.locomotion_router import StairPolicyRouter, ProprioceptionRoutingConfig
from bpx_sim2sim.behaviors import BehaviorPolicies
from bpx_sim2sim.runner import Sim2SimRunner
from bpx_sim2sim.supervisor import BehaviorMode, SupervisorDecision
from test_history import RecordingPolicy


class RouterTests(unittest.TestCase):
    def test_imu_entry_stable_exit_and_standing_freeze(self):
        router = StairPolicyRouter(ProprioceptionRoutingConfig())
        flat = np.array([0., 0., -1.])
        tilted = np.array([np.sin(.25), 0., -np.cos(.25)])
        for _ in range(100):
            self.assertFalse(router.update(flat, np.zeros(12), .02))
        # A short impact must not switch experts.
        router.update(tilted, np.ones(12), .02)
        self.assertFalse(router.on_stairs)
        for _ in range(60):
            router.update(tilted, np.zeros(12), .02)
        self.assertTrue(router.on_stairs)
        self.assertEqual(router.reason, 'imu_tilt')
        for _ in range(200):
            router.update(flat, np.zeros(12), .02, moving=False)
        self.assertTrue(router.on_stairs)
        for _ in range(250):
            router.update(flat, np.zeros(12), .02)
        self.assertFalse(router.on_stairs)
        self.assertEqual(router.reason, 'stable_proprioception')

    def test_joint_error_entry_and_startup_grace(self):
        router = StairPolicyRouter(ProprioceptionRoutingConfig())
        for _ in range(30):
            self.assertFalse(router.update([0, 0, -1], np.ones(12), .02))
        for _ in range(50):
            router.update([0, 0, -1], np.ones(12), .02)
        self.assertTrue(router.on_stairs)
        self.assertEqual(router.reason, 'joint_tracking_error')
        with self.assertRaises(ValueError):
            ProprioceptionRoutingConfig(exit_tilt=.2)

    def test_normal_stair_gait_peak_error_can_return_to_flat(self):
        router = StairPolicyRouter(ProprioceptionRoutingConfig())
        router.on_stairs = True
        error = np.full(12, .08)
        error[0] = .28  # Above the old .22 peak exit limit, but normal whole-body RMS.
        for _ in range(250):
            router.update([0, 0, -1], error, .02)
        self.assertFalse(router.on_stairs)
        self.assertEqual(router.reason, 'stable_proprioception')

    def test_exit_window_tolerates_brief_gait_peaks(self):
        router = StairPolicyRouter(ProprioceptionRoutingConfig(filter_time=.001))
        router.on_stairs = True
        for i in range(250):
            error = np.full(12, .5 if i % 50 < 3 else .1)
            router.update([0, 0, -1], error, .02)
        self.assertFalse(router.on_stairs)

    def test_persistent_obstruction_or_tilt_blocks_exit(self):
        for gravity, error in (([0, 0, -1], [.5] + [0.] * 11),
                               ([.2, 0, -.98], [0.] * 12)):
            router = StairPolicyRouter(ProprioceptionRoutingConfig())
            router.on_stairs = True
            for _ in range(300):
                router.update(gravity, np.array(error), .02)
            self.assertTrue(router.on_stairs)

    def test_walk_and_stopping_always_use_the_unified_locomotion_actor(self):
        cfg = load_config(Path(__file__).parent / 'config/bpx_stairs_history.toml')
        for mode, command in ((BehaviorMode.WALK, (.2, 0., 0.)),
                              (BehaviorMode.STOPPING, (0., 0., 0.)),
                              (BehaviorMode.WALK, (0., 0., .6))):
            supervisor = Mock()
            supervisor.update.return_value = SupervisorDecision(
                mode, np.array(command, dtype=np.float32), False, 'test')
            locomotion, flat = RecordingPolicy(450, 0), RecordingPolicy(450, 0)
            runner = Sim2SimRunner(cfg, locomotion, supervisor, BehaviorPolicies(locomotion, flat=flat))
            runner.run(command, duration=.025, realtime=False, print_interval=100)
            self.assertTrue(locomotion.inputs)
            # The legacy flat expert stays loaded but is never selected; terrain
            # difficulty no longer switches the walking policy.
            self.assertFalse(flat.inputs)

    def test_walk_history_stays_continuous_across_the_single_actor(self):
        cfg = load_config(Path(__file__).parent / 'config/bpx_stairs_history.toml')
        locomotion, flat = RecordingPolicy(450, 0), RecordingPolicy(450, 0)
        supervisor = Mock()
        supervisor.update.return_value = SupervisorDecision(
            BehaviorMode.WALK, np.array([.2, 0, 0], dtype=np.float32), False, 'test')
        runner = Sim2SimRunner(cfg, locomotion, supervisor, BehaviorPolicies(locomotion, flat=flat))
        frames = [np.full(48, i, dtype=np.float32) for i in range(3)]
        with patch.object(runner.robot, 'observation', side_effect=frames):
            runner.run((.2, 0, 0), duration=.05, realtime=False, print_interval=100)
        self.assertEqual(len(flat.inputs), 0)
        self.assertEqual(len(locomotion.inputs), 3)
        # Angular velocity history is term-major; the first frame fills all
        # ten slots, then newer frames push in from the right.
        np.testing.assert_array_equal(locomotion.inputs[0][:30], [0.0] * 30)
        np.testing.assert_array_equal(locomotion.inputs[1][:30], [0] * 27 + [1] * 3)
        np.testing.assert_array_equal(locomotion.inputs[2][:30], [0] * 24 + [1] * 3 + [2] * 3)


if __name__ == '__main__':
    unittest.main()
