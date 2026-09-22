"""Run: python -m unittest discover -s sim2sim -p 'test_history.py' -v"""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bpx_sim2sim.behaviors import BehaviorPolicies
from bpx_sim2sim.config import load_config
from bpx_sim2sim.history import ObservationHistory
from bpx_sim2sim.runner import Sim2SimRunner
from bpx_sim2sim.supervisor import BehaviorMode, SupervisorDecision


class RecordingPolicy:
    def __init__(self, dimension, action):
        self.dimension = dimension
        self.action = action
        self.inputs = []

    def __call__(self, observation):
        assert observation.shape == (self.dimension,)
        self.inputs.append(observation.copy())
        return np.full(12, self.action, dtype=np.float32)


class HistoryTest(unittest.TestCase):
    def test_layout_fill_and_rollover(self):
        history = ObservationHistory(10)
        frames = [np.arange(48, dtype=np.float32) + 100 * i for i in range(13)]
        # Independent reference: explicit term/time/feature traversal.
        boundaries = (0, 3, 6, 9, 12, 24, 36, 48)
        for i, frame in enumerate(frames):
            result = history.append(frame)
            times = [max(0, t) for t in range(i - 9, i + 1)]
            expected = [frames[t][j] for lo, hi in zip(boundaries, boundaries[1:])
                        for t in times for j in range(lo, hi)]
            np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(ObservationHistory(1).append(frames[-1]), frames[-1])

    def test_runner_reset_and_previous_action(self):
        cfg = load_config(Path(__file__).parent / 'config/bpx_terrain_history.toml')
        policy = RecordingPolicy(480, .1)
        runner = Sim2SimRunner(cfg, policy)
        for _ in range(2):
            policy.inputs.clear()
            runner.run((.2, 0, 0), duration=.05, realtime=False, print_interval=100)
            self.assertGreaterEqual(len(policy.inputs), 2)
            np.testing.assert_array_equal(policy.inputs[0][360:], 0)
            np.testing.assert_allclose(policy.inputs[1][-12:], .1)
            np.testing.assert_array_equal(policy.inputs[1][360:-12], 0)

    def test_supervisor_mixed_dimensions_and_actual_action(self):
        cfg = load_config(Path(__file__).parent / 'config/bpx_terrain_history.toml')
        walk = RecordingPolicy(480, .2)
        stand = RecordingPolicy(48, -.1)
        init = RecordingPolicy(48, .1)
        supervisor = Mock()
        supervisor.update.side_effect = [
            SupervisorDecision(mode, np.zeros(3, dtype=np.float32), True, 'test')
            for mode in (BehaviorMode.INIT, BehaviorMode.STAND, BehaviorMode.WALK)
        ]
        runner = Sim2SimRunner(cfg, walk, supervisor, BehaviorPolicies(walk, stand, init))
        applied = []
        desired_position = runner.robot.desired_joint_position

        def record_action(action):
            applied.append((float(runner.data.time), action.copy()))
            return desired_position(action)

        with patch.object(runner.robot, 'desired_joint_position', side_effect=record_action):
            runner.run((0, 0, 0), duration=.05, realtime=False, print_interval=100)
        self.assertEqual([len(p.inputs) for p in (init, stand, walk)], [1, 1, 1])
        # Compare the history with the actual mixed actions sent to the PD target.
        np.testing.assert_array_equal(walk.inputs[0][360:-24], 0)
        init_action = [action for time, action in applied if time < .02 - 1e-6][-1]
        stand_action = [action for time, action in applied if time < .04 - 1e-6][-1]
        np.testing.assert_array_equal(walk.inputs[0][-24:-12], init_action)
        np.testing.assert_array_equal(walk.inputs[0][-12:], stand_action)
        self.assertFalse(np.allclose(walk.inputs[0][-12:], stand.action))


if __name__ == '__main__':
    unittest.main()
