"""Single-frame expert observation contracts, without loading trained models."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

SIM2SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM2SIM_DIR))

from bpx_sim2sim.policy import SingleFrameOnnxPolicy
from run_mujoco import load_optional_policy


class SingleFramePolicyTest(unittest.TestCase):
    def load_expert(self, shape):
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="obs", shape=shape)]
        session.get_outputs.return_value = [SimpleNamespace(name="actions")]
        session.run.return_value = [np.zeros((1, 12), dtype=np.float32)]
        runtime = SimpleNamespace(InferenceSession=Mock(return_value=session))
        with patch.dict(sys.modules, {"onnxruntime": runtime}), patch.object(Path, "is_file", return_value=True):
            policy = load_optional_policy("init", Path("init.onnx"), 48, 12)
        self.assertIsInstance(policy, SingleFrameOnnxPolicy)
        return policy, session

    def test_init_actor_excludes_linear_velocity(self):
        policy, session = self.load_expert([1, 45])
        observation = np.arange(48, dtype=np.float32)
        original = observation.copy()
        action = policy(observation)
        first_input = session.run.call_args.args[1]["obs"].copy()
        np.testing.assert_array_equal(first_input, original[3:].reshape(1, 45))
        np.testing.assert_array_equal(observation, original)
        self.assertEqual(action.shape, (12,))
        observation[:3] = [1000, -2000, 3000]
        policy(observation)
        np.testing.assert_array_equal(session.run.call_args.args[1]["obs"], first_input)

    def test_legacy_expert_retains_full_observation(self):
        policy, session = self.load_expert([1, 48])
        observation = np.arange(48, dtype=np.float32)
        policy(observation)
        np.testing.assert_array_equal(session.run.call_args.args[1]["obs"], observation.reshape(1, 48))

    def test_rejects_incompatible_model_dimensions(self):
        for shape in ([1, 450], [1, 46], [45], [1, "features"]):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "单帧专家输入"):
                self.load_expert(shape)

    def test_rejects_incomplete_source_observation(self):
        policy, session = self.load_expert([1, 45])
        with self.assertRaises(ValueError):
            policy(np.zeros(45, dtype=np.float32))
        session.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
