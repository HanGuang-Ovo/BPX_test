"""Stand and Init observation contracts, without loading trained models."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

SIM2SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM2SIM_DIR))

from bpx_sim2sim.policy import SingleFrameOnnxPolicy, StandOnnxPolicy
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

    def test_stand_accepts_history_and_legacy_models(self):
        for dimension in (450, 48, 45):
            with self.subTest(dimension=dimension):
                session = Mock()
                session.get_inputs.return_value = [SimpleNamespace(name="obs", shape=[1, dimension])]
                session.get_outputs.return_value = [SimpleNamespace(name="actions")]
                session.run.return_value = [np.zeros((1, 12), dtype=np.float32)]
                runtime = SimpleNamespace(InferenceSession=Mock(return_value=session))
                with patch.dict(sys.modules, {"onnxruntime": runtime}), patch.object(Path, "is_file", return_value=True):
                    policy = load_optional_policy("stand", Path("stand.onnx"), 48, 12)
                self.assertIsInstance(policy, StandOnnxPolicy)
                self.assertEqual(policy.uses_history, dimension == 450)
                source = np.arange(450 if dimension == 450 else 48, dtype=np.float32)
                policy(source)
                expected = source[3:] if dimension == 45 else source
                np.testing.assert_array_equal(session.run.call_args.args[1]["obs"], expected.reshape(1, dimension))

    def test_stand_rejects_unknown_dimension(self):
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="obs", shape=[1, 480])]
        session.get_outputs.return_value = [SimpleNamespace(name="actions")]
        runtime = SimpleNamespace(InferenceSession=Mock(return_value=session))
        with patch.dict(sys.modules, {"onnxruntime": runtime}), patch.object(Path, "is_file", return_value=True):
            with self.assertRaisesRegex(ValueError, "Stand 输入"):
                load_optional_policy("stand", Path("stand.onnx"), 48, 12)

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
