"""Check compact scalar logging and lossless source-preserving log snapshots."""

import ast
import hashlib
import importlib.util
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def stairs_pattern():
    tree = ast.parse((ROOT / "source/BPX_test/BPX_test/tasks/manager_based/bpx_test/agents/"
                      "bpx_stairs_history_ppo_cfg.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    field = next(node for node in cls.body if isinstance(node, ast.AnnAssign)
                 and node.target.id == "tensorboard_scalar_filter")
    return ast.literal_eval(field.value)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = load_module("test_metrics_filter", ROOT / "scripts/rsl_rl/tensorboard_metrics.py")
snapshot = load_module("test_log_snapshot", ROOT / "scripts/filter_tensorboard_logs.py")


class FakeWriter:
    def __init__(self):
        self.calls = []

    def add_scalar(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "written"

    def flush(self):
        return "flushed"


class TensorboardMetricsTests(unittest.TestCase):
    def test_default_keeps_exactly_the_26_task_and_health_metrics(self):
        expression = re.compile(stairs_pattern())
        expected = {f"Curriculum/terrain_levels/{kind}/{metric}"
                    for kind in ("up", "down", "flat")
                    for metric in ("success_rate", "failure_rate", "timeout_rate", "actual_speed", "command_speed")}
        expected.update({"Curriculum/terrain_levels/" + tag for tag in (
            "up/terrain_level", "down/terrain_level", "command_stage", "terrain_sampling_phase",
            "flat/stable_fraction", "flat/max_unstable_time")})
        expected.update({"Train/mean_reward", "Train/mean_episode_length", "Policy/mean_noise_std",
                         "Loss/value_function", "Loss/learning_rate"})
        self.assertEqual(len(expected), 26)
        self.assertTrue(all(expression.fullmatch(tag) for tag in expected))
        for tag in ("Episode_Reward/action_rate_l2", "Perf/total_fps", "Loss/surrogate",
                    "Train/mean_reward/time", "Curriculum/terrain_levels/up/forward/success_rate",
                    "Curriculum/terrain_levels/flat/terrain_level", "Curriculum/terrain_levels/up/velocity_x",
                    "Curriculum/terrain_levels/up/window_samples"):
            self.assertIsNone(expression.fullmatch(tag), tag)

    def test_writer_preserves_arguments_results_and_non_scalar_operations(self):
        writer = FakeWriter()
        filtered = metrics.ScalarFilterWriter(writer, stairs_pattern())
        self.assertEqual(filtered.add_scalar("Train/mean_reward", 3.5, 12, walltime=7.), "written")
        self.assertIsNone(filtered.add_scalar("Train/mean_reward/time", 3.5, 7.))
        self.assertEqual(writer.calls, [(("Train/mean_reward", 3.5, 12), {"walltime": 7.})])
        self.assertEqual(filtered.flush(), "flushed")

    def test_runner_filter_is_scoped_and_restores_writer_even_on_error(self):
        class Parent:
            def log(self, locs, width=80, pad=35):
                self.seen = (locs, width, pad)
                self.writer.add_scalar("Train/mean_reward", 1, 3)
                self.writer.add_scalar("Perf/total_fps", 100, 3)
                if locs.get("raise"):
                    raise RuntimeError("test error")
                return "logged"

        runners = ModuleType("rsl_rl.runners")
        runners.OnPolicyRunner = Parent
        with patch.dict(sys.modules, {"rsl_rl": ModuleType("rsl_rl"), "rsl_rl.runners": runners,
                                      "torch": ModuleType("torch"), "tensorboard_metrics": metrics}):
            module = load_module("test_curriculum_runner", ROOT / "scripts/rsl_rl/curriculum_runner.py")
        runner = module.CurriculumOnPolicyRunner()
        runner.cfg = {"tensorboard_scalar_filter": stairs_pattern()}
        runner.logger_type = "tensorboard"
        writer = runner.writer = FakeWriter()
        locs = {"episode": "original input"}
        self.assertEqual(runner.log(locs, width=90, pad=20), "logged")
        self.assertIs(runner.seen[0], locs)
        self.assertEqual(runner.seen[1:], (90, 20))
        self.assertEqual(len(writer.calls), 1)
        self.assertIs(runner.writer, writer)
        with self.assertRaisesRegex(RuntimeError, "test error"):
            runner.log({"raise": True})
        self.assertIs(runner.writer, writer)
        for logger, pattern in (("wandb", stairs_pattern()), ("tensorboard", None)):
            runner.logger_type = logger
            runner.cfg = {"tensorboard_scalar_filter": pattern}
            writer.calls.clear()
            runner.log(locs)
            self.assertEqual(len(writer.calls), 2)

    def test_snapshot_preserves_scalar_values_steps_times_and_original_file(self):
        from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "compact"
            writer = EventFileWriter(str(source / "run1"))
            writer.add_event(Event(step=7, wall_time=123., summary=Summary(value=[
                Summary.Value(tag="Train/mean_reward", simple_value=3.5),
                Summary.Value(tag="Perf/total_fps", simple_value=100.)])))
            writer.add_event(Event(step=8, wall_time=124., summary=Summary(value=[
                Summary.Value(tag="Train/mean_reward", simple_value=4.5)])))
            writer.close()
            original = next(source.rglob("events.out.tfevents.*"))
            digest = hashlib.sha256(original.read_bytes()).hexdigest()
            stats = snapshot.filter_logs(source, output, stairs_pattern())
            self.assertEqual(stats[0]["original_tags"], 2)
            self.assertEqual(stats[0]["retained_tags"], ["Train/mean_reward"])
            self.assertEqual(stats[0]["retained_values"], 2)
            events = list(LegacyEventFileLoader(str(next(output.rglob("events.out.tfevents.*")))).Load())
            values = [(value.tag, event.step, event.wall_time, value.simple_value)
                      for event in events for value in event.summary.value]
            self.assertEqual(values, [("Train/mean_reward", 7, 123., 3.5),
                                      ("Train/mean_reward", 8, 124., 4.5)])
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), digest)

    def test_snapshot_refuses_overwrites_or_overlapping_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            for destination in (source, source / "nested", Path(temp)):
                with self.assertRaises(ValueError):
                    snapshot.filter_logs(source, destination, stairs_pattern())
            existing = Path(temp) / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                snapshot.filter_logs(source, existing, stairs_pattern())


if __name__ == "__main__":
    unittest.main()
