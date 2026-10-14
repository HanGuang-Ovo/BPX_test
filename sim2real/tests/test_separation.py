"""部署/调试边界及新旧命令入口回归；仅使用录制和模拟 SDK。"""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from test_runtime import ROOT, Clock, Policy, Source, config, sample
from bpx_runtime.jsonl import JsonlWriter
from bpx_sim2real.cli import main as deployment_main
from bpx_sim2real.controller import ActorController
from bpx_sim2real_tools.capture import capture
from bpx_sim2real_tools.cli import main as tools_main


class SeparationTest(unittest.TestCase):
    def write_recording(self, path):
        with JsonlWriter(path, {"mode": "test"}) as log:
            for tick in range(5):
                log.write({"type": "sample", "sample": sample(t=tick * .02).to_dict()})

    def test_deployment_works_without_tools_or_gui_and_never_imports_sdk(self):
        # 新进程排除测试模块已经导入的工具，且不把 tools 目录加入搜索路径。
        code = textwrap.dedent("""
            import builtins
            import contextlib
            import io
            import sys
            from pathlib import Path

            root = Path(sys.argv[1])
            sys.path[:0] = [str(root / 'sim2real'), str(root)]
            original = builtins.__import__

            def guarded(name, *args, **kwargs):
                if name.split('.')[0] in ('bpx_sim2real_tools', 'tkinter', 'bpx_sdk') or name == 'bpx_runtime.supervisor':
                    raise AssertionError('部署不应导入: ' + name)
                return original(name, *args, **kwargs)

            builtins.__import__ = guarded
            from bpx_sim2real.cli import main
            with contextlib.redirect_stdout(io.StringIO()):
                assert main(['--config', str(root / 'sim2real/config/runtime.toml'), 'check']) == 0
                assert main(['--config', str(root / 'sim2real/config/runtime.toml'), 'run']) == 0
            assert not any(name.startswith('bpx_sim2real_tools') for name in sys.modules)
        """)
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT)],
                                    cwd=tmp, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_new_and_forwarded_analyze_commands_match_without_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recording.jsonl"
            self.write_recording(path)
            results = []
            # 分析工具完全不需要窗口、策略推理或硬件对象。
            with patch.dict(sys.modules, {"bpx_sim2real_tools.monitor": None,
                                         "bpx_sim2real_tools.replay": None}), \
                 patch("bpx_sim2real.sdk_robot.SDKSource.__init__", side_effect=AssertionError("hardware")):
                for main in (tools_main, deployment_main):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        self.assertEqual(main(["analyze", str(path)]), 0)
                    results.append(json.loads(output.getvalue()))
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0]["samples"], 5)

    def test_new_and_forwarded_replay_commands_match_with_actual_onnx(self):
        cfg = config()
        if any(not cfg.model_path(mode).is_file() for mode in ("stand", "walk")):
            self.skipTest("仓库未附带本地模型")
        try:
            import onnxruntime
        except ImportError:
            self.skipTest("当前解释器未安装 ONNX Runtime")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recording.jsonl"
            self.write_recording(path)
            policies = []
            with patch.dict(sys.modules, {"bpx_sim2real_tools.monitor": None}), \
                 patch("bpx_sim2real.sdk_robot.SDKSource.__init__", side_effect=AssertionError("hardware")):
                for index, main in enumerate((tools_main, deployment_main)):
                    output = Path(tmp) / f"policy-{index}.jsonl"
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(main(["replay", str(path), "--output", str(output), "--vx", ".2"]), 0)
                    policies.append([json.loads(line) for line in output.read_text().splitlines()
                                     if json.loads(line).get("type") == "policy"])
            self.assertEqual(len(policies[0]), 5)
            self.assertEqual(policies[0], policies[1])

    def test_capture_keeps_readonly_mode_and_recording_format(self):
        from bpx_sim2real_tools.recording import load_samples

        cfg, clock = config(), Clock()
        source = Source(cfg, clock)
        modes = []

        def factory(config, *, joint_control):
            modes.append(joint_control)
            return source

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recording.jsonl"
            with patch("bpx_sim2real_tools.capture.time.monotonic", clock.now), \
                 patch("bpx_sim2real_tools.capture.time.sleep", clock.sleep), redirect_stdout(io.StringIO()):
                report = capture(cfg, path, .05, 100., source_factory=factory)
            metadata, samples = load_samples(path)
        self.assertEqual(modes, [False])
        self.assertEqual(metadata["mode"], "observe")
        self.assertEqual(len(samples), report["samples"])
        self.assertTrue(samples)
        self.assertEqual(source.commands, [])
        self.assertTrue(source.exited)

    def test_actor_routes_bounded_commands_and_rejects_invalid_input(self):
        cfg = config()
        policy = Policy(0.)
        controller = ActorController(cfg, policies={"stand": policy, "walk": policy})
        for tick in range(100):
            result = controller.step(sample(cfg, t=tick * .02), [5., -2., 3.])
            controller.commit_target(result["target_model"])
        np.testing.assert_allclose(result["command"], cfg.control["command_limit"] * np.array([1., -1., 1.]))
        for observation in policy.observations:
            self.assertTrue(np.all(np.abs(observation[60:90].reshape(10, 3))
                                   <= np.asarray(cfg.control["command_limit"]) + 1e-7))
        for command in ([np.nan, 0., 0.], [np.inf, 0., 0.], [1., 2.]):
            with self.subTest(command=command), self.assertRaises(ValueError):
                controller.step(sample(cfg), command)

    def test_removed_arm_flag_and_unknown_deployment_options_are_rejected(self):
        for option in ("--arm", "--unknown"):
            with self.subTest(option=option), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                deployment_main(["run", option])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
