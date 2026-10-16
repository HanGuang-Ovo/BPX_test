"""Guard against reporting a broken training environment as ready. No GPU required."""

import contextlib
import importlib.metadata
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import check_dependencies as check


class TrainingSetupTests(unittest.TestCase):
    def test_driver_command_failure_is_required_failure(self):
        result = subprocess.CompletedProcess([], 9, "", "driver unavailable")
        with patch.object(check.shutil, "which", return_value="/mock/nvidia-smi"), \
             patch.object(check.subprocess, "run", return_value=result):
            item = check.check_gpu()
        self.assertFalse(item.ok)
        self.assertTrue(item.required)
        self.assertIn("driver unavailable", item.detail)

    def test_cuda_runtime_failure_is_required_failure(self):
        result = subprocess.CompletedProcess([], 1, "", "CUDA allocation failed")
        with patch.object(check.subprocess, "run", return_value=result):
            item = check.check_training_runtime("cuda:0")
        self.assertFalse(item.ok)
        self.assertTrue(item.required)

    def test_stale_metadata_does_not_count_as_installed(self):
        with patch.object(check.importlib.metadata, "version", return_value="2.3.3"), \
             patch.object(check.importlib.util, "find_spec", return_value=None):
            self.assertFalse(check.check_package("RSL", "rsl-rl-lib").ok)

    def test_local_scripts_namespace_is_not_rsl_rl(self):
        with patch.object(check.importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError), \
             patch.object(check.importlib.util, "find_spec", return_value=SimpleNamespace(origin=None)):
            self.assertFalse(check.check_package("RSL", "rsl-rl-lib").ok)

    def test_bpx_install_requires_exact_repo_path(self):
        spec = SimpleNamespace(submodule_search_locations=[str(check.PROJECT_ROOT) + "_old/source/BPX_test/BPX_test"])
        with patch.object(check.importlib.util, "find_spec", return_value=spec):
            self.assertFalse(check.check_bpx_install().ok)

    def test_empty_usd_is_not_valid_asset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "robot.usd").touch()
            with patch.object(check, "PROJECT_ROOT", root):
                self.assertFalse(check.check_assets([Path("robot.usd")], "USD", "copy asset")[0].ok)

    def run_fake_training(self, body):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = root / "scripts/rsl_rl/train.py"
            entry.parent.mkdir(parents=True)
            entry.write_text(body)
            with patch.object(check, "PROJECT_ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                return check.smoke_test("BPX-Stand-v0", "cuda:0", 4, 10)

    def test_smoke_exit_zero_without_final_checkpoint_fails(self):
        self.assertEqual(self.run_fake_training("print('simulator closed early')\n"), 1)

    def test_smoke_requires_final_checkpoint_after_second_iteration(self):
        body = """from pathlib import Path
p = Path('logs/rsl_rl/check/run/model_1.pt')
p.parent.mkdir(parents=True)
p.write_bytes(b'fake checkpoint')
"""
        self.assertEqual(self.run_fake_training(body), 0)

    def test_smoke_nonzero_exit_even_with_checkpoint_fails(self):
        body = """from pathlib import Path
p = Path('logs/rsl_rl/check/run/model_1.pt')
p.parent.mkdir(parents=True)
p.write_bytes(b'fake checkpoint')
raise SystemExit(1)
"""
        self.assertEqual(self.run_fake_training(body), 1)

    def test_full_install_dry_run_does_not_create_source_directory(self):
        installer = Path(check.__file__).resolve().parent / "install_training_deps.sh"
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "path with spaces" / "IsaacLab"
            result = subprocess.run(
                ["bash", str(installer), "--full", "--dry-run", "--python", sys.executable,
                 "--isaaclab", str(target)], cwd=temp, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(target.parent.exists())
            self.assertIn("v2.2.1", result.stdout)
            self.assertIn("2.3.3", result.stdout)
            self.assertIn("未执行安装或检查", result.stdout)


if __name__ == "__main__":
    unittest.main()
