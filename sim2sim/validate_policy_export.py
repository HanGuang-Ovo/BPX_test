#!/usr/bin/env python3
"""比较 TorchScript 与 ONNX 对同一批观测的策略输出。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from bpx_sim2sim.config import load_config
from bpx_sim2sim.policy import OnnxPolicy, TorchScriptPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config" / "bpx_flat.toml")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()

    config = load_config(args.config)
    torchscript = TorchScriptPolicy(
        config.paths.torchscript_policy,
        config.observation.dimension,
        config.observation.action_dimension,
    )
    onnx = OnnxPolicy(
        config.paths.onnx_policy,
        config.observation.dimension,
        config.observation.action_dimension,
    )
    generator = np.random.default_rng(args.seed)
    maximum_error = 0.0
    for _ in range(args.samples):
        observation = generator.standard_normal(config.observation.dimension).astype(np.float32)
        error = float(np.max(np.abs(torchscript(observation) - onnx(observation))))
        maximum_error = max(maximum_error, error)
    print(f"samples={args.samples}, max_abs_error={maximum_error:.9g}, tolerance={args.tolerance:.9g}")
    if maximum_error > args.tolerance:
        print("FAILED: TorchScript与ONNX输出超过误差阈值")
        return 1
    print("PASSED: TorchScript与ONNX策略输出一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
