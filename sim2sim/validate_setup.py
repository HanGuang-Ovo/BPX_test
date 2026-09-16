#!/usr/bin/env python3
"""不依赖策略运行时，验证 BPX MJCF、关节映射、观测和 PD 控制。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from bpx_sim2sim.config import load_config
from bpx_sim2sim.robot import BpxMujocoRobot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config" / "bpx_flat.toml")
    parser.add_argument("--steps", type=int, default=200, help="零动作 PD 测试的物理步数")
    args = parser.parse_args()

    config = load_config(args.config)
    model = mujoco.MjModel.from_xml_path(str(config.paths.mjcf))
    data = mujoco.MjData(model)
    robot = BpxMujocoRobot(model, data, config)
    robot.reset()

    zero_action = np.zeros(config.observation.action_dimension, dtype=np.float32)
    observation = robot.observation(np.asarray(config.command.as_tuple(), dtype=np.float32), zero_action)
    print(f"MJCF: {config.paths.mjcf}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(
        f"physics_dt={model.opt.timestep}, policy_dt={config.simulation.policy_dt}, "
        f"integrator={config.simulation.integrator}, control={config.control.mode}"
    )
    print(f"observation={observation.shape}, action={zero_action.shape}")
    print("joint order:")
    for index, name in enumerate(config.joint_names):
        print(f"  {index:2d}: {name}")

    max_torque = 0.0
    for _ in range(args.steps):
        _, torque = robot.apply_pd(zero_action)
        max_torque = max(max_torque, float(np.max(np.abs(torque))))
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise FloatingPointError("零动作 PD 测试出现 NaN 或 Inf")
    print(
        f"zero-action PD: {args.steps} steps passed, height={robot.base_height():.4f}m, "
        f"tilt={robot.tilt_angle():.4f}rad, max_torque={max_torque:.4f}Nm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
