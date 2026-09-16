#!/usr/bin/env python3
"""比较 probe_isaaclab.py 的 JSON 与 MuJoCo CSV 轨迹。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rmse(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - reference))))


def csv_vector(row: dict[str, str], names: list[str]) -> np.ndarray:
    return np.asarray([float(row[name]) for name in names], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", type=Path, required=True, help="probe_isaaclab.py 生成的 JSON")
    parser.add_argument("--mujoco", type=Path, required=True, help="run_mujoco.py 生成的 CSV")
    args = parser.parse_args()

    isaac = json.loads(args.isaac.read_text(encoding="utf-8"))
    with args.mujoco.open(encoding="utf-8", newline="") as stream:
        mujoco_rows = list(csv.DictReader(stream))
    isaac_rows = isaac["trajectory"]
    if not isaac_rows:
        raise ValueError("Isaac JSON没有轨迹；运行 probe_isaaclab.py 时请设置 --steps")

    # Isaac轨迹第一项为 t=policy_dt，MuJoCo CSV第一项为 t=0，因此按时间寻找最近样本。
    comparisons: dict[str, list[float]] = {
        "base_linear_velocity_body": [],
        "base_angular_velocity_body": [],
        "projected_gravity_body": [],
        "joint_position": [],
        "joint_velocity": [],
        "action": [],
    }
    joint_names = isaac["joint_names"]
    mujoco_times = np.asarray([float(row["time"]) for row in mujoco_rows])
    for isaac_row in isaac_rows:
        target_time = float(isaac_row["time"])
        index = int(np.argmin(np.abs(mujoco_times - target_time)))
        mujoco_row = mujoco_rows[index]
        pairs = {
            "base_linear_velocity_body": (
                csv_vector(mujoco_row, ["body_vx", "body_vy", "body_vz"]),
                np.asarray(isaac_row["base_linear_velocity_body"]),
            ),
            "base_angular_velocity_body": (
                csv_vector(mujoco_row, ["body_wx", "body_wy", "body_wz"]),
                np.asarray(isaac_row["base_angular_velocity_body"]),
            ),
            "projected_gravity_body": (
                csv_vector(mujoco_row, ["gravity_x", "gravity_y", "gravity_z"]),
                np.asarray(isaac_row["projected_gravity_body"]),
            ),
            "joint_position": (
                csv_vector(mujoco_row, [f"joint_pos_{name}" for name in joint_names]),
                np.asarray(isaac_row["joint_position"]),
            ),
            "joint_velocity": (
                csv_vector(mujoco_row, [f"joint_vel_{name}" for name in joint_names]),
                np.asarray(isaac_row["joint_velocity"]),
            ),
            "action": (
                csv_vector(mujoco_row, [f"action_{name}" for name in joint_names]),
                np.asarray(isaac_row["action"]),
            ),
        }
        for name, (mujoco_value, isaac_value) in pairs.items():
            comparisons[name].append(rmse(mujoco_value, isaac_value))

    print(f"aligned_samples={len(isaac_rows)}")
    for name, errors in comparisons.items():
        print(f"{name:30s} rmse={np.sqrt(np.mean(np.square(errors))):.9g}  max_frame_rmse={max(errors):.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
