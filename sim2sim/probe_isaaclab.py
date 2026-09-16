#!/usr/bin/env python3
"""从 Isaac Lab 环境提取策略接口，用于和 MuJoCo 对拍。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", type=Path, default=None, help="TorchScript策略；默认使用本项目导出文件")
parser.add_argument("--vx", type=float, default=0.2)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--steps", type=int, default=0, help="额外执行的策略步数")
parser.add_argument("--json-output", type=Path, default=None)
parser.add_argument("--skip-policy", action="store_true", help="使用零动作，仅提取接口和物理响应")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import BPX_test.tasks  # noqa: F401, E402
from BPX_test.tasks.manager_based.bpx_test.bpx_locomotion_env_cfg import BpxLocomotionEnvCfg  # noqa: E402


def tensor_list(tensor: torch.Tensor) -> list[float]:
    return tensor[0].detach().cpu().to(torch.float64).tolist()


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    policy_path = args.policy or (
        repository_root / "logs/rsl_rl/bpx_flat/2026-09-08_11-28-30/exported/policy.pt"
    )
    # 对拍基准明确使用行走任务；不再依赖旧 BPX-Test-v0 兼容别名。
    env_cfg = BpxLocomotionEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    env_cfg.seed = 42
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.base_velocity.debug_vis = False
    # 接口探测使用确定性参数，不启用训练时材质/增益随机化和推力事件。
    env_cfg.events.physics_material = None
    env_cfg.events.actuator_gains = None
    env_cfg.events.push_robot = None
    # 必须保留 reset 事件来把状态写回默认值，只把随机采样范围收缩为零。
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    print("[probe] 正在创建 Isaac Lab 环境……", flush=True)
    env = gym.make("BPX-Locomotion-v0", cfg=env_cfg)
    base_env = env.unwrapped
    try:
        print("[probe] 环境已创建，正在重置……", flush=True)
        observation_dict, _ = env.reset()
        command = torch.tensor([[args.vx, args.vy, args.wz]], device=base_env.device)
        command_term = base_env.command_manager.get_term("base_velocity")
        command_term.vel_command_b[:] = command
        policy = None if args.skip_policy else torch.jit.load(str(policy_path), map_location=base_env.device).eval()

        print("[probe] 正在读取观测和策略动作……", flush=True)
        print("[probe] 读取 policy 观测……", flush=True)
        observation = observation_dict["policy"].clone()
        # reset() 时指令已经被采样；这里改为用户指定指令，其他状态项保持原样。
        observation[:, 9:12] = command
        print("[probe] 计算首帧动作……", flush=True)
        if policy is None:
            action = torch.zeros((1, 12), device=base_env.device)
        else:
            with torch.inference_mode():
                action = policy(observation)

        print("[probe] 从 sim2sim 配置读取已确认的关节元数据……", flush=True)
        sim2sim_config_path = repository_root / "sim2sim/config/bpx_flat.toml"
        with sim2sim_config_path.open("rb") as stream:
            sim2sim_config = tomllib.load(stream)
        print("[probe] sim2sim 配置读取完成……", flush=True)
        joint_names = list(sim2sim_config["robot"]["joint_names"])
        default_joint_position = list(sim2sim_config["initial_state"]["default_joint_position"])
        default_joint_position_tensor = torch.tensor(
            default_joint_position, dtype=observation.dtype, device=observation.device
        ).unsqueeze(0)
        print("[probe] 默认关节角张量创建完成……", flush=True)
        # 这些项来自 PolicyCfg；避免在存在 Warp/驱动兼容问题时再次访问 manager 元数据。
        observation_term_names = [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
        observation_term_dimensions = [[3], [3], [3], [3], [12], [12], [12]]
        print("[probe] 从已计算的 policy 观测读取机器人状态……", flush=True)
        base_linear_velocity_body = tensor_list(observation[:, 0:3])
        base_angular_velocity_body = tensor_list(observation[:, 3:6])
        projected_gravity_body = tensor_list(observation[:, 6:9])
        print("[probe] 首帧观测已转换为 JSON 数据……", flush=True)
        result: dict[str, object] = {
            "joint_names": joint_names,
            "default_joint_position": default_joint_position,
            "observation_term_names": observation_term_names,
            "observation_term_dimensions": observation_term_dimensions,
            "observation": tensor_list(observation),
            "action": tensor_list(action),
            "base_linear_velocity_body": base_linear_velocity_body,
            "base_angular_velocity_body": base_angular_velocity_body,
            "projected_gravity_body": projected_gravity_body,
        }

        trajectory: list[dict[str, object]] = []
        for step in range(args.steps):
            command_term.vel_command_b[:] = command
            if policy is not None:
                with torch.inference_mode():
                    action = policy(observation)
            observation_dict, _, _, _, _ = env.step(action)
            observation = observation_dict["policy"]
            joint_position = observation[:, 12:24] + default_joint_position_tensor
            trajectory.append(
                {
                    "step": step + 1,
                    "time": (step + 1) * base_env.step_dt,
                    "base_linear_velocity_body": tensor_list(observation[:, 0:3]),
                    "base_angular_velocity_body": tensor_list(observation[:, 3:6]),
                    "projected_gravity_body": tensor_list(observation[:, 6:9]),
                    "joint_position": tensor_list(joint_position),
                    "joint_velocity": tensor_list(observation[:, 24:36]),
                    "action": tensor_list(action),
                }
            )
        result["trajectory"] = trajectory

        output = json.dumps(result, ensure_ascii=False, indent=2)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(output + "\n", encoding="utf-8")
            print(f"[probe] 接口提取完成：{args.json_output}", flush=True)
        else:
            print("[probe] 接口提取完成。", flush=True)
            print(output)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
