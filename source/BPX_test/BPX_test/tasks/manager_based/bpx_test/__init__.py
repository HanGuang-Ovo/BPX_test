# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


def _register_bpx_task(task_id: str, env_cfg: str, runner_cfg: str) -> None:
    """用统一入口注册一个 BPX ManagerBasedRLEnv 任务。"""

    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.{runner_cfg}",
        },
    )


# 新任务名明确表达每个 checkpoint 所学习的行为。
_register_bpx_task(
    "BPX-Locomotion-v0",
    "bpx_locomotion_env_cfg:BpxLocomotionEnvCfg",
    "bpx_locomotion_ppo_cfg:BpxLocomotionPPORunnerCfg",
)
_register_bpx_task(
    "BPX-Stand-v0",
    "bpx_stand_env_cfg:BpxStandEnvCfg",
    "bpx_stand_ppo_cfg:BpxStandPPORunnerCfg",
)
_register_bpx_task(
    "BPX-Init-v0",
    "bpx_init_env_cfg:BpxInitEnvCfg",
    "bpx_init_ppo_cfg:BpxInitPPORunnerCfg",
)

# 保留旧任务名和原日志目录，已有训练/回放命令无需立刻修改。
_register_bpx_task(
    "BPX-Test-v0",
    "bpx_test_env_cfg:BpxTestEnvCfg",
    "rsl_rl_ppo_cfg:PPORunnerCfg",
)

_register_bpx_task(
    "BPX-Locomotion-Rough-v0",
    "bpx_rough_env_cfg:BpxRoughEnvCfg",
    "bpx_rough_ppo_cfg:BpxRoughPPORunnerCfg",
)

_register_bpx_task(
    "BPX-Locomotion-Rough-History-v0",
    "bpx_rough_history_env_cfg:BpxRoughHistoryEnvCfg",
    "bpx_rough_history_ppo_cfg:BpxRoughHistoryPPORunnerCfg",
)

_register_bpx_task(
    "BPX-Stand-Rough-v0",
    "bpx_rough_stand_env_cfg:BpxRoughStandEnvCfg",
    "bpx_rough_stand_ppo_cfg:BpxRoughStandPPORunnerCfg",
)
