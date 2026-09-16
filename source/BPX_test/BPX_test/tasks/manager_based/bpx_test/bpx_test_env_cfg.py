# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""旧版 BPX 环境名称的兼容导入层。

新代码应分别从 ``bpx_base_env_cfg`` 或 ``bpx_locomotion_env_cfg`` 导入。
保留本文件是为了让已有脚本、笔记和 ``BPX-Test-v0`` 训练命令不会因重构立即失效。
"""

from .bpx_base_env_cfg import (
    BPX_CFG,
    BPX_POLICY_JOINT_NAMES,
    BPX_USD_PATH,
    BpxActionsCfg,
    BpxObservationsCfg,
    BpxSceneCfg,
)
from .bpx_locomotion_env_cfg import (
    BpxLocomotionCommandsCfg,
    BpxLocomotionEnvCfg,
    BpxLocomotionEventCfg,
    BpxLocomotionRewardsCfg,
    BpxLocomotionTerminationsCfg,
)

# 旧类名别名。别名不会创建新的配置类型，实际内容来自重构后的模块。
BpxTestSceneCfg = BpxSceneCfg
ActionsCfg = BpxActionsCfg
ObservationsCfg = BpxObservationsCfg
CommandsCfg = BpxLocomotionCommandsCfg
EventCfg = BpxLocomotionEventCfg
RewardsCfg = BpxLocomotionRewardsCfg
TerminationsCfg = BpxLocomotionTerminationsCfg
BpxTestEnvCfg = BpxLocomotionEnvCfg

__all__ = [
    "BPX_CFG",
    "BPX_POLICY_JOINT_NAMES",
    "BPX_USD_PATH",
    "ActionsCfg",
    "BpxTestEnvCfg",
    "BpxTestSceneCfg",
    "CommandsCfg",
    "EventCfg",
    "ObservationsCfg",
    "RewardsCfg",
    "TerminationsCfg",
]
