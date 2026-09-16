# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""旧 Recovery 配置名的导入兼容层。

起身策略已统一改名为 Init，新的代码和训练命令应导入 ``bpx_init_env_cfg``。这里保留
别名，避免已有脚本因模块路径改变而立刻失效；它不再注册独立的 Recovery Gym 任务。
"""

from .bpx_init_env_cfg import (
    BpxInitCommandsCfg as BpxRecoveryCommandsCfg,
    BpxInitEnvCfg as BpxRecoveryEnvCfg,
    BpxInitEventCfg as BpxRecoveryEventCfg,
    BpxInitRewardsCfg as BpxRecoveryRewardsCfg,
    BpxInitTerminationsCfg as BpxRecoveryTerminationsCfg,
)
