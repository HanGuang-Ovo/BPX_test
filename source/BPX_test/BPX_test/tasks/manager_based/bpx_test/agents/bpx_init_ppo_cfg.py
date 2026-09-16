# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BPX 趴卧起身（Init）策略的 RSL-RL 配置。"""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import BpxPpoRunnerCfg


@configclass
class BpxInitPPORunnerCfg(BpxPpoRunnerCfg):
    """Init 训练单独写入 ``logs/rsl_rl/bpx_init``。"""

    experiment_name = "bpx_init"
    # 起身比稳态站立/行走探索更难，允许更长训练；命令行仍可临时覆盖。
    max_iterations = 3000
