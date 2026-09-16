# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BPX 行走策略的 RSL-RL 配置。"""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import BpxPpoRunnerCfg


@configclass
class BpxLocomotionPPORunnerCfg(BpxPpoRunnerCfg):
    """行走训练单独写入 ``logs/rsl_rl/bpx_locomotion``。"""

    experiment_name = "bpx_locomotion"
