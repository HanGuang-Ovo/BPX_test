# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BPX 任务使用的课程学习函数。"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
根据最近 episode 的成功率调整奖励项权重。
逻辑如下：
- 最近 8192 个 episode 作为滑动窗口
- 至少收集 4096 个 episode 后才开始调整
- 成功率 ≤70%：权重保持 -0.1
- 成功率 70%–90%：线性调整到 -1.0
- 成功率 ≥90%：权重为 -1.0
- 成功率下降时，惩罚权重会自动减弱
成功定义为：episode 结束时仍满足 recovered 的高度、姿态、四足接触和前足宽度条件。
"""

class modify_reward_weight_by_success_rate(ManagerTermBase):
    """根据最近 episode 的成功率调整奖励项权重。"""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        window_size = cfg.params["window_size"]
        minimum_episodes = cfg.params["minimum_episodes"]
        if window_size <= 0:
            raise ValueError("window_size 必须大于零")
        if minimum_episodes <= 0 or minimum_episodes > window_size:
            raise ValueError("minimum_episodes 必须位于 (0, window_size] 区间")
        self._outcomes: deque[float] = deque(maxlen=window_size)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        success_term_name: str,
        initial_weight: float,
        final_weight: float,
        start_success_rate: float,
        full_success_rate: float,
        minimum_episodes: int,
        window_size: int,
    ) -> dict[str, float]:
        del window_size
        if not math.isfinite(initial_weight) or not math.isfinite(final_weight):
            raise ValueError("奖励权重必须是有限值")
        if not 0.0 <= start_success_rate < full_success_rate <= 1.0:
            raise ValueError("成功率阈值必须满足 0 <= start_success_rate < full_success_rate <= 1")

        # CurriculumManager 在场景复位前调用本项，因此这里读取到的是 episode 最后一帧状态。
        episode_lengths = env.episode_length_buf[env_ids]
        completed = episode_lengths > 0
        if torch.any(completed):
            success_cfg = env.reward_manager.get_term_cfg(success_term_name)
            success = success_cfg.func(env, **success_cfg.params)[env_ids][completed] > 0.5
            self._outcomes.extend(success.detach().to(device="cpu", dtype=torch.float32).tolist())

        num_episodes = len(self._outcomes)
        success_rate = sum(self._outcomes) / num_episodes if num_episodes > 0 else 0.0
        if num_episodes < minimum_episodes:
            weight = initial_weight
        else:
            progress = (success_rate - start_success_rate) / (full_success_rate - start_success_rate)
            progress = min(max(progress, 0.0), 1.0)
            weight = initial_weight + progress * (final_weight - initial_weight)

        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if term_cfg.weight != weight:
            term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, term_cfg)

        return {
            "success_rate": success_rate,
            "weight": weight,
            "window_episodes": float(num_episodes),
        }
