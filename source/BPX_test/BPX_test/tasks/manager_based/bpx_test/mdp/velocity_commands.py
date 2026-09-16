# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BPX locomotion-specific velocity command generators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class StratifiedVelocityCommand(UniformVelocityCommand):
    """Sample standing, pure-yaw, and mixed velocity commands as disjoint modes."""

    cfg: StratifiedVelocityCommandCfg

    def __init__(self, cfg: StratifiedVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if self.cfg.heading_command:
            raise ValueError("StratifiedVelocityCommand 不支持 heading_command=True")
        if not 0.0 <= self.cfg.rel_pure_rotation_envs <= 1.0:
            raise ValueError("rel_pure_rotation_envs 必须位于 [0, 1]")
        if not 0.0 <= self.cfg.rel_standing_envs <= 1.0:
            raise ValueError("rel_standing_envs 必须位于 [0, 1]")
        if self.cfg.rel_pure_rotation_envs + self.cfg.rel_standing_envs > 1.0:
            raise ValueError("纯旋转和静止采样比例之和不能超过 1")

        min_speed, max_speed = self.cfg.pure_rotation_speed_range
        if min_speed < 0.0 or max_speed <= min_speed:
            raise ValueError("pure_rotation_speed_range 必须满足 0 <= min < max")
        negative_limit = -self.cfg.ranges.ang_vel_z[0]
        positive_limit = self.cfg.ranges.ang_vel_z[1]
        if min(negative_limit, positive_limit) < max_speed:
            raise ValueError("ang_vel_z 的正负范围必须覆盖 pure_rotation_speed_range")

    def __str__(self) -> str:
        msg = super().__str__()
        mixed_probability = 1.0 - self.cfg.rel_standing_envs - self.cfg.rel_pure_rotation_envs
        msg += f"\n\tPure rotation probability: {self.cfg.rel_pure_rotation_envs}"
        msg += f"\n\tMixed motion probability: {mixed_probability}"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample mutually exclusive command modes for the requested environments."""

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._resample_command(env_ids_tensor)

        # Use one categorical draw so standing, pure rotation, and mixed motion
        # have the configured total probabilities rather than overlapping masks.
        category = torch.rand(env_ids_tensor.numel(), device=self.device)
        standing_mask = category < self.cfg.rel_standing_envs
        pure_rotation_mask = torch.logical_and(
            category >= self.cfg.rel_standing_envs,
            category < self.cfg.rel_standing_envs + self.cfg.rel_pure_rotation_envs,
        )
        self.is_standing_env[env_ids_tensor] = standing_mask

        pure_rotation_ids = env_ids_tensor[pure_rotation_mask]
        if pure_rotation_ids.numel() == 0:
            return

        self.vel_command_b[pure_rotation_ids, :2] = 0.0
        min_speed, max_speed = self.cfg.pure_rotation_speed_range
        magnitudes = torch.empty(pure_rotation_ids.numel(), device=self.device).uniform_(min_speed, max_speed)
        signs = torch.where(
            torch.rand(pure_rotation_ids.numel(), device=self.device) < 0.5,
            -torch.ones_like(magnitudes),
            torch.ones_like(magnitudes),
        )
        self.vel_command_b[pure_rotation_ids, 2] = signs * magnitudes


@configclass
class StratifiedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for stratified locomotion command sampling."""

    class_type: type = StratifiedVelocityCommand

    rel_pure_rotation_envs: float = 0.25
    """Probability of sampling a pure-yaw command with zero planar velocity."""

    pure_rotation_speed_range: tuple[float, float] = (0.2, 1.0)
    """Absolute yaw-speed range for pure-rotation commands in rad/s."""
