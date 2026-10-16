"""Episode-level terrain sampling with forward/backward stair traversal."""

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class MixedStairVelocityCommand(UniformVelocityCommand):
    # Terrain IDs and column order: up=0, down=1, flat=2.

    @property
    def mode_probabilities(self):
        # Preserve mode IDs for metrics: forward=0, backward=2.
        return (1.0, 0.0, 0.0, 0.0) if self.stage == 0 else (0.5, 0.0, 0.5, 0.0)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.initial_stage not in (0, 1) or cfg.initial_sideward_ramp != 0:
            raise ValueError("楼梯仅支持 stage 0 前进、stage 1 前进/后退，sideward_ramp 必须为 0")
        if cfg.heading_command or cfg.rel_standing_envs != 0.0:
            raise ValueError("楼梯指令要求 heading_command=False 且 rel_standing_envs=0")
        for name, limits in (("forward", cfg.ranges.lin_vel_x), ("backward", cfg.backward_speed_range)):
            if not 0.0 < limits[0] <= limits[1]:
                raise ValueError(f"{name} 速度幅值必须满足 0 < lower <= upper")
        self.stage = cfg.initial_stage
        # Retained zero-valued phase fields preserve version-2 checkpoint/log compatibility.
        self.sideward_ramp = 0
        self.episode_mode = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.episode_stage = torch.full_like(self.episode_mode, self.stage)
        self.episode_sideward_ramp = torch.zeros_like(self.episode_mode)
        self.next_mode = torch.zeros_like(self.episode_mode)
        self.next_stage = torch.zeros_like(self.episode_mode)
        self.next_sideward_ramp = torch.zeros_like(self.episode_mode)
        self.pending_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        # Lightweight CPU tests may use a minimal command stub; the real
        # UniformVelocityCommand always provides this mask.
        if not hasattr(self, "is_standing_env"):
            self.is_standing_env = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        # IsaacLab 基类按 rel_standing_envs 在所有重采样环境中抽取站立掩码，并且
        # 每个策略步都把站立环境的指令清零；这会让部分楼梯回合被静音而永远无法
        # 登顶。平地站立比例只能由下面的 rel_flat_standing_envs 控制。
        probabilities = cfg.traversal_probabilities
        if len(probabilities) != 3 or any(p <= 0 for p in probabilities) or abs(sum(probabilities) - 1) > 1.e-6:
            raise ValueError("traversal_probabilities 必须为三个正数且总和为 1")
        for name in ("descent_initial_speed_range", "descent_final_speed_range"):
            low, high = getattr(cfg, name)
            if not 0 < low <= high:
                raise ValueError(f"{name} 必须满足 0 < lower <= upper")
        pure_probability = getattr(cfg, "rel_pure_rotation_envs", 0.25)
        standing_probability = getattr(cfg, "rel_flat_standing_envs", 0.10)
        if not 0.0 <= pure_probability <= 1.0:
            raise ValueError("rel_pure_rotation_envs 必须位于 [0, 1]")
        if not 0.0 <= standing_probability <= 1.0 or standing_probability + pure_probability > 1.0:
            raise ValueError("平地静止与纯旋转采样比例必须位于 [0, 1] 且总和不超过 1")
        low, high = getattr(cfg, "pure_rotation_speed_range", (0.2, 1.0))
        if not 0.0 <= low < high:
            raise ValueError("pure_rotation_speed_range 必须满足 0 <= lower < upper")
        for name in ("flat_lin_vel_x", "flat_lin_vel_y", "flat_ang_vel_z"):
            flat_low, flat_high = getattr(cfg, name, (-1.0, 1.0))
            if not flat_low < flat_high:
                raise ValueError(f"{name} 必须满足 lower < upper")
        yaw_low, yaw_high = getattr(cfg, "flat_ang_vel_z", (-1.0, 1.0))
        if min(-yaw_low, yaw_high) < high:
            raise ValueError("flat_ang_vel_z 范围必须覆盖 pure_rotation_speed_range")
        self.episode_traversal = torch.zeros_like(self.episode_mode)
        self.next_traversal = torch.zeros_like(self.episode_mode)
        self.episode_standing = torch.zeros_like(self.episode_mode, dtype=torch.bool)
        self.next_descent_speed_range = torch.zeros((env.num_envs, 2), device=self.device)
        self.traversal_probabilities = tuple(float(p) for p in cfg.traversal_probabilities)

    def set_traversal_probabilities(self, probabilities):
        probabilities = tuple(float(p) for p in probabilities)
        if len(probabilities) != 3 or any(p <= 0 for p in probabilities) or abs(sum(probabilities) - 1) > 1.e-6:
            raise ValueError("traversal_probabilities 必须为三个正数且总和为 1")
        self.traversal_probabilities = probabilities

    def prepare_next_episode(self, env_ids):
        """Select the next tile/mode before reset events; keep completed tags intact."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if not ids.numel():
            return
        mode_probabilities = torch.tensor(self.mode_probabilities, device=self.device)
        self.next_mode[ids] = torch.multinomial(mode_probabilities, ids.numel(), replacement=True)
        self.next_stage[ids] = self.stage
        self.next_sideward_ramp[ids] = 0
        self.pending_reset[ids] = True
        probabilities = torch.tensor(self.traversal_probabilities, device=self.device)
        self.next_traversal[ids] = torch.multinomial(probabilities, ids.numel(), replacement=True)

    def _resample_command(self, env_ids):
        """Consume prepared resets only; timer expiry cannot change a live episode."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        ids = ids[self.pending_reset[ids]]
        if not ids.numel():
            return
        self.episode_traversal[ids] = self.next_traversal[ids]
        self.episode_standing[ids] = False
        super()._resample_command(ids)
        self.episode_stage[ids] = self.next_stage[ids]
        self.episode_sideward_ramp[ids] = self.next_sideward_ramp[ids]
        self.episode_mode[ids] = self.next_mode[ids]
        self.pending_reset[ids] = False
        self.vel_command_b[ids, 1:] = 0.0
        backward = ids[self.episode_mode[ids] == 2]
        self.vel_command_b[backward, 0] = -torch.empty(backward.numel(), device=self.device).uniform_(
            *self.cfg.backward_speed_range)
        flat = ids[self.episode_traversal[ids] == 2]
        if flat.numel():
            self._resample_flat_command(flat)
        stairs = ids[self.episode_traversal[ids] != 2]
        if stairs.numel():
            # 上楼/下楼回合绝不继承基类的站立掩码，否则指令会被逐步清零。
            self.is_standing_env[stairs] = False
        down = ids[self.episode_traversal[ids] == 1]
        limits = self.next_descent_speed_range[down]
        self.vel_command_b[down, 0] = limits[:, 0] + torch.rand(down.numel(), device=self.device) * (
            limits[:, 1] - limits[:, 0])
        self.vel_command_b[down, 0] *= torch.where(self.episode_mode[down] == 2, -1.0, 1.0)
        self.vel_command_b[down, 1:] = 0.0

    def _resample_flat_command(self, ids):
        """Use the terrain locomotion distribution on flat episodes.

        The command remains fixed for the episode, matching the stair task's
        episode-level curriculum bookkeeping.  Within the flat subset the
        categories are 10% standing, 25% pure yaw, and 65% mixed motion drawn
        from the signed flat_* ranges (the terrain distribution), not the
        stair forward-only ranges.lin_vel_x.
        """
        category = torch.rand(ids.numel(), device=self.device)
        standing_probability = getattr(self.cfg, "rel_flat_standing_envs", 0.10)
        pure_probability = getattr(self.cfg, "rel_pure_rotation_envs", 0.25)
        standing = category < standing_probability
        pure_rotation = torch.logical_and(
            category >= standing_probability,
            category < standing_probability + pure_probability,
        )
        mixed = ~(standing | pure_rotation)

        self.is_standing_env[ids] = standing
        self.episode_standing[ids] = standing
        self.vel_command_b[ids] = 0.0

        pure_ids = ids[pure_rotation]
        if pure_ids.numel():
            low, high = getattr(self.cfg, "pure_rotation_speed_range", (0.2, 1.0))
            magnitude = torch.empty(pure_ids.numel(), device=self.device).uniform_(low, high)
            sign = torch.where(torch.rand(pure_ids.numel(), device=self.device) < 0.5, -1.0, 1.0)
            self.vel_command_b[pure_ids, 2] = sign * magnitude

        mixed_ids = ids[mixed]
        if mixed_ids.numel():
            x_low, x_high = getattr(self.cfg, "flat_lin_vel_x", (-1.0, 1.0))
            y_low, y_high = getattr(self.cfg, "flat_lin_vel_y", (-0.5, 0.5))
            yaw_low, yaw_high = getattr(self.cfg, "flat_ang_vel_z", (-1.0, 1.0))
            self.vel_command_b[mixed_ids, 0] = torch.empty(
                mixed_ids.numel(), device=self.device).uniform_(x_low, x_high)
            self.vel_command_b[mixed_ids, 1] = torch.empty(
                mixed_ids.numel(), device=self.device).uniform_(y_low, y_high)
            self.vel_command_b[mixed_ids, 2] = torch.empty(
                mixed_ids.numel(), device=self.device).uniform_(yaw_low, yaw_high)

        # Keep curriculum metrics meaningful: classify flat commands by their
        # dominant motion direction, while standing remains in the forward slot.
        flat_cmd = self.vel_command_b[ids]
        dominant = flat_cmd.abs().argmax(dim=1)
        mode = torch.zeros(ids.numel(), dtype=torch.long, device=self.device)
        mode[dominant == 1] = 1
        mode[(dominant == 0) & (flat_cmd[:, 0] < 0.0)] = 2
        mode[dominant == 2] = 3
        self.episode_mode[ids] = mode


@configclass
class MixedStairVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = MixedStairVelocityCommand
    initial_stage: int = 0  # Stage 0: forward; stage 1: forward/backward equally.
    initial_sideward_ramp: int = 0  # Reserved; must remain zero.
    backward_speed_range: tuple[float, float] = (0.2, 0.4)
    # Fresh runs start flat-heavy; MixedStairCurriculum promotes to 40/40/20
    # after two qualified flat-quality windows and restores it on resume.
    traversal_probabilities: tuple[float, float, float] = (0.30, 0.30, 0.40)
    descent_initial_speed_range: tuple[float, float] = (0.15, 0.30)
    descent_final_speed_range: tuple[float, float] = (0.30, 0.60)
    # Flat episodes reuse the terrain locomotion command strata and its signed
    # velocity distribution; stairs keep the forward-only ranges.lin_vel_x.
    rel_flat_standing_envs: float = 0.10
    rel_pure_rotation_envs: float = 0.25
    pure_rotation_speed_range: tuple[float, float] = (0.2, 1.0)
    flat_lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    flat_lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    flat_ang_vel_z: tuple[float, float] = (-1.0, 1.0)
