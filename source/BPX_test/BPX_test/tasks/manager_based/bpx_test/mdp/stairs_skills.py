"""Independent Move and Hold curricula; Hold receives real locomotion states.

Hold levels: 0 static flat, 1 flat braking, 2 low stair tread, 3 low stair
edge, 4..10 stair edges on terrain rows 1..7. Warmup is controlled by a frozen
Move actor; only phase 2 actions belong to the trainable Hold policy.
"""

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass

from . import stairs


class StairMoveCurriculum(stairs.StairMasteryCurriculum):
    def __call__(self, env, env_ids, initial_level=0, successes_to_promote=3,
                 failures_to_demote=2, max_linear_error=.20, max_angular_error=.30, enabled=True):
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        c = env._bpx_stairs_command
        finished = ids[(env.episode_length_buf[ids] > 0) & (c.route[ids] != 0)]
        duration = c.duration.clamp(min=env.step_dt)
        passed = (c.completed & ~env.termination_manager.terminated
                  & (c.linear_error / duration < max_linear_error)
                  & (c.angular_error / duration < max_angular_error))
        active = finished[self.adaptive[finished]]
        self._advance(active, passed[active], self.mastery, self.successes, self.failures,
                      successes_to_promote, failures_to_demote,
                      env.scene.terrain.cfg.terrain_generator.num_rows - 1, enabled)
        c.task_success[:] = passed
        c.metrics["task_success"][:] = passed.float()
        mean = lambda x: x.float().mean() if x.numel() else torch.tensor(0., device=env.device)
        return {"level": self.mastery.float().mean(), "success_rate": mean(passed[finished]),
                "sample_count": float(len(active)),
                "ascent_success": mean(passed[finished[c.route[finished] == 1]]),
                "descent_success": mean(passed[finished[c.route[finished] == 2]])}


class StairHoldCommand(stairs.StairVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hold_level = torch.zeros_like(self.phase)
        self.next_hold_level = torch.zeros_like(self.phase)
        self.hold_started = torch.zeros_like(self.completed)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.hold_level[ids] = self.next_hold_level[ids]
        self.stop_required[ids] = True  # Flat braking is a real Hold sample too.
        self.hold_started[ids] = self.hold_level[ids] == 0
        flat = self.route[ids] == 0
        local_x = self.robot.data.root_pos_w[ids, 0] - self._env.scene.env_origins[ids, 0]
        flat_ids = ids[flat]
        self.stop_x[flat_ids] = local_x[flat] + stairs._uniform(
            len(flat_ids), self.cfg.flat_approach_range, self.device)
        static = ids[self.hold_level[ids] == 0]
        self.phase[static] = 2
        self.hold_time[static] = self.hold_duration[static]
        self.was_stopped[static] = True

    def _update_command(self):
        holding = self.phase == 2
        elapsed = self.hold_elapsed.clone()
        remaining = self.hold_time.clone()
        finished = self.stop_done.clone()
        # Reuse the exact command slew and spatial braking used by Move. Hold
        # episodes end on the timer; they never restart or travel to the exit.
        super()._update_command()
        self.phase[holding] = 2
        self.vel_command_b[holding] = 0.
        self.target[holding] = 0.
        # Only the physical-step monitor advances Hold time. Command.compute()
        # also runs immediately after an auto-reset, before any new Hold action.
        self.hold_elapsed[holding] = elapsed[holding]
        self.hold_time[holding] = remaining[holding]
        self.stop_done[holding] = finished[holding]
        self.hold_started |= self.phase == 2


@configclass
class StairHoldCommandCfg(stairs.StairVelocityCommandCfg):
    class_type: type = StairHoldCommand
    stop_probability: float = 1.
    start_wait_range: tuple[float, float] = (0., .2)
    hold_time_range: tuple[float, float] = (2., 4.)
    flat_standing_probability: float = 0.
    flat_x_range: tuple[float, float] = (.10, .40)
    flat_y_range: tuple[float, float] = (0., 0.)
    flat_yaw_range: tuple[float, float] = (0., 0.)
    flat_approach_range: tuple[float, float] = (.60, 1.20)


class StairHoldCurriculum(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        maximum = env.scene.terrain.cfg.terrain_generator.num_rows + 2
        initial = cfg.params.get("initial_level", 0)
        if not 0 <= initial <= maximum:
            raise ValueError(f"Hold initial_level must be in [0, {maximum}]")
        self.level = torch.full((env.num_envs,), initial, dtype=torch.long, device=env.device)
        self.successes = torch.zeros_like(self.level)
        self.failures = torch.zeros_like(self.level)
        self.adaptive = torch.zeros_like(self.level, dtype=torch.bool)
        # Geometry/reset adapters used by the shared root reset, not two coupled
        # learning curricula. Only self.level is promoted by Hold results.
        self.mastery = torch.zeros_like(self.level)
        self.stop_mastery = torch.zeros_like(self.level)
        self.stop_adaptive = torch.zeros_like(self.adaptive)
        env._bpx_stair_curriculum = self

    def __call__(self, env, env_ids, initial_level=0, successes_to_promote=3,
                 failures_to_demote=2, enabled=True):
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        c = env._bpx_stairs_command
        finished = ids[env.episode_length_buf[ids] > 0]
        reached = finished[c.hold_started[finished]]
        active = reached[self.adaptive[reached]]
        passed = stairs.stop_success(c) & ~env.termination_manager.terminated
        stairs.StairMasteryCurriculum._advance(
            active, passed[active], self.level, self.successes, self.failures,
            successes_to_promote, failures_to_demote,
            env.scene.terrain.cfg.terrain_generator.num_rows + 2, enabled)
        mean = lambda x: x.float().mean() if x.numel() else torch.tensor(0., device=env.device)
        return {"level": self.level.float().mean(), "success_rate": mean(passed[reached]),
                "sample_count": float(len(reached)), "success_count": passed[reached].sum().float(),
                "warmup_failure_count": float(len(finished) - len(reached))}


def reset_stair_hold_root(env, env_ids, replay_probability=.20, approach_range=(.60, 1.50), height=.40):
    """Spawn on genuine flat approaches; all stair support states come from walking."""
    if not 0 <= replay_probability <= 1:
        raise ValueError("replay_probability must be in [0, 1]")
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    curriculum = env._bpx_stair_curriculum
    level = curriculum.level[ids].clone()
    replay = (level > 0) & (torch.rand(len(ids), device=env.device) < replay_probability)
    lower = (torch.rand(len(ids), device=env.device) * level.clamp(min=1)).long()
    level[replay] = lower[replay]
    curriculum.mastery[ids] = (level - 3).clamp(min=0)
    curriculum.stop_mastery[ids] = (level - 1).clamp(min=0)
    for flat in (True, False):
        group = ids[(level < 2) == flat]
        if len(group):
            stairs.reset_stair_root(env, group, float(flat), 0., approach_range, height)
    curriculum.adaptive[ids] = ~replay
    env._bpx_stairs_command.next_hold_level[ids] = level


class StairHoldEpisodeMonitor(ManagerTermBase):
    def __call__(self, env, feet_cfg, sensor_cfg, settle_time=.30):
        c = env._bpx_stairs_command
        data = env.scene["robot"].data
        c.duration += env.step_dt
        foot = stairs.update_foot_contacts(env, feet_cfg, sensor_cfg)
        holding = c.phase == 2
        c.hold_elapsed += holding.float() * env.step_dt
        c.hold_time[holding] = (c.hold_duration[holding] - c.hold_elapsed[holding]).clamp(min=0.)
        hold = holding & (c.hold_elapsed > settle_time)
        conditions = {
            "linear": torch.linalg.vector_norm(data.root_lin_vel_w, dim=1) < .08,
            "angular": torch.linalg.vector_norm(data.root_ang_vel_b, dim=1) < .20,
            "support": foot.raw.sum(dim=1) >= 3,
            "drift": torch.linalg.vector_norm(data.root_pos_w[:, :2] - c.anchor, dim=1) < .10,
            "upright": data.projected_gravity_b[:, 2] < -.85,
        }
        joint = torch.stack(tuple(conditions.values()), dim=1).all(dim=1)
        c.hold_observed += hold.float() * env.step_dt
        c.hold_good += (hold & joint).float() * env.step_dt
        for name, good in conditions.items():
            getattr(c, f"hold_{name}_good").add_((hold & good).float() * env.step_dt)
        c.stop_done[:] = (c.phase == 2) & (c.hold_elapsed >= c.hold_duration)
        c.completed[:] = c.stop_done
        c._update_metrics()
        c.metrics["task_success"][:] = stairs.stop_success(c).float()
        return c.stop_done


def hold_only(env, function, kwargs=None):
    """Gate every Hold reward on the controller that executed this physical step."""
    return function(env, **(kwargs or {})) * env._bpx_hold_action_mask


class HoldReward(ManagerTermBase):
    """Resolve nested entity configs before delegating to a gated reward.

    Isaac Lab 2.2 resolves only top-level SceneEntityCfg params. This wrapper's
    reward-specific kwargs are nested, so resolve those explicitly at startup.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        for value in cfg.params["kwargs"].values():
            if isinstance(value, SceneEntityCfg):
                value.resolve(env.scene)

    def __call__(self, env, function, kwargs):
        return hold_only(env, function, kwargs)


def hold_success_reward(env):
    c = env._bpx_stairs_command
    eligible = stairs.stop_success(c) & ~env.termination_manager.terminated
    event = eligible & ~c.stop_rewarded
    c.stop_rewarded |= eligible
    return event.float() / env.step_dt


def hold_component_quality(env, sensor_cfg, linear_scale=.12, angular_scale=.30):
    """Additive dense feedback remains useful when the strict conjunction fails."""
    data = env.scene["robot"].data
    linear = torch.exp(-data.root_lin_vel_w.square().sum(dim=1) / linear_scale**2)
    angular = torch.exp(-data.root_ang_vel_b.square().sum(dim=1) / angular_scale**2)
    support = (stairs.contacts(env, sensor_cfg).sum(dim=1).float() / 3.).clamp(max=1.)
    return (linear + angular + support) / 3.


def settled_joint_motion(env, settle_time=.30):
    return env.scene["robot"].data.joint_vel.square().sum(dim=1) * (
        env._bpx_stairs_command.hold_elapsed >= settle_time)
