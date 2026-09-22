"""Traversability curriculum; uses simulator state only, never policy observations."""
import torch
from isaaclab.managers import ManagerTermBase


class RoughTraversalBoundary(ManagerTermBase):
    """Accumulate rough-ground traversal and truncate before entering a neighbour tile.

    TerminationManager calls this once per policy step. CurriculumManager reads the
    completed episode before events/manager resets clear these accumulators.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env._bpx_rough_traversal = self
        self.previous_xy = torch.zeros((env.num_envs, 2), device=env.device)
        self.previous_rough = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for name in ("rough_distance", "rough_time", "linear_error", "angular_error", "duration"):
            setattr(self, name, torch.zeros(env.num_envs, device=env.device))

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        for name in ("rough_distance", "rough_time", "linear_error", "angular_error", "duration"):
            getattr(self, name)[env_ids] = 0
        self.previous_rough[env_ids] = False

    def __call__(self, env, rough_start: float, boundary: float):
        robot = env.scene["robot"]
        xy = robot.data.root_pos_w[:, :2]
        local = xy - env.scene.env_origins[:, :2]
        radius = local.abs().amax(dim=1)
        rough = (radius > rough_start) & (radius < boundary)
        distance = torch.linalg.vector_norm(xy - self.previous_xy, dim=1)
        # Exclude reset teleports and reject implausible per-step displacement spikes.
        valid = rough & self.previous_rough & (env.episode_length_buf > 1)
        self.rough_distance += torch.where(valid, distance.clamp(max=2.0 * env.step_dt), 0.0)
        self.rough_time += rough.float() * env.step_dt
        command = env.command_manager.get_command("base_velocity")
        self.linear_error += torch.linalg.vector_norm(command[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1) * env.step_dt
        self.angular_error += (command[:, 2] - robot.data.root_ang_vel_b[:, 2]).abs() * env.step_dt
        self.duration += env.step_dt
        self.previous_xy[:] = xy
        self.previous_rough[:] = rough
        return radius >= boundary


class TraversabilityCurriculum(ManagerTermBase):
    """Two consecutive traversable episodes promote; consecutive failures demote.

    A low-speed/turning episode that never reaches rough ground cannot promote,
    but is not demoted solely for low displacement. Demotion needs failures_to_demote
    consecutive bad episodes (physical failure, or enough duration with poor
    tracking), so a single unlucky fall no longer bounces the env down a level;
    any non-bad episode clears the failure streak. Highest-level success stays
    at the highest level (unlike the standard random wrap-around).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.success_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.failure_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env, env_ids, min_rough_distance: float, min_rough_time: float,
                 max_linear_error: float, max_angular_error: float, successes_to_promote: int,
                 failures_to_demote: int = 1):
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        ids = ids[env.episode_length_buf[ids] > 0]
        terrain = env.scene.terrain
        tracker = env._bpx_rough_traversal
        if ids.numel() == 0:
            return {"level": terrain.terrain_levels.float().mean(), "success_rate": 0.0,
                    "rough_distance": 0.0, "rough_time": 0.0}
        duration = tracker.duration[ids].clamp(min=env.step_dt)
        tracking_ok = ((tracker.linear_error[ids] / duration <= max_linear_error)
                       & (tracker.angular_error[ids] / duration <= max_angular_error))
        failed = env.termination_manager.terminated[ids]
        exposed = ((tracker.rough_time[ids] >= min_rough_time)
                   & (tracker.rough_distance[ids] >= min_rough_distance))
        success = exposed & tracking_ok & ~failed
        # A bad episode is a physical failure, or enough duration with poor tracking.
        bad_episode = failed | ((duration >= min_rough_time) & ~tracking_ok)
        self.success_streak[ids] = torch.where(success, self.success_streak[ids] + 1, 0)
        self.failure_streak[ids] = torch.where(bad_episode, self.failure_streak[ids] + 1, 0)
        promote = self.success_streak[ids] >= successes_to_promote
        demote = self.failure_streak[ids] >= failures_to_demote
        # Clamp the top level before TerrainImporter can randomly wrap it around.
        move_up = promote & (terrain.terrain_levels[ids] < terrain.max_terrain_level - 1)
        terrain.update_env_origins(ids, move_up, demote)
        self.success_streak[ids[promote | demote]] = 0
        self.failure_streak[ids[promote | demote]] = 0
        return {"level": terrain.terrain_levels.float().mean(),
                "success_rate": success.float().mean(),
                "rough_distance": tracker.rough_distance[ids].mean(),
                "rough_time": tracker.rough_time[ids].mean()}
