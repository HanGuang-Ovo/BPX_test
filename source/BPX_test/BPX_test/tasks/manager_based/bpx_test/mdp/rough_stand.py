"""Training-only ground queries, resets and stability curriculum; no extra observations."""
import numpy as np
import torch
from pxr import UsdGeom
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh


def ground_heights(env, xy):
    """Query a 0.8 m square on the actual collision mesh (world coordinates)."""
    if not hasattr(env, '_bpx_stand_ground_mesh'):
        prim = sim_utils.get_first_matching_child_prim(
            env.scene.terrain.cfg.prim_path, lambda p: p.GetTypeName() == 'Mesh')
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get())
        transform = np.array(omni.usd.get_world_transform_matrix(prim)).T
        points = points @ transform[:3, :3].T + transform[:3, 3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        env._bpx_stand_ground_mesh = convert_to_warp_mesh(points, indices, device=env.device)
    axis = torch.linspace(-.4, .4, 9, device=env.device)
    offsets = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), dim=-1).reshape(-1, 2)
    starts = torch.zeros((len(xy), len(offsets), 3), device=env.device)
    starts[..., :2] = xy[:, None, :] + offsets
    starts[..., 2] = 10.
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1
    heights = raycast_mesh(starts, directions, env._bpx_stand_ground_mesh)[0][..., 2]
    if not torch.isfinite(heights).all():
        raise RuntimeError('Rough stand ground query missed terrain')
    return heights


def reset_rough_stand(env, env_ids, flat_probability=.1):
    robot = env.scene['robot']
    n = len(env_ids)
    local = torch.zeros((n, 2), device=env.device)
    rough = torch.rand(n, device=env.device) >= flat_probability
    local[:, 0] = torch.where(torch.rand(n, device=env.device) < .5, -2.5, 2.5)
    local[:, 1].uniform_(-.6, .6)
    local[~rough] = 0
    xy = env.scene.env_origins[env_ids, :2] + local
    heights = ground_heights(env, xy)
    state = robot.data.default_root_state[env_ids].clone()
    state[:, :2] = xy
    state[:, 2] = heights.amax(dim=1) + .42
    zero = torch.zeros(n, device=env.device)
    state[:, 3:7] = quat_from_euler_xyz(zero, zero, torch.empty_like(zero).uniform_(-torch.pi, torch.pi))
    state[:, 7:] = 0
    robot.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
    tracker = env._bpx_stand_stability
    tracker.spawn_xy[env_ids] = xy
    tracker.rough[env_ids] = rough


def relative_stand_height_l2(env, target_height):
    pos = env.scene['robot'].data.root_pos_w
    return (pos[:, 2] - ground_heights(env, pos[:, :2]).mean(dim=1) - target_height).square()


class StandStability(ManagerTermBase):
    """Accumulate stable time after settling; excessive drift is a physical failure."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env._bpx_stand_stability = self
        self.spawn_xy = torch.zeros((env.num_envs, 2), device=env.device)
        self.rough = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for name in ('duration', 'stable_time', 'max_drift'):
            setattr(self, name, torch.zeros(env.num_envs, device=env.device))

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in ('duration', 'stable_time', 'max_drift'):
            getattr(self, name)[ids] = 0

    def __call__(self, env, settling_time, max_speed, max_yaw_rate, max_tilt, max_drift, failure_drift):
        data = env.scene['robot'].data
        drift = torch.linalg.vector_norm(data.root_pos_w[:, :2] - self.spawn_xy, dim=1)
        active = env.episode_length_buf * env.step_dt > settling_time
        tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1., 1.))
        stable = ((torch.linalg.vector_norm(data.root_lin_vel_w[:, :2], dim=1) <= max_speed)
                  & (data.root_ang_vel_b[:, 2].abs() <= max_yaw_rate)
                  & (tilt <= max_tilt) & (drift <= max_drift))
        self.duration += active.float() * env.step_dt
        self.stable_time += (active & stable).float() * env.step_dt
        self.max_drift = torch.maximum(self.max_drift, drift)
        return drift > failure_drift


class StandingCurriculum(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env, env_ids, min_duration, stable_fraction, successes_to_promote):
        ids = torch.arange(env.num_envs, device=env.device)
        if env_ids is not None:
            ids = ids[env_ids]
        ids = ids[env.episode_length_buf[ids] > 0]
        terrain, tracker = env.scene.terrain, env._bpx_stand_stability
        result = {'level': terrain.terrain_levels.float().mean(), 'success_rate': 0., 'stable_fraction': 0., 'max_drift': 0.}
        if not len(ids):
            return result
        fraction = tracker.stable_time[ids] / tracker.duration[ids].clamp(min=env.step_dt)
        failed = env.termination_manager.terminated[ids]
        complete = tracker.duration[ids] >= min_duration
        success = complete & (fraction >= stable_fraction) & ~failed
        rough = tracker.rough[ids]
        self.streak[ids] = torch.where(rough, torch.where(success, self.streak[ids] + 1, 0), self.streak[ids])
        promote = rough & (self.streak[ids] >= successes_to_promote)
        down = rough & (failed | (complete & ~success))
        terrain.update_env_origins(ids, promote & (terrain.terrain_levels[ids] < terrain.max_terrain_level - 1), down)
        self.streak[ids[promote | down]] = 0
        return {'level': terrain.terrain_levels.float().mean(), 'success_rate': success.float().mean(),
                'stable_fraction': fraction.mean(), 'max_drift': tracker.max_drift[ids].mean()}
