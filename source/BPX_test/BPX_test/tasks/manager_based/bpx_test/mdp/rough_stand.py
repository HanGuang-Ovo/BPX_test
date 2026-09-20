"""Training-only ground queries, stratified resets and stability curriculum."""
import numpy as np
import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_from_euler_xyz

from ..rough_terrain_geometry import wave_height_field
from .rewards import left_right_joint_symmetry_error, whole_body_com_support_error


def _ground_height_fields(env):
    """Recreate the generated wave grids without building a second GPU mesh BVH."""
    if hasattr(env, '_bpx_stand_height_fields'):
        return env._bpx_stand_height_fields

    terrain = env.scene.terrain
    generator_cfg = terrain.cfg.terrain_generator
    if generator_cfg is None or len(generator_cfg.sub_terrains) != 1:
        raise RuntimeError('Rough Stand height lookup requires one generated wave terrain type')
    if generator_cfg.seed is None:
        raise RuntimeError('Rough Stand height lookup requires a deterministic terrain seed')

    wave_cfg = next(iter(generator_cfg.sub_terrains.values())).copy()
    wave_cfg.size = generator_cfg.size
    wave_cfg.seed = generator_cfg.seed
    rows, cols = generator_cfg.num_rows, generator_cfg.num_cols
    lower, upper = generator_cfg.difficulty_range
    rng = np.random.default_rng(generator_cfg.seed)
    fields = [[None for _ in range(cols)] for _ in range(rows)]
    # Match TerrainGenerator._generate_curriculum_terrains: columns outermost,
    # then rows, with exactly one generator RNG draw per tile.
    for col in range(cols):
        for row in range(rows):
            difficulty = lower + (upper - lower) * (row + rng.uniform()) / rows
            fields[row][col] = wave_height_field(difficulty, wave_cfg).astype(np.float32)
    height_fields = torch.from_numpy(np.stack([np.stack(row, axis=0) for row in fields], axis=0)).to(env.device)
    lower_xy = terrain.terrain_origins[0, 0, :2] - height_fields.new_tensor(generator_cfg.size) * .5
    grid_step = height_fields.new_tensor(generator_cfg.size) / height_fields.new_tensor(
        (height_fields.shape[2] - 1, height_fields.shape[3] - 1)
    )
    env._bpx_stand_height_fields = height_fields, lower_xy, grid_step
    return env._bpx_stand_height_fields


def _interpolate_ground_heights(env, xy):
    """Interpolate the same two triangles used to construct each wave mesh cell."""
    fields, lower_xy, grid_step = _ground_height_fields(env)
    rows, cols, nx, ny = fields.shape
    size = grid_step * fields.new_tensor((nx - 1, ny - 1))
    terrain_xy = xy - lower_xy
    row = torch.floor(terrain_xy[:, 0] / size[0]).long().clamp(0, rows - 1)
    col = torch.floor(terrain_xy[:, 1] / size[1]).long().clamp(0, cols - 1)
    tile_xy = terrain_xy - torch.stack((row * size[0], col * size[1]), dim=1)
    grid_xy = tile_xy / grid_step
    i = torch.floor(grid_xy[:, 0]).long().clamp(0, nx - 2)
    j = torch.floor(grid_xy[:, 1]).long().clamp(0, ny - 2)
    u = (grid_xy[:, 0] - i).clamp(0., 1.)
    v = (grid_xy[:, 1] - j).clamp(0., 1.)

    z00 = fields[row, col, i, j]
    z10 = fields[row, col, i + 1, j]
    z01 = fields[row, col, i, j + 1]
    z11 = fields[row, col, i + 1, j + 1]
    lower_triangle = z00 + u * (z10 - z00) + v * (z11 - z10)
    upper_triangle = z00 + u * (z11 - z01) + v * (z01 - z00)
    return torch.where(v <= u, lower_triangle, upper_triangle)


def ground_heights(env, xy):
    """Query a 0.8 m square on the generated collision height field."""
    axis = torch.linspace(-.4, .4, 9, device=env.device)
    offsets = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), dim=-1).reshape(-1, 2)
    sample_xy = xy[:, None, :] + offsets
    return _interpolate_ground_heights(env, sample_xy.reshape(-1, 2)).reshape(len(xy), -1)


def _uniform(count, value_range, device):
    """Sample a scalar range without relying on CPU-side random state."""
    low, high = value_range
    return torch.empty(count, device=device).uniform_(low, high)


def _sample_root_state(state, mask, pose_range, velocity_range):
    """Apply root pose/velocity perturbations to the selected rows in-place."""
    count = int(mask.sum())
    if count == 0:
        return
    device = state.device
    roll = _uniform(count, pose_range['roll'], device)
    pitch = _uniform(count, pose_range['pitch'], device)
    yaw = _uniform(count, pose_range['yaw'], device)
    state[mask, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
    for offset, name in enumerate(('x', 'y', 'z', 'roll', 'pitch', 'yaw')):
        state[mask, 7 + offset] = _uniform(count, velocity_range[name], device)


def reset_rough_stand(
    env,
    env_ids,
    flat_probability=.25,
    replay_probability=.25,
    handoff_probability=.25,
    pose_range=None,
    velocity_range=None,
    handoff_pose_range=None,
    handoff_velocity_range=None,
):
    """Reset from flat, adaptive and historical terrain strata.

    ``flat`` preserves level-ground behavior, ``adaptive`` evaluates the current
    mastery level, and ``replay`` revisits a uniformly sampled lower level.  Only
    adaptive rough episodes are allowed to change curriculum mastery.

    A configurable subset uses the wider ``handoff`` distribution to approximate
    the body state seen after Init or locomotion hands control to Stand.  Joint
    state and the first observation's previous action are initialized by
    :func:`reset_rough_stand_joints`.
    """
    if min(flat_probability, replay_probability, handoff_probability) < 0.0:
        raise ValueError('reset probabilities must be non-negative')
    if flat_probability + replay_probability > 1.0:
        raise ValueError('flat_probability + replay_probability must not exceed 1')
    if handoff_probability > 1.0:
        raise ValueError('handoff_probability must not exceed 1')
    pose_range = pose_range or {
        'roll': (-.05, .05), 'pitch': (-.05, .05), 'yaw': (-torch.pi, torch.pi),
    }
    velocity_range = velocity_range or {
        'x': (-.05, .05), 'y': (-.05, .05), 'z': (-.03, .03),
        'roll': (-.08, .08), 'pitch': (-.08, .08), 'yaw': (-.08, .08),
    }
    handoff_pose_range = handoff_pose_range or {
        'roll': (-.10, .10), 'pitch': (-.10, .10), 'yaw': (-torch.pi, torch.pi),
    }
    handoff_velocity_range = handoff_velocity_range or {
        'x': (-.08, .08), 'y': (-.08, .08), 'z': (-.05, .05),
        'roll': (-.15, .15), 'pitch': (-.15, .15), 'yaw': (-.15, .15),
    }

    robot = env.scene['robot']
    terrain = env.scene.terrain
    tracker = env._bpx_stand_stability
    n = len(env_ids)
    category = torch.rand(n, device=env.device)
    flat = category < flat_probability
    replay = (category >= flat_probability) & (category < flat_probability + replay_probability)
    adaptive = ~(flat | replay)
    rough = ~flat
    handoff = torch.rand(n, device=env.device) < handoff_probability

    curriculum = getattr(env, '_bpx_stand_curriculum', None)
    mastery = (curriculum.mastery_level[env_ids] if curriculum is not None
               else terrain.terrain_levels[env_ids]).clone()
    sampled_level = mastery.clone()
    # Historical replay samples strictly lower levels whenever one exists.
    lower_count = mastery.clamp(min=1)
    replay_level = torch.floor(torch.rand(n, device=env.device) * lower_count).long()
    sampled_level[replay] = replay_level[replay]
    terrain.terrain_levels[env_ids] = sampled_level
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        sampled_level, terrain.terrain_types[env_ids]
    ]

    local = torch.zeros((n, 2), device=env.device)
    local[:, 0] = torch.where(torch.rand(n, device=env.device) < .5, -2.5, 2.5)
    local[:, 1].uniform_(-.6, .6)
    local[flat] = 0
    xy = env.scene.env_origins[env_ids, :2] + local
    heights = ground_heights(env, xy)
    state = robot.data.default_root_state[env_ids].clone()
    state[:, :2] = xy
    state[:, 2] = heights.amax(dim=1) + .42
    _sample_root_state(state, ~handoff, pose_range, velocity_range)
    _sample_root_state(state, handoff, handoff_pose_range, handoff_velocity_range)
    robot.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
    tracker.spawn_xy[env_ids] = xy
    tracker.rough[env_ids] = rough
    tracker.adaptive[env_ids] = adaptive
    tracker.handoff[env_ids] = handoff
    tracker.sampled_level[env_ids] = sampled_level


def reset_rough_stand_joints(
    env,
    env_ids,
    position_range=(-.05, .05),
    velocity_range=(-.05, .05),
    handoff_position_range=(-.15, .15),
    handoff_velocity_range=(-.30, .30),
    action_scale=.5,
):
    """Reset joints and synthesize the action preceding a Stand handoff."""
    if action_scale <= 0.0:
        raise ValueError('action_scale must be positive')
    robot = env.scene['robot']
    tracker = env._bpx_stand_stability
    handoff = tracker.handoff[env_ids]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    regular_count = int((~handoff).sum())
    handoff_count = int(handoff.sum())
    if regular_count:
        joint_pos[~handoff] += torch.empty_like(joint_pos[~handoff]).uniform_(*position_range)
        joint_vel[~handoff] += torch.empty_like(joint_vel[~handoff]).uniform_(*velocity_range)
    if handoff_count:
        joint_pos[handoff] += torch.empty_like(joint_pos[handoff]).uniform_(*handoff_position_range)
        joint_vel[handoff] += torch.empty_like(joint_vel[handoff]).uniform_(*handoff_velocity_range)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    tracker.handoff_action[env_ids] = 0.0
    tracker.handoff_action[env_ids[handoff]] = torch.clamp(
        (joint_pos[handoff] - robot.data.default_joint_pos[env_ids[handoff]]) / action_scale,
        min=-1.0,
        max=1.0,
    )


def rough_stand_last_action(env):
    """Expose a synthetic previous action on the first handoff observation only."""
    action = env.action_manager.action
    tracker = getattr(env, '_bpx_stand_stability', None)
    if tracker is None:
        return action
    handoff = (env.episode_length_buf == 0) & tracker.handoff
    if not handoff.any():
        return action
    action = action.clone()
    action[handoff] = tracker.handoff_action[handoff]
    return action


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
        self.adaptive = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.handoff = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.sampled_level = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.handoff_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        for name in (
            'duration', 'stable_time', 'four_feet_time', 'symmetry_time',
            'symmetry_error_sum', 'support_center_time',
            'support_center_error_sum', 'max_drift',
        ):
            setattr(self, name, torch.zeros(env.num_envs, device=env.device))

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in (
            'duration', 'stable_time', 'four_feet_time', 'symmetry_time',
            'symmetry_error_sum', 'support_center_time',
            'support_center_error_sum', 'max_drift',
        ):
            getattr(self, name)[ids] = 0

    def __call__(
        self, env, settling_time, max_speed, max_yaw_rate, max_tilt, max_drift,
        failure_drift, contact_force_threshold, sensor_cfg,
        symmetry_tolerance, symmetry_joint_cfg,
        support_center_tolerance, support_feet_cfg,
    ):
        data = env.scene['robot'].data
        drift = torch.linalg.vector_norm(data.root_pos_w[:, :2] - self.spawn_xy, dim=1)
        active = env.episode_length_buf * env.step_dt > settling_time
        tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1., 1.))
        contact_sensor = env.scene.sensors[sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        all_feet_contact = (forces.norm(dim=-1).amax(dim=1) > contact_force_threshold).all(dim=1)
        symmetry_error = left_right_joint_symmetry_error(env, symmetry_joint_cfg)
        symmetric = symmetry_error <= symmetry_tolerance
        support_center_error = whole_body_com_support_error(
            env, support_feet_cfg, support_feet_cfg,
        )
        support_centered = all_feet_contact & (support_center_error <= support_center_tolerance)
        stable = ((torch.linalg.vector_norm(data.root_lin_vel_w[:, :2], dim=1) <= max_speed)
                  & (data.root_ang_vel_b[:, 2].abs() <= max_yaw_rate)
                  & (tilt <= max_tilt) & (drift <= max_drift))
        self.duration += active.float() * env.step_dt
        self.stable_time += (active & stable).float() * env.step_dt
        self.four_feet_time += (active & all_feet_contact).float() * env.step_dt
        self.symmetry_time += (active & symmetric).float() * env.step_dt
        self.symmetry_error_sum += active.float() * symmetry_error * env.step_dt
        self.support_center_time += (active & support_centered).float() * env.step_dt
        self.support_center_error_sum += active.float() * support_center_error * env.step_dt
        self.max_drift = torch.maximum(self.max_drift, drift)
        return drift > failure_drift


class StandingCurriculum(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env._bpx_stand_curriculum = self
        self.streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.mastery_level = env.scene.terrain.terrain_levels.clone()

    def __call__(
        self, env, env_ids, min_duration, stable_fraction,
        minimum_four_feet_fraction, minimum_symmetry_fraction,
        minimum_support_center_fraction, successes_to_promote,
    ):
        ids = torch.arange(env.num_envs, device=env.device)
        if env_ids is not None:
            ids = ids[env_ids]
        ids = ids[env.episode_length_buf[ids] > 0]
        terrain, tracker = env.scene.terrain, env._bpx_stand_stability
        result = {
            'level': self.mastery_level.float().mean(), 'sampled_level': 0.,
            'success_rate': 0., 'stable_fraction': 0., 'four_feet_fraction': 0.,
            'symmetry_fraction': 0., 'symmetry_error': 0.,
            'support_center_fraction': 0., 'support_center_error': 0.,
            'max_drift': 0.,
        }
        if not len(ids):
            return result
        fraction = tracker.stable_time[ids] / tracker.duration[ids].clamp(min=env.step_dt)
        four_feet_fraction = tracker.four_feet_time[ids] / tracker.duration[ids].clamp(min=env.step_dt)
        symmetry_fraction = tracker.symmetry_time[ids] / tracker.duration[ids].clamp(min=env.step_dt)
        symmetry_error = tracker.symmetry_error_sum[ids] / tracker.duration[ids].clamp(min=env.step_dt)
        support_center_fraction = (
            tracker.support_center_time[ids] / tracker.duration[ids].clamp(min=env.step_dt)
        )
        support_center_error = (
            tracker.support_center_error_sum[ids] / tracker.duration[ids].clamp(min=env.step_dt)
        )
        failed = env.termination_manager.terminated[ids]
        complete = tracker.duration[ids] >= min_duration
        success = (complete & (fraction >= stable_fraction)
                   & (four_feet_fraction >= minimum_four_feet_fraction)
                   & (symmetry_fraction >= minimum_symmetry_fraction)
                   & (support_center_fraction >= minimum_support_center_fraction)
                   & ~failed)
        adaptive = tracker.adaptive[ids]
        self.streak[ids] = torch.where(
            adaptive, torch.where(success, self.streak[ids] + 1, 0), self.streak[ids]
        )
        promote = adaptive & (self.streak[ids] >= successes_to_promote)
        down = adaptive & (failed | (complete & ~success))
        self.mastery_level[ids] += promote.long() - down.long()
        # Advanced indexing returns a copy, so assign the clamped values back.
        self.mastery_level[ids] = self.mastery_level[ids].clamp(
            0, terrain.max_terrain_level - 1
        )
        self.streak[ids[promote | down]] = 0
        return {
            'level': self.mastery_level.float().mean(),
            'sampled_level': tracker.sampled_level[ids].float().mean(),
            'success_rate': success.float().mean(),
            'stable_fraction': fraction.mean(),
            'four_feet_fraction': four_feet_fraction.mean(),
            'symmetry_fraction': symmetry_fraction.mean(),
            'symmetry_error': symmetry_error.mean(),
            'support_center_fraction': support_center_fraction.mean(),
            'support_center_error': support_center_error.mean(),
            'max_drift': tracker.max_drift[ids].mean(),
        }
