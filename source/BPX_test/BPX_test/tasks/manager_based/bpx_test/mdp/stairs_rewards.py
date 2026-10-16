"""Stair-specific contact costs; simulator forces are not policy observations."""

import torch

from isaaclab.managers import SceneEntityCfg


def _feet_time_excess(
    env, sensor_cfg, command_name, time_field, max_time, ramp_time, command_threshold,
):
    """Sum bounded per-foot costs; positive values require a negative reward weight."""
    if max_time <= 0.0 or ramp_time <= 0.0 or command_threshold < 0.0:
        raise ValueError("时间阈值与渐增时间必须为正，移动指令阈值不能为负")
    sensor = env.scene.sensors[sensor_cfg.name]
    durations = getattr(sensor.data, time_field)
    if durations is None:
        raise ValueError("逐足时间约束要求接触传感器启用 track_air_time=True")
    # The sensor resets each foot's timer at its own contact transition, and
    # resets all selected environments' timers at episode reset.
    excess = ((durations[:, sensor_cfg.body_ids] - max_time) / ramp_time).clamp(0.0, 1.0)
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.vector_norm(command[:, :2], dim=1) > command_threshold
    return excess.sum(dim=1) * moving


def feet_air_time_excess(
    env, sensor_cfg: SceneEntityCfg, command_name: str,
    max_air_time: float = 0.6, ramp_time: float = 0.4, command_threshold: float = 0.1,
):
    """Continuously penalize prolonged unloading during commanded translation.

    Each foot is free for 0.6 s, then its cost increases linearly to 1 at
    1.0 s. A landing is not required to trigger the cost. Stationary and pure
    yaw commands are exempt; actual robot speed does not gate stalled policies.
    """
    return _feet_time_excess(
        env, sensor_cfg, command_name, "current_air_time", max_air_time, ramp_time, command_threshold,
    )


def feet_contact_time_excess(
    env, sensor_cfg: SceneEntityCfg, command_name: str,
    max_contact_time: float = 1.0, ramp_time: float = 0.5, command_threshold: float = 0.1,
):
    """Penalize a foot that remains in continuous stance without taking a step.

    Each foot is free for 1.0 s, then its cost increases linearly to 1 at
    1.5 s. This permits temporary support on stairs without prescribing a gait.
    """
    return _feet_time_excess(
        env, sensor_cfg, command_name, "current_contact_time", max_contact_time, ramp_time, command_threshold,
    )


def feet_stumble(env, sensor_cfg: SceneEntityCfg, force_ratio: float = 4.0, min_horizontal_force: float = 5.0):
    """Return 0/1 when any foot hits a vertical face, ignoring tiny force noise.

    World horizontal force is compared with absolute vertical force, following
    Extreme Parkour. Use a small external negative weight for blind probing.
    """
    if force_ratio <= 0.0 or min_horizontal_force < 0.0:
        raise ValueError("force_ratio 必须为正，min_horizontal_force 不能为负")
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    horizontal = torch.linalg.vector_norm(forces[..., :2], dim=-1)
    kicked = (horizontal > force_ratio * forces[..., 2].abs()) & (horizontal > min_horizontal_force)
    return kicked.any(dim=1).float()
