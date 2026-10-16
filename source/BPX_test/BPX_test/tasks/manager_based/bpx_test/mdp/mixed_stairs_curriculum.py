"""Supported stair landings and episode-level flat locomotion quality."""

import warnings

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .rough_curriculum import RoughTraversalBoundary


class MixedStairTraversalBoundary(RoughTraversalBoundary):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.velocity_integral = torch.zeros((env.num_envs, 2), device=env.device)
        self.speed_integral = torch.zeros(env.num_envs, device=env.device)
        self.along_command_integral = torch.zeros(env.num_envs, device=env.device)
        self.command_speed_integral = torch.zeros(env.num_envs, device=env.device)
        self.landing_time = torch.zeros(env.num_envs, device=env.device)
        self.settled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.along_yaw_integral = torch.zeros(env.num_envs, device=env.device)
        self.command_yaw_integral = torch.zeros(env.num_envs, device=env.device)
        for name in ("flat_quality_time", "flat_stable_time", "flat_unstable_time", "flat_max_unstable_time"):
            setattr(self, name, torch.zeros(env.num_envs, device=env.device))

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for name in ("velocity_integral", "speed_integral", "along_command_integral", "command_speed_integral"):
            getattr(self, name)[ids] = 0
        self.landing_time[ids] = 0
        self.settled[ids] = False
        self.along_yaw_integral[ids] = 0
        self.command_yaw_integral[ids] = 0
        for name in ("flat_quality_time", "flat_stable_time", "flat_unstable_time", "flat_max_unstable_time"):
            getattr(self, name)[ids] = 0

    def __call__(self, env, rough_start: float, boundary: float, completion_radius: float,
                 landing_duration: float, height_tolerance: float, max_tilt: float,
                 max_vertical_speed: float, max_angular_speed: float,
                 contact_force: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
                 flat_completion_duration: float = 1.0, flat_quality_warmup: float = 1.0):
        if flat_quality_warmup < 0:
            raise ValueError("flat_quality_warmup 必须非负")
        outside = super().__call__(env, rough_start, boundary)
        command = env.command_manager.get_term("base_velocity")
        flat = command.episode_traversal == 2
        robot = env.scene["robot"].data
        velocity = robot.root_lin_vel_b[:, :2]
        command_vector = env.command_manager.get_command("base_velocity")
        speed = command_vector[:, :2].norm(dim=1)
        direction = command_vector[:, :2] / speed.clamp(min=1.0e-6).unsqueeze(1)
        self.velocity_integral += velocity * env.step_dt
        self.speed_integral += velocity.norm(dim=1) * env.step_dt
        self.along_command_integral += (velocity * direction).sum(dim=1) * env.step_dt
        self.command_speed_integral += speed * env.step_dt
        yaw_command = command_vector[:, 2]
        self.along_yaw_integral += robot.root_ang_vel_b[:, 2] * yaw_command.sign() * env.step_dt
        self.command_yaw_integral += yaw_command.abs() * env.step_dt
        feet = robot.body_pos_w[:, asset_cfg.body_ids]
        origin = env.scene.env_origins
        radius = (robot.root_pos_w[:, :2] - origin[:, :2]).abs().amax(dim=1)
        foot_radius = (feet[:, :, :2] - origin[:, None, :2]).abs().amax(dim=2)
        forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
        supported = (forces[:, :, 2] > contact_force).sum(dim=1) >= 2
        # All generated tiles have an outer platform at world z=0. Origins are
        # on the central spawn surface: negative for ascent, positive for descent.
        clearance = robot.root_pos_w[:, 2]
        nominal = robot.default_root_state[:, 2]
        upright = robot.projected_gravity_b[:, 2] < -torch.cos(torch.tensor(max_tilt, device=env.device))
        on_platform = ((radius >= completion_radius) & (radius < boundary)
                       & torch.all((foot_radius >= 3.05) & (foot_radius < boundary), dim=1))
        common_stable = (supported & upright
                         & ((clearance - nominal).abs() <= height_tolerance)
                         & (robot.root_lin_vel_w[:, 2].abs() <= max_vertical_speed)
                         & (robot.root_ang_vel_b[:, :2].norm(dim=1) <= max_angular_speed))
        stair_stable = on_platform & common_stable
        # Flat episodes are validated in place. They must not walk to the
        # stair landing radius, which would make standing and pure-yaw samples
        # impossible to complete.
        flat_stable = common_stable
        stable = torch.where(flat, flat_stable, stair_stable)
        dwell = torch.where(
            flat,
            torch.full_like(self.landing_time, flat_completion_duration),
            torch.full_like(self.landing_time, landing_duration),
        )
        self.landing_time[:] = torch.where(stable, self.landing_time + env.step_dt, 0.0)
        self.settled[:] = self.landing_time >= dwell - 1.e-6
        # Boundary crossing still truncates for tile isolation; it is never a
        # success unless the supported dwell has already been completed.
        # Flat success uses the whole measured episode, not the final dwell.
        # Exclude reset settling; account for a partially elapsed warmup step.
        quality_dt = (self.duration - flat_quality_warmup).clamp(min=0., max=env.step_dt) * flat
        self.flat_quality_time += quality_dt
        self.flat_stable_time += quality_dt * common_stable
        self.flat_unstable_time[:] = torch.where(
            (quality_dt > 0) & ~common_stable, self.flat_unstable_time + quality_dt, 0.)
        self.flat_max_unstable_time[:] = torch.maximum(self.flat_max_unstable_time, self.flat_unstable_time)
        # Collect sustained flat locomotion until tile exit/time limit, even
        # after its quality conditions have been met.
        return outside | (self.settled & ~flat)


class MixedStairCurriculum(ManagerTermBase):
    TRAVERSAL_NAMES = ("up", "down", "flat")
    MODE_NAMES = ("forward", "sideward", "backward", "turning")
    FLAT_QUALITY_NAMES = ("stable_fraction", "max_unstable_time", "quality_time")
    STAT_NAMES = (
        "success_rate", "timeout_rate", "failure_rate", "actual_speed",
        "speed_xy", "velocity_x", "velocity_y", "command_speed", "terrain_level",
    ) + FLAT_QUALITY_NAMES

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        shape = (env.num_envs, 3, 4)
        self.levels = torch.zeros(shape, dtype=torch.long, device=env.device)
        low, high = cfg.params["initial_up_levels"]
        if not 0 <= low <= high < env.scene.terrain.max_terrain_level:
            raise ValueError("initial_up_levels 超出地形等级范围")
        self.levels[:, 0] = torch.randint(low, high + 1, (env.num_envs, 4), device=env.device)
        self.success_streak = torch.zeros_like(self.levels)
        self.failure_streak = torch.zeros_like(self.levels)
        window = cfg.params["statistics_window"]
        if window < 1:
            raise ValueError("statistics_window 必须为正")
        self.history = torch.zeros((3, 4, window, len(self.STAT_NAMES)), device=env.device)
        self.counts = torch.zeros((3, 4), dtype=torch.long, device=env.device)
        self.cursors = torch.zeros_like(self.counts)
        self.command = env.command_manager.get_term("base_velocity")
        self.command_attempts = torch.zeros((3, 4), dtype=torch.long, device=env.device)
        self.command_successes = torch.zeros_like(self.command_attempts)
        # Independent flat-quality gate used to increase stair exposure after
        # ordinary locomotion and standing have both become reliable.
        self.flat_attempts = torch.zeros(2, dtype=torch.long, device=env.device)
        self.flat_successes = torch.zeros_like(self.flat_attempts)
        self.flat_good_windows = torch.zeros((), dtype=torch.long, device=env.device)
        self.sampling_phase = torch.zeros((), dtype=torch.long, device=env.device)
        # Tensor state keeps direction phase and gate progress in checkpoints.
        self.schema_version = torch.tensor(3, device=env.device)
        self.command_phase = torch.tensor(
            [self.command.stage, self.command.sideward_ramp, 0], device=env.device)

    def _update_command_stage(self, env, ids, success, min_episodes, threshold, windows, min_level):
        command = self.command
        kinds, modes = command.episode_traversal[ids], command.episode_mode[ids]
        eligible = ((command.episode_stage[ids] == command.stage)
                    & (command.episode_sideward_ramp[ids] == command.sideward_ramp)
                    & ((kinds == 2) | (env.scene.terrain.terrain_levels[ids] >= min_level)))
        slots = 4 * kinds + modes
        self.command_attempts += torch.bincount(slots[eligible], minlength=12).reshape(3, 4)
        self.command_successes += torch.bincount(slots[eligible & success], minlength=12).reshape(3, 4)
        active = torch.zeros_like(self.command_attempts, dtype=torch.bool)
        stair_modes = [0] if command.stage == 0 else [0, 2]
        active[0, stair_modes] = True
        active[1, stair_modes] = True
        active[2, :] = True  # Flat commands already sample all four categories.
        if torch.all(self.command_attempts[active] >= min_episodes):
            rates = self.command_successes[active].float() / self.command_attempts[active]
            self.command_phase[2] = self.command_phase[2] + 1 if torch.all(rates >= threshold) else 0
            if command.stage == 0 and self.command_phase[2] >= windows:
                command.stage = 1
                self.command_phase[2] = 0
            self.command_phase[0] = command.stage
            self.command_phase[1] = command.sideward_ramp
            self.command_attempts.zero_()
            self.command_successes.zero_()


    def state_dict(self):
        return {name: getattr(self, name).clone() for name in
                ("schema_version", "levels", "success_streak", "failure_streak", "history", "counts", "cursors",
                 "command_attempts", "command_successes", "command_phase", "flat_attempts",
                 "flat_successes", "flat_good_windows", "sampling_phase")}

    def load_state_dict(self, state):
        # Version 2 already has the same direction/terrain state. Preserve its
        # progress, but never mix old terminal-dwell outcomes into new flat gates.
        version = int(state["schema_version"]) if "schema_version" in state else -1
        if version not in (2, 3):
            raise ValueError("旧楼梯课程与前进/后退课程不兼容，请用 --finetune 加载权重")
        if set(state) != set(self.state_dict()):
            raise ValueError("课程状态不完整，请用 --finetune 加载权重")
        if int(state["command_phase"][0]) not in (0, 1) or int(state["command_phase"][1]) != 0:
            raise ValueError("无效的前进/后退课程阶段")
        expected_history_shape = self.history.shape if version == 3 else (*self.history.shape[:-1], 9)
        # Validate every shape before mutating either live or caller-owned state.
        for name, value in state.items():
            shape = expected_history_shape if name == "history" else getattr(self, name).shape
            if value.shape != shape:
                raise ValueError(f"课程状态 {name} 形状不匹配；续训须保持 num_envs/统计窗口，或使用 --finetune")
        restored = {name: value.clone() for name, value in state.items()}
        if version == 2:
            history = torch.zeros((*state["history"].shape[:-1], len(self.STAT_NAMES)),
                                  dtype=state["history"].dtype, device=state["history"].device)
            history[..., :9] = state["history"]
            history[2] = 0
            restored["history"] = history
            for name in ("counts", "cursors"):
                restored[name][2] = 0
            for name in ("success_streak", "failure_streak"):
                restored[name][:, 2] = 0
            for name in ("command_attempts", "command_successes", "flat_attempts",
                         "flat_successes", "flat_good_windows"):
                restored[name].zero_()
            restored["command_phase"][2] = 0
            restored["schema_version"].fill_(3)
        for name, value in restored.items():
            destination = getattr(self, name)
            destination.copy_(value.to(destination.device))
        self.command.stage = int(self.command_phase[0])
        self.command.sideward_ramp = int(self.command_phase[1])
        self.command.set_traversal_probabilities(
            (0.40, 0.40, 0.20) if int(self.sampling_phase) else (0.30, 0.30, 0.40)
        )
        if version == 2:
            warnings.warn("课程 v2 已迁移至 v3：保留地形等级、方向和采样阶段，重建平地统计及晋级考核窗口。",
                          stacklevel=2)

    def _update_flat_sampling(self, kinds, success, episode_standing, min_episodes, threshold, windows):
        """Promote flat coverage once moving and standing flat skills are reliable.

        ``kinds``/``success``/``episode_standing`` are all indexed by the same
        completed-episode ids, so partial reset batches stay shape-safe.
        """
        flat = kinds == 2
        if not torch.any(flat):
            return
        standing = flat & episode_standing
        moving = flat & ~episode_standing
        self.flat_attempts[0] += moving.sum()
        self.flat_successes[0] += (moving & success).sum()
        self.flat_attempts[1] += standing.sum()
        self.flat_successes[1] += (standing & success).sum()
        if torch.all(self.flat_attempts >= min_episodes):
            rates = self.flat_successes.float() / self.flat_attempts.float().clamp(min=1)
            passed = torch.all(rates >= threshold)
            self.flat_good_windows += passed.long()
            if self.sampling_phase.item() == 0 and self.flat_good_windows >= windows and bool(passed):
                self.sampling_phase.fill_(1)
                self.command.set_traversal_probabilities((0.40, 0.40, 0.20))
                self.flat_good_windows.zero_()
            elif not bool(passed):
                self.flat_good_windows.zero_()
            self.flat_attempts.zero_()
            self.flat_successes.zero_()

    def _record(self, env, ids, kinds, modes, success, timeout, failed, duration):
        tracker = env._bpx_rough_traversal
        values = torch.stack((success.float(), timeout.float(), failed.float(),
                              tracker.along_command_integral[ids] / duration,
                              tracker.speed_integral[ids] / duration,
                              tracker.velocity_integral[ids, 0] / duration,
                              tracker.velocity_integral[ids, 1] / duration,
                              tracker.command_speed_integral[ids] / duration,
                              env.scene.terrain.terrain_levels[ids].float(),
                              tracker.flat_stable_time[ids] / tracker.flat_quality_time[ids].clamp(min=env.step_dt),
                              tracker.flat_max_unstable_time[ids], tracker.flat_quality_time[ids]), dim=1)
        window = self.history.shape[2]
        for kind in range(3):
            for mode in range(4):
                rows = values[(kinds == kind) & (modes == mode)]
                count = len(rows)
                if count:
                    kept = min(count, window)
                    positions = (torch.arange(kept, device=env.device) + self.cursors[kind, mode]
                                 + count - kept) % window
                    self.history[kind, mode, positions] = rows[-kept:]
                    self.cursors[kind, mode] = (self.cursors[kind, mode] + count) % window
                    self.counts[kind, mode] += count

    def _metrics(self, env, command):
        result = {"level": env.scene.terrain.terrain_levels.float().mean()}
        result["flat/gate_moving_success_rate"] = (
            self.flat_successes[0].float() / self.flat_attempts[0].float().clamp(min=1)
        )
        result["flat/gate_standing_success_rate"] = (
            self.flat_successes[1].float() / self.flat_attempts[1].float().clamp(min=1)
        )
        result["flat/gate_moving_samples"] = self.flat_attempts[0].float()
        result["flat/gate_standing_samples"] = self.flat_attempts[1].float()
        result["flat/gate_good_windows"] = self.flat_good_windows.float()
        result["terrain_sampling_phase"] = self.sampling_phase.float()
        for kind, kind_name in enumerate(self.TRAVERSAL_NAMES):
            combined = []
            for mode, mode_name in enumerate(self.MODE_NAMES):
                size = min(int(self.counts[kind, mode]), self.history.shape[2])
                rows = self.history[kind, mode, :size]
                combined.append(rows)
                means = rows.mean(dim=0) if size else torch.zeros(len(self.STAT_NAMES), device=env.device)
                for key, value in zip(self.STAT_NAMES, means):
                    if kind == 2 or key not in self.FLAT_QUALITY_NAMES:
                        result[f"{kind_name}/{mode_name}/{key}"] = value
                result[f"{kind_name}/{mode_name}/window_samples"] = float(size)
            rows = torch.cat(combined)
            means = rows.mean(dim=0) if len(rows) else torch.zeros(len(self.STAT_NAMES), device=env.device)
            result.update({f"{kind_name}/{key}": value for key, value in zip(self.STAT_NAMES, means)
                           if kind == 2 or key not in self.FLAT_QUALITY_NAMES})
            result[f"{kind_name}/window_samples"] = float(len(rows))
            result[f"{kind_name}/episode_count"] = self.counts[kind].sum().float()
            result[f"{kind_name}/active_fraction"] = (command.episode_traversal == kind).float().mean()
            stored = self.levels[:, kind, [0, 2]] if kind != 2 else self.levels[:, kind]
            result[f"{kind_name}/stored_terrain_level"] = stored.float().mean()
        return result

    def __call__(self, env, env_ids, initial_up_levels: tuple[int, int], statistics_window: int,
                 successes_to_promote: int, failures_to_demote: int, min_stair_time: float,
                 max_linear_error: float, max_descent_linear_error: float,
                 max_angular_error: float, min_speed_ratio: float,
                 command_min_episodes: int = 128, command_success_threshold: float = 0.8,
                 command_success_windows: int = 2, command_min_level: int = 1,
                 flat_success_threshold: float = 0.90,
                 flat_success_windows: int = 2, flat_min_quality_time: float = 2.0,
                 flat_min_stable_fraction: float = 0.90, flat_max_unstable_time: float = 0.5):
        if (flat_min_quality_time <= 0 or not 0 < flat_min_stable_fraction <= 1
                or flat_max_unstable_time < 0):
            raise ValueError("平地质量要求：观察时间须为正、稳定占比在 (0, 1]、连续失稳容忍时间非负")
        terrain = env.scene.terrain
        command = env.command_manager.get_term("base_velocity")
        reset_ids = (torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]
                     if env_ids is None or isinstance(env_ids, slice)
                     else torch.as_tensor(env_ids, dtype=torch.long, device=env.device))
        ids = reset_ids[env.episode_length_buf[reset_ids] > 0]
        success_rate = 0.0
        if ids.numel():
            tracker = env._bpx_rough_traversal
            duration = tracker.duration[ids].clamp(min=env.step_dt)
            kinds, modes = command.episode_traversal[ids], command.episode_mode[ids]
            limit = torch.where(kinds == 1, max_descent_linear_error, max_linear_error)
            tracking_ok = ((tracker.linear_error[ids] / duration <= limit)
                           & (tracker.angular_error[ids] / duration <= max_angular_error)
                           & (tracker.along_command_integral[ids]
                              >= min_speed_ratio * tracker.command_speed_integral[ids]))
            # Require signed yaw progress; standing still or rotating backwards
            # must never validate a nonzero rotation command.
            yaw_target = tracker.command_yaw_integral[ids]
            rotating = yaw_target / duration > 0.1
            yaw_ok = ((tracker.along_yaw_integral[ids] >= min_speed_ratio * yaw_target)
                      & (tracker.angular_error[ids] <= 0.5 * yaw_target))
            tracking_ok &= ~rotating | yaw_ok
            standing = command.episode_standing[ids]
            tracking_ok &= ~standing | ((tracker.linear_error[ids] / duration <= 0.1)
                                        & (tracker.angular_error[ids] / duration <= 0.1))
            failed = env.termination_manager.terminated[ids]
            timeout = env.termination_manager.get_term("time_out")[ids]
            flat_quality_ok = (
                (tracker.flat_quality_time[ids] >= flat_min_quality_time - 1.e-5)
                & (tracker.flat_stable_time[ids]
                   >= flat_min_stable_fraction * tracker.flat_quality_time[ids] - 1.e-5)
                & (tracker.flat_max_unstable_time[ids] <= flat_max_unstable_time + 1.e-5))
            quality_ok = torch.where(kinds == 2, flat_quality_ok, tracker.settled[ids])
            success = (quality_ok & tracking_ok & ~failed
                       & ((tracker.rough_time[ids] >= min_stair_time) | (kinds == 2)))
            # Every unfinished crossing/timeout is a failed traversal, including
            # jumping out of the tile without a supported bottom landing.
            bad = ~success
            self.success_streak[ids, kinds, modes] = torch.where(
                success, self.success_streak[ids, kinds, modes] + 1, 0)
            self.failure_streak[ids, kinds, modes] = torch.where(
                bad, self.failure_streak[ids, kinds, modes] + 1, 0)
            promote = self.success_streak[ids, kinds, modes] >= successes_to_promote
            demote = self.failure_streak[ids, kinds, modes] >= failures_to_demote
            self.levels[ids, kinds, modes] = torch.where(
                kinds == 2, 0, (terrain.terrain_levels[ids] + promote.long() - demote.long()).clamp(
                    0, terrain.max_terrain_level - 1))
            changed = promote | demote
            self.success_streak[ids[changed], kinds[changed], modes[changed]] = 0
            self.failure_streak[ids[changed], kinds[changed], modes[changed]] = 0
            self._record(env, ids, kinds, modes, success, timeout, failed, duration)
            self._update_flat_sampling(
                kinds, success, command.episode_standing[ids], command_min_episodes,
                flat_success_threshold, flat_success_windows,
            )
            self._update_command_stage(env, ids, success, command_min_episodes,
                                       command_success_threshold, command_success_windows, command_min_level)
            success_rate = success.float().mean()
        metrics = self._metrics(env, command)  # Completed episode tags, before sampling.
        command.prepare_next_episode(reset_ids)
        kinds, modes = command.next_traversal[reset_ids], command.next_mode[reset_ids]
        levels = self.levels[reset_ids, kinds, modes]
        terrain.terrain_levels[reset_ids] = levels
        # Equal geometry proportions allocate two columns to each terrain kind;
        # episode probabilities are independent of this physical column layout.
        terrain.terrain_types[reset_ids] = 2 * kinds + torch.randint(0, 2, (len(reset_ids),), device=env.device)
        alpha = levels.float().unsqueeze(1) / (terrain.max_terrain_level - 1)
        initial = torch.tensor(command.cfg.descent_initial_speed_range, device=env.device)
        final = torch.tensor(command.cfg.descent_final_speed_range, device=env.device)
        command.next_descent_speed_range[reset_ids] = initial + alpha * (final - initial)
        zeros = torch.zeros(len(reset_ids), dtype=torch.bool, device=env.device)
        terrain.update_env_origins(reset_ids, zeros, zeros)
        return {**metrics, "success_rate": success_rate,
                "command_stage": float(command.stage),
                "command_sideward_ramp": float(command.sideward_ramp),
                "command_good_windows": float(self.command_phase[2])}
