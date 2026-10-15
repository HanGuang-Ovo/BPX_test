# -*- coding: utf-8 -*-
"""楼梯任务的训练侧专用组件:速度指令、重置、奖励函数与课程(掌握度)追踪。

设计原则:
- 策略输入中不附加相位时钟、地形高度或路线状态 —— 策略保持"盲走";
- 到达楼梯时的步态相位通过真实的行走过程自然产生(随机接近距离/速度/
  起步时序),而不是把一个任意关节位姿"传送"到台阶边缘上。
"""

import numpy as np
import torch

from isaaclab.envs.mdp import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from ..stair_geometry import generator_layouts


HOLD_CONDITIONS = ("linear", "angular", "support", "drift", "upright")
FOOT_DIAGNOSTIC_BUFFERS = (
    "walking_time", "swing_feet_time", "clearance_active_time", "clearance_deficit_time",
    "near_ground_drag_time", "riser_strike_time", "touchdown_count", "touchdown_speed_sum",
)


class StairFootContacts:
    """每策略步更新一次的接触缓存;奖励保留上一帧,验收使用 raw。

    首个 reset 后的采样不结算冲击。touchdown 要求上一帧过滤接触已经
    消失,避免一次接触信号缺失后重复结算同一次落脚。
    """

    def __init__(self, contact):
        self.raw = torch.zeros_like(contact)
        self.filtered = torch.zeros_like(contact)
        self.touchdown = torch.zeros_like(contact)
        self.previous_velocity_z = torch.zeros_like(contact, dtype=torch.float)
        self.touchdown_speed = torch.zeros_like(self.previous_velocity_z)
        self.history_valid = torch.zeros(contact.shape[0], device=contact.device, dtype=torch.bool)

    def update(self, contact, velocity_z):
        self.touchdown[:] = contact & ~self.filtered & self.history_valid[:, None]
        self.touchdown_speed[:] = torch.where(self.touchdown, (-self.previous_velocity_z).clamp(min=0), 0.)
        self.filtered[:] = contact | self.raw
        self.raw[:] = contact
        self.previous_velocity_z[:] = velocity_z
        self.history_valid[:] = True

    def reset(self, env_ids):
        for name in ("raw", "filtered", "touchdown", "history_valid"):
            getattr(self, name)[env_ids] = False
        self.previous_velocity_z[env_ids] = self.touchdown_speed[env_ids] = 0.


def stop_success(command):
    """统一的实际停驻验收;豁免回合不计为停驻成功。"""
    return (command.stop_required & command.stop_done & (command.hold_observed >= .5)
            & (command.hold_good >= .8 * command.hold_observed))


def _uniform(n, limits, device):
    """在 limits=(low, high) 内批量采样 n 个均匀随机数(torch 实现)。"""
    return torch.empty(n, device=device).uniform_(*limits)


def terrain_table(env):
    """构建(并缓存到 env 上)全场地形几何表,供重置/奖励/指令逻辑查询。

    表的内容由 stair_geometry.generator_layouts 复现的地形生成结果打包而成:
      edges:  (rows, cols, capacity) 每块 tile 各台阶边缘的局部 x 坐标,
              不足处填充 +inf(便于布尔比较);
      deltas: 同形状,每条边缘相对前一级的高度增量(踏差);
      widths: (rows, cols) 每块 tile 的楼梯有效宽度;
      counts: (rows, cols) 每块 tile 的单侧台阶数(平地为 0);
      flat_columns / stair_columns: 平地列与楼梯列的编号。

    首次调用后缓存于 env._bpx_stair_table,避免每步重建。
    """
    if hasattr(env, "_bpx_stair_table"):
        return env._bpx_stair_table
    cfg = env.scene.terrain.cfg.terrain_generator
    layouts, flat_columns = generator_layouts(cfg)
    # 任务要求同时存在平地列(回放/平地训练)与楼梯列(主任务)。
    if not flat_columns or len(flat_columns) == cfg.num_cols:
        raise ValueError("Stair task requires both flat and stair columns")
    # capacity = 单侧最大台阶数的两倍(上行+下行边缘总数)。
    capacity = max(2 * item.count for row in layouts for item in row)
    edges = torch.full((cfg.num_rows, cfg.num_cols, capacity), float("inf"), device=env.device)
    deltas = torch.zeros_like(edges)
    widths = torch.empty((cfg.num_rows, cfg.num_cols), device=env.device)
    counts = torch.zeros_like(widths, dtype=torch.long)
    for row, items in enumerate(layouts):
        for col, item in enumerate(items):
            n = len(item.edges)
            # 边缘坐标从 tile 中心系转为以 tile 原点为基准的局部坐标。
            edges[row, col, :n] = torch.as_tensor(item.edges - cfg.size[0] / 2, device=env.device)
            # 高度增量 = 相邻台阶高度差(首级相对平地)。
            deltas[row, col, :n] = torch.as_tensor(np.diff(np.r_[0., item.heights]), device=env.device)
            widths[row, col], counts[row, col] = item.width, item.count
    flat_columns = torch.tensor(flat_columns, device=env.device, dtype=torch.long)
    stair_columns = torch.tensor([i for i in range(cfg.num_cols) if i not in flat_columns.tolist()],
                                 device=env.device, dtype=torch.long)
    env._bpx_stair_table = edges, deltas, widths, counts, flat_columns, stair_columns
    return env._bpx_stair_table


def ground_height(env, xy, env_ids=None):
    """按已生成实体几何求解世界高度(不做边缘插值)。

    xy 形状为 (N, ..., 2)。tile 原点的 z 被抬到其平台高度上,
    但网格顶点本身以世界 z=0 为地面基准,因此这里不加 origin.z。
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    edges, deltas, widths, *_ = terrain_table(env)
    terrain = env.scene.terrain
    row, col = terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    # extra: 让 per-env 的表格能广播到 xy 的额外维度(如脚数维度)。
    extra = (1,) * (xy.ndim - 2)
    local = xy - env.scene.env_origins[env_ids, :2].reshape(len(env_ids), *extra, 2)
    edge = edges[row, col].reshape(len(env_ids), *extra, -1)
    delta = deltas[row, col].reshape(len(env_ids), *extra, -1)
    # 高度 = 所有过往边缘的高度增量之和(x 在边缘右侧即为"已跨过")。
    height = ((local[..., 0, None] >= edge) * delta).sum(dim=-1)
    # 楼梯侧向之外的区域(|y| 超出半宽)高度为 0(侧向落差)。
    inside = local[..., 1].abs() <= widths[row, col].reshape(len(env_ids), *extra) / 2
    return torch.where(inside, height, 0.)


class StairPositionAction(JointPositionAction):
    """部署用的仿射动作 + 每回合固定的执行器目标误差(建模电机零偏)。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # 每个环境采样一次、整回合保持不变的目标偏置,模拟真实电机
        # 的固定零位误差(部署时无法消除的常值扰动)。
        self.target_error = torch.zeros_like(self.raw_actions)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.target_error[ids] = torch.empty_like(self.target_error[ids]).uniform_(*self.cfg.target_error_range)

    def process_actions(self, actions):
        super().process_actions(actions)
        self._processed_actions += self.target_error


@configclass
class StairPositionActionCfg(JointPositionActionCfg):
    class_type: type = StairPositionAction
    # 执行器目标误差的采样范围(单位:rad),整回合固定。
    target_error_range: tuple[float, float] = (-.02, .02)


class StairVelocityCommand(CommandTerm):
    """每个采样回合内至多一次"空间触发的停车",之后重启并驶出。

    phase 状态机:0 前进, 1 减速, 2 原地保持。phase 只是生成器内部状态,
    actor 只收到三个速度指令分量。速度目标按时间斜率受限地逼近(slew),
    保持时长从指令精确归零后才开始计。
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros((self.num_envs, 3), device=self.device)  # 实际输出指令(机体系 x/y/z)
        self.target = torch.zeros_like(self.vel_command_b)                        # 斜坡逼近的目标指令
        self.anchor = torch.zeros((self.num_envs, 2), device=self.device)         # 停车锚点(世界系 xy)
        # 各类按环境维度的标量状态缓冲(计时器/误差累计/噪声等)。
        for name in ("speed_timer", "start_wait", "stop_x", "goal_x", "hold_time", "hold_elapsed",
                     "hold_duration", "hold_good", "hold_observed", "goal_stable", "duration",
                     "linear_error", "angular_error", "previous_x", "progress", "speed", "lateral_noise", "yaw_noise"):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.route = torch.zeros_like(self.phase)  # 路线: 0 平地, 1 上行, 2 下行
        self.stop_stage = torch.zeros_like(self.phase)  # 停车点所处阶段(前/中/末台阶)
        self.stop_difficulty = torch.zeros_like(self.phase)  # 0 接近区,1 踏面中心,2 台阶边缘
        # reset 事件早于 command.reset 的诊断结算。新回合计划先暂存,
        # 直到 _resample_command 才替换旧回合的采样标记。
        self.next_stop_required = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.next_stop_difficulty = torch.zeros_like(self.phase)
        for name in ("stop_required", "stop_done", "goal_reached", "completed", "was_stopped"):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
        self.completion_rewarded = torch.zeros_like(self.completed)
        self.traversal_rewarded = torch.zeros_like(self.completed)
        self.stop_rewarded = torch.zeros_like(self.completed)
        self.task_success = torch.zeros_like(self.completed)
        self.last_stair_stage = torch.full_like(self.phase, -1)
        for name in (*FOOT_DIAGNOSTIC_BUFFERS, *(f"hold_{item}_good" for item in HOLD_CONDITIONS)):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))
        self.metrics = {name: torch.zeros(self.num_envs, device=self.device)
                        for name in ("completed", "task_success", "stop_success", "ascent", "descent", "flat",
                                     "hold_observed_s", "hold_joint_pass_fraction", "stop_difficulty",
                                     "clearance_active_fraction", "clearance_deficit_fraction",
                                     "near_ground_drag_fraction", "riser_strike_fraction", "touchdowns",
                                     "mean_touchdown_speed", *(f"hold_{item}_pass_fraction" for item in HOLD_CONDITIONS))}
        # 挂到 env 上供重置事件/终止项/课程项互相访问。
        env._bpx_stairs_command = self
        terrain_table(env)

    @property
    def command(self):
        return self.vel_command_b

    def reset(self, env_ids=None):
        """诊断按实际采样人群统计,避免未安排停驻的回合稀释结果。"""
        all_ids = torch.arange(self.num_envs, device=self.device)
        ids = all_ids if env_ids is None else (all_ids[env_ids] if isinstance(env_ids, slice)
                                               else torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
        sampled = ids[self.stop_required[ids]]
        observed = sampled[self.hold_observed[sampled] > 0]
        walking = ids[self.walking_time[ids] > 0]
        diagnostic = {
            "hold_sample_count": float(len(sampled)), "hold_observed_count": float(len(observed)),
            "hold_completed_count": self.stop_done[sampled].sum().item(),
            "hold_success_count": stop_success(self)[sampled].sum().item(),
            "walking_sample_count": float(len(walking)),
            "stop_success": stop_success(self)[sampled].float().mean().item() if len(sampled) else 0.,
            "hold_observed_s": self.hold_observed[sampled].mean().item() if len(sampled) else 0.,
            "hold_joint_pass_fraction": (self.hold_good[observed] / self.hold_observed[observed]).mean().item()
            if len(observed) else 0.,
            "stop_difficulty": self.stop_difficulty[sampled].float().mean().item() if len(sampled) else 0.,
        }
        for name in HOLD_CONDITIONS:
            key = f"hold_{name}_pass_fraction"
            diagnostic[key] = self.metrics[key][observed].mean().item() if len(observed) else 0.
        for key in ("clearance_active_fraction", "clearance_deficit_fraction",
                    "near_ground_drag_fraction", "riser_strike_fraction"):
            diagnostic[key] = self.metrics[key][walking].mean().item() if len(walking) else 0.
        # reset 事件已经换了地形/路线;终止前的阶段和路线使用监视器快照。
        stairs = (self.metrics["ascent"][ids] + self.metrics["descent"][ids]) > 0
        failed = ids[stairs & self._env.termination_manager.terminated[ids]]
        diagnostic["stair_failure_count"] = float(len(failed))
        for stage, name in enumerate(("start", "middle", "end")):
            diagnostic[f"failure_{name}_fraction"] = (
                (self.last_stair_stage[failed] == stage).float().mean().item() if len(failed) else 0.)
        extras = super().reset(ids)
        extras.update(diagnostic)
        return extras

    def _resample_command(self, env_ids):
        """重置时为指定环境重抽:路线、停车点、目标线、各计时器。"""
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(ids)
        self.stop_required[ids] = self.next_stop_required[ids]
        self.stop_difficulty[ids] = self.next_stop_difficulty[ids]
        terrain = self._env.scene.terrain
        edges, _, _, counts, *_ = terrain_table(self._env)
        row, col = terrain.terrain_levels[ids], terrain.terrain_types[ids]
        count = counts[row, col]
        flat = count == 0
        # 下行路线:当前 env 的 route 已由 reset_stair_root 提前写入(见其注释),
        # 这里沿用;否则非平地默认上行。
        descent = ~flat & (self.route[ids] == 2)
        self.route[ids] = torch.where(flat, 0, torch.where(descent, 2, 1))
        # 上行从第 0 级边缘开始;下行(倒序)从最后一条边缘开始。
        start_index = torch.where(descent, count, 0)
        end_index = torch.where(descent, 2 * count - 1, count - 1).clamp(min=0)
        last = edges[row, col, end_index]
        # 目标线:平地固定 8 m;楼梯为最后一条边缘 + exit_distance。
        self.goal_x[ids] = torch.where(flat, 8., last + self.cfg.exit_distance)
        # 停车阶段三选一:前段/中段/末段台阶,对应不同的台阶边缘。
        stage = torch.randint(0, 3, (n,), device=self.device)
        step_index = torch.where(stage == 0, 0, torch.where(stage == 1, count // 2, count - 1)).clamp(min=0)
        stop_edge = edges[row, col, start_index + step_index]
        # 独立停驻课程:先在平坦接近区停,再到踏面中心,最后在台阶边缘停。
        first = edges[row, col, start_index]
        next_index = (start_index + step_index + 1).clamp(max=edges.shape[-1] - 1)
        next_edge = edges[row, col, next_index]
        span = (next_edge - stop_edge).nan_to_num(nan=.4, posinf=.4).clamp(0., .4)
        approach = first - self.cfg.stop_approach_distance + _uniform(n, (-.03, .03), self.device)
        tread = stop_edge + .5 * torch.where(span > 0, span, .4)
        edge_stop = stop_edge + _uniform(n, (-.08, .08), self.device)
        stop = torch.where(self.stop_difficulty[ids] == 0, approach,
                           torch.where(self.stop_difficulty[ids] == 1, tread, edge_stop))
        self.stop_x[ids] = torch.where(flat, 2., stop)
        self.stop_stage[ids] = stage
        # 是否停车已由 reset 事件决定,用于为停驻选择匹配难度的地形。
        self.stop_required[ids] &= ~flat
        self.hold_duration[ids] = _uniform(n, self.cfg.hold_time_range, self.device)     # 保持时长
        self.start_wait[ids] = _uniform(n, self.cfg.start_wait_range, self.device)       # 起步前等待
        self.previous_x[ids] = self.robot.data.root_pos_w[ids, 0] - self._env.scene.env_origins[ids, 0]
        self.anchor[ids] = self.robot.data.root_pos_w[ids, :2]
        self.vel_command_b[ids] = 0
        self.target[ids] = 0
        self.phase[ids] = 0
        # 清零各类计时/统计缓冲。
        for name in ("speed_timer", "hold_time", "hold_elapsed", "hold_good", "hold_observed",
                     "goal_stable", "duration", "linear_error", "angular_error", "progress"):
            getattr(self, name)[ids] = 0
        for name in ("stop_done", "goal_reached", "completed", "was_stopped"):
            getattr(self, name)[ids] = False
        self.completion_rewarded[ids] = self.task_success[ids] = False
        self.traversal_rewarded[ids] = self.stop_rewarded[ids] = False
        self.last_stair_stage[ids] = -1
        for name in (*FOOT_DIAGNOSTIC_BUFFERS, *(f"hold_{item}_good" for item in HOLD_CONDITIONS)):
            getattr(self, name)[ids] = 0.
        if hasattr(self, "foot_contacts"):
            self.foot_contacts.reset(ids)

    def _update_command(self):
        """每步更新:重抽速度噪声、推进停车状态机、斜坡逼近目标指令。"""
        dt = self._env.step_dt
        xy = self.robot.data.root_pos_w[:, :2]
        local = xy - self._env.scene.env_origins[:, :2]
        flat = self.route == 0
        self.start_wait -= dt
        self.speed_timer -= dt
        # 速度重采样计时到期的环境:重抽前向速度与横向/偏航噪声。
        resample = self.speed_timer <= 0
        ids = resample.nonzero().flatten()
        n = len(ids)
        self.speed[ids] = _uniform(n, self.cfg.forward_speed_range, self.device)
        self.lateral_noise[ids] = _uniform(n, self.cfg.lateral_range, self.device)
        self.yaw_noise[ids] = _uniform(n, self.cfg.yaw_range, self.device)
        self.speed_timer[ids] = _uniform(n, self.cfg.speed_resampling_range, self.device)
        self.target[:, 0] = self.speed
        # 横向/偏航目标 = 回中控制项(把走偏拉回中线/朝向)+ 有界噪声。
        self.target[:, 1] = (-self.cfg.lateral_gain * local[:, 1] + self.lateral_noise).clamp(-.08, .08)
        self.target[:, 2] = (-self.cfg.heading_gain * self.robot.data.heading_w + self.yaw_noise).clamp(-.20, .20)
        # 平地回放:保留前进/后退、横向与偏航指令(更丰富的平地指令分布)。
        flat_ids = (resample & flat).nonzero().flatten()
        if len(flat_ids):
            self.speed[flat_ids] = _uniform(len(flat_ids), self.cfg.flat_x_range, self.device)
            self.lateral_noise[flat_ids] = _uniform(len(flat_ids), self.cfg.flat_y_range, self.device)
            self.yaw_noise[flat_ids] = _uniform(len(flat_ids), self.cfg.flat_yaw_range, self.device)
            # 概率性抽到"原地站立"回合:三个分量全部置零。
            standing = torch.rand(len(flat_ids), device=self.device) < self.cfg.flat_standing_probability
            self.speed[flat_ids[standing]] = 0
            self.lateral_noise[flat_ids[standing]] = 0
            self.yaw_noise[flat_ids[standing]] = 0
        self.target[flat, 0] = self.speed[flat]
        self.target[flat, 1] = self.lateral_noise[flat]
        self.target[flat, 2] = self.yaw_noise[flat]
        # 制动距离估计 v^2/(2a) 再加跟踪/延迟裕量。若用固定 0.30 m 包络,
        # 慢速接近时会在脚还没够到第一级台阶之前就触发停车,
        # 训练不到"骑跨在台阶边缘上保持"的真正场景。
        braking_distance = (self.vel_command_b[:, 0].clamp(min=0).square()
                            / (2 * self.cfg.slew_rate[0]) + .04).clamp(.06, .30)
        # 进入减速相位的条件:要求停车、尚未完成、仍在前进、
        # 且当前位置已进入停车线之前的制动包络。
        brake = (self.stop_required & ~self.stop_done & (self.phase == 0)
                 & (local[:, 0] >= self.stop_x - braking_distance))
        self.phase[brake] = 1
        # 保持阶段计时:剩余时间递减、已保持时长递增;计时结束后
        # 回到前进相位并标记停车完成。
        holding = self.phase == 2
        self.hold_time[holding] -= dt
        self.hold_elapsed[holding] += dt
        restart = holding & (self.hold_time <= 0)
        self.phase[restart] = 0
        self.stop_done[restart] = True
        # 到达目标线(仅楼梯回合计入;之后由监视器在双脚越过后结束回合)。
        self.goal_reached |= ~flat & (local[:, 0] >= self.goal_x)
        # 到达是穿越终点,而不是额外的无限期站立任务:
        # 监视器会在所有脚越过最后一条边缘后结束回合。
        zero_target = (self.phase != 0) | (self.start_wait > 0)
        self.target[zero_target] = 0
        # 指令以 slew_rate 为上限的斜坡逼近目标,模拟平滑加减速。
        max_delta = self.vel_command_b.new_tensor(self.cfg.slew_rate) * dt
        self.vel_command_b += (self.target - self.vel_command_b).clamp(-max_delta, max_delta)
        # 减速相位且指令范数已低于阈值 -> 精确置零并进入保持相位。
        at_zero = torch.linalg.vector_norm(self.vel_command_b, dim=1) < .005
        enter_hold = (self.phase == 1) & at_zero
        self.vel_command_b[enter_hold] = 0
        self.phase[enter_hold] = 2
        self.hold_time[enter_hold] = self.hold_duration[enter_hold]
        # 锚点只属于"一段连续的零指令区间":新进入停车时更新锚点。
        stopped = torch.linalg.vector_norm(self.vel_command_b, dim=1) < .02
        self.anchor[stopped & ~self.was_stopped] = xy[stopped & ~self.was_stopped]
        self.was_stopped[:] = stopped

    def _update_metrics(self):
        """写入 TensorBoard 指标:完成率、停车成功率、各路线占比。"""
        self.metrics["completed"][:] = self.completed.float()
        self.metrics["task_success"][:] = self.task_success.float()
        self.metrics["stop_success"][:] = stop_success(self).float()
        self.metrics["hold_observed_s"][:] = self.hold_observed
        self.metrics["hold_joint_pass_fraction"][:] = self.hold_good / self.hold_observed.clamp(min=.01)
        self.metrics["stop_difficulty"][:] = self.stop_difficulty.float()
        for name in HOLD_CONDITIONS:
            self.metrics[f"hold_{name}_pass_fraction"][:] = (
                getattr(self, f"hold_{name}_good") / self.hold_observed.clamp(min=.01))
        self.metrics["clearance_active_fraction"][:] = self.clearance_active_time / self.swing_feet_time.clamp(min=.01)
        self.metrics["clearance_deficit_fraction"][:] = self.clearance_deficit_time / self.clearance_active_time.clamp(min=.01)
        self.metrics["near_ground_drag_fraction"][:] = self.near_ground_drag_time / (4 * self.walking_time).clamp(min=.01)
        self.metrics["riser_strike_fraction"][:] = self.riser_strike_time / self.walking_time.clamp(min=.01)
        self.metrics["touchdowns"][:] = self.touchdown_count
        self.metrics["mean_touchdown_speed"][:] = self.touchdown_speed_sum / self.touchdown_count.clamp(min=1.)
        for index, name in enumerate(("flat", "ascent", "descent")):
            self.metrics[name][:] = (self.route == index).float()


@configclass
class StairVelocityCommandCfg(CommandTermCfg):
    class_type: type = StairVelocityCommand
    asset_name: str = "robot"
    # 指令重采样周期极长(回合内由内部状态机管理,而非按时间重抽)。
    resampling_time_range: tuple[float, float] = (1000., 1000.)
    debug_vis: bool = False
    forward_speed_range: tuple[float, float] = (.10, .40)   # 前向速度范围
    lateral_range: tuple[float, float] = (-.02, .02)        # 横向噪声范围
    yaw_range: tuple[float, float] = (-.05, .05)            # 偏航噪声范围
    speed_resampling_range: tuple[float, float] = (2., 5.)  # 速度重采样周期
    slew_rate: tuple[float, float, float] = (.5, .4, .8)    # 指令斜率上限 (vx, vy, wz)
    start_wait_range: tuple[float, float] = (0., .5)        # 起步前随机等待
    stop_probability: float = .75                           # 楼梯回合要求停车的概率
    stop_approach_distance: float = .50                     # 初期停驻点距首阶的距离
    hold_time_range: tuple[float, float] = (1., 3.)         # 停车保持时长
    exit_distance: float = .85                              # 越过最后边缘后的驶出距离
    lateral_gain: float = .4                                # 横向回中增益
    heading_gain: float = .8                                # 朝向回中增益
    flat_x_range: tuple[float, float] = (-.7, .7)           # 平地前/后速度范围
    flat_y_range: tuple[float, float] = (-.25, .25)         # 平地横向指令范围
    flat_yaw_range: tuple[float, float] = (-.7, .7)         # 平地偏航指令范围
    flat_standing_probability: float = .2                   # 平地抽到原地站立的概率


def reset_stair_root(env, env_ids, flat_probability, replay_probability, approach_range, height):
    """在地面或顶部平台出生;绝不在踢面上强加一个虚假的步态位姿。"""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    n = len(ids)
    terrain = env.scene.terrain
    edges, _, _, counts, flat_cols, stair_cols = terrain_table(env)
    # 按概率选择平地列或楼梯列。
    flat = torch.rand(n, device=env.device) < flat_probability
    chosen_flat = flat_cols[torch.randint(len(flat_cols), (n,), device=env.device)]
    chosen_stair = stair_cols[torch.randint(len(stair_cols), (n,), device=env.device)]
    col = torch.where(flat, chosen_flat, chosen_stair)
    # 连续通过与停驻独立选难度。停驻级别 0/1/2 对应最低级地形上的
    # 接近区/踏面/边缘;之后每升一级,增加一级楼梯难度。
    curriculum = env._bpx_stair_curriculum
    command = env._bpx_stairs_command
    mastery = curriculum.mastery[ids]
    stop_level = curriculum.stop_mastery[ids]
    planned_hold = ~flat & (torch.rand(n, device=env.device) < command.cfg.stop_probability)
    stop_terrain_level = (stop_level - 2).clamp(min=0)
    row = torch.where(planned_hold, torch.minimum(mastery, stop_terrain_level), mastery)
    frontier = row.clone()
    # 回放回合:楼梯列中按概率随机降到更低难度行(复习旧等级)。
    replay = ~flat & (torch.rand(n, device=env.device) < replay_probability) & (row > 0)
    lower = (torch.rand(n, device=env.device) * row.clamp(min=1)).long()
    row[replay] = lower[replay]
    # 写回 terrain 的行/列索引与原点,使该 env 使用选定的 tile。
    terrain.terrain_levels[ids], terrain.terrain_types[ids] = row, col
    terrain.env_origins[ids] = terrain.terrain_origins[row, col]
    # 指令重置发生在 reset 事件之后。这里先把上/下行决策存进 command.route,
    # 保证根部重置与指令重采样对路线的判断始终一致。
    descent = ~flat & (torch.rand(n, device=env.device) < .5)
    count = counts[row, col]
    # 上行从第 0 级边缘之前出生;下行从最后一条边缘(顶部)之前出生。
    first_index = torch.where(descent, count, 0)
    first = edges[row, col, first_index]
    state = env.scene["robot"].data.default_root_state[ids].clone()
    state[:, :2] = terrain.env_origins[ids, :2]
    # x:平地在 tile 原点;楼梯在首级边缘前 0.60~1.50 m 的随机接近距离处,
    # 使到达台阶时的步态相位自然多样化。
    state[:, 0] += torch.where(flat, torch.zeros(n, device=env.device), first - _uniform(n, approach_range, env.device))
    # y:在楼梯宽度内向中线加 ±8 cm 抖动。
    state[:, 1] += _uniform(n, (-.08, .08), env.device)
    # z:出生位置的真实地面高度 + 名义站立高度 + ±1 cm 抖动。
    state[:, 2] = ground_height(env, state[:, :2], ids) + height + _uniform(n, (-.01, .01), env.device)
    # 姿态:小随机欧拉角(横滚/俯仰 ±3°,偏航 ±8°);速度/角速度 ±0.03。
    state[:, 3:7] = quat_from_euler_xyz(_uniform(n, (-.03, .03), env.device),
                                      _uniform(n, (-.03, .03), env.device),
                                      _uniform(n, (-.08, .08), env.device))
    state[:, 7:] = torch.empty_like(state[:, 7:]).uniform_(-.03, .03)
    env.scene["robot"].write_root_state_to_sim(state, env_ids=ids)
    # 指令管理器稍后重采样时,依据出生位置推断路线(此处提前写好)。
    command.route[ids] = torch.where(flat, 0, torch.where(descent, 2, 1))
    command.next_stop_required[ids] = planned_hold
    command.next_stop_difficulty[ids] = stop_level.clamp(max=2)
    # 标记自适应(可升降级)回合:平地与回放回合不参与升降级统计。
    curriculum.adaptive[ids] = ~flat & ~replay & (frontier == mastery)
    curriculum.stop_adaptive[ids] = planned_hold & ~replay & (frontier == stop_terrain_level)


class StairEpisodeMonitor(ManagerTermBase):
    """在 reset 清空状态前,记录完整驶出与受令保持的统计(以终止项形式挂载)。"""

    def __call__(self, env, feet_cfg, sensor_cfg, final_hold, flat_episode_s, exit_speed_limit=.65,
                 diagnostic_clearance=None):
        c = env._bpx_stairs_command
        robot = env.scene["robot"]
        local_x = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        dx = local_x - c.previous_x
        moving = torch.linalg.vector_norm(c.command[:, :2], dim=1) > .04
        # 只在"楼梯回合 + 指令要求移动 + 已离开第 0 步"时统计前进速度;
        # 带符号位移可抵消被踢面卡住时的前后摇摆 —— 若把负向运动截为零,
        # 会白白奖励每一个向前的半周期。
        valid = (env.episode_length_buf > 1) & (c.route != 0) & moving
        c.progress[:] = torch.where(valid, dx.clamp(-.6 * env.step_dt, .6 * env.step_dt) / env.step_dt, 0.)
        c.previous_x[:] = local_x
        c.duration += env.step_dt
        # 速度跟踪误差按时间积分(供课程项的事后判定使用)。
        c.linear_error += torch.linalg.vector_norm(c.command[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1) * env.step_dt
        c.angular_error += (c.command[:, 2] - robot.data.root_ang_vel_b[:, 2]).abs() * env.step_dt
        # "稳定"判定:速度小、角速度小、至少 3 脚支撑且躯干接近直立。
        speed_ok = (torch.linalg.vector_norm(robot.data.root_lin_vel_w, dim=1) < .08)
        angular_ok = torch.linalg.vector_norm(robot.data.root_ang_vel_b, dim=1) < .20
        foot_contacts = update_foot_contacts(env, feet_cfg, sensor_cfg)
        contact = foot_contacts.raw
        supported = contact.sum(dim=1) >= 3
        stable = speed_ok & angular_ok & supported & (robot.data.projected_gravity_b[:, 2] < -.85)
        # 保持统计:处于保持相位且超过 0.3 s 后开始计入"观测保持";
        # 同时满足稳定 + 距锚点漂移 < 0.10 m 才计入"优质保持"。
        hold = (c.phase == 2) & (c.hold_elapsed >= .3)
        drift = torch.linalg.vector_norm(robot.data.root_pos_w[:, :2] - c.anchor, dim=1) < .10
        c.hold_observed += hold.float() * env.step_dt
        c.hold_good += (hold & stable & drift).float() * env.step_dt
        conditions = {"linear": speed_ok, "angular": angular_ok, "support": supported,
                      "drift": drift, "upright": robot.data.projected_gravity_b[:, 2] < -.85}
        for name, good in conditions.items():
            getattr(c, f"hold_{name}_good").add_((hold & good).float() * env.step_dt)
        c.touchdown_count += foot_contacts.touchdown.sum(dim=1)
        c.touchdown_speed_sum += foot_contacts.touchdown_speed.sum(dim=1)
        if diagnostic_clearance is not None:
            record_foot_diagnostics(env, feet_cfg, sensor_cfg, **diagnostic_clearance)
        # 所有脚的 x 都越过 goal_x - 0.55 m:确保全身离开最后一级。
        feet_x = robot.data.body_pos_w[:, feet_cfg.body_ids, 0] - env.scene.env_origins[:, 0, None]
        all_feet_out = feet_x.amin(dim=1) > c.goal_x - .55
        # 有支撑、直立且在移动的驶出即算有效。若在此处再要求一次完美的
        # 零速保持,会让整个回合卡在出口处积累不可观测的位置误差惩罚,
        # 哪怕楼梯早已走完。
        safe_exit = ((torch.linalg.vector_norm(robot.data.root_lin_vel_w[:, :2], dim=1) <= exit_speed_limit)
                     & (contact.sum(dim=1) >= 1) & (robot.data.projected_gravity_b[:, 2] < -.85))
        goal_ok = c.goal_reached & safe_exit & all_feet_out & (~c.stop_required | c.stop_done)
        # 需连续 final_hold 秒满足条件才算完成(防瞬时抖动误判)。
        c.goal_stable[:] = torch.where(goal_ok, c.goal_stable + env.step_dt, 0.)
        c.completed[:] = (c.route != 0) & (c.goal_stable >= final_hold)
        # 平地回合固定时长结束。
        flat_end = (c.route == 0) & (c.duration >= flat_episode_s)
        # completed 会在下一次 command.compute() 之前被 reset 消费掉。
        # 必须在这里快照终局指标,否则 completed 永远记录为 0。
        c._update_metrics()
        return c.completed | flat_end


class StairMasteryCurriculum(ManagerTermBase):
    """通过地形与停驻位置/地形分开培养;完整任务仍联合验收。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        initial = cfg.params.get("initial_level", 0)
        # 初始等级必须是合法地形行。
        if not 0 <= initial < env.scene.terrain.cfg.terrain_generator.num_rows:
            raise ValueError("initial_level must be an existing terrain row")
        self.mastery = torch.full((env.num_envs,), initial, dtype=torch.long, device=env.device)
        self.adaptive = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # 本回合是否计成绩
        self.successes = torch.zeros_like(self.mastery)  # 连续成功计数
        self.failures = torch.zeros_like(self.mastery)   # 连续失败计数
        initial_stop = cfg.params.get("initial_stop_level", 0)
        # 0 接近区、1 踏面、2 低阶边缘,之后提升边缘停驻的地形等级。
        if not 0 <= initial_stop <= env.scene.terrain.cfg.terrain_generator.num_rows + 1:
            raise ValueError("initial_stop_level must be in [0, num_rows + 1]")
        self.stop_mastery = torch.full_like(self.mastery, initial_stop)
        self.stop_adaptive = torch.zeros_like(self.adaptive)
        self.stop_successes = torch.zeros_like(self.mastery)
        self.stop_failures = torch.zeros_like(self.mastery)
        env._bpx_stair_curriculum = self

    def __call__(self, env, env_ids, successes_to_promote, failures_to_demote, max_linear_error, max_angular_error,
                 initial_level=0, enabled=True, initial_stop_level=0,
                 stop_successes_to_promote=3, stop_failures_to_demote=2):
        """停车质量不阻塞通过课程;未到停车区的失败不结算停驻技能。"""
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        c = env._bpx_stairs_command
        finished = ids[(env.episode_length_buf[ids] > 0) & (c.route[ids] != 0)]
        duration = c.duration.clamp(min=env.step_dt)
        tracking = ((c.linear_error / duration < max_linear_error)
                    & (c.angular_error / duration < max_angular_error))
        traversed = c.completed & tracking & ~env.termination_manager.terminated
        held = stop_success(c)
        success = traversed & (~c.stop_required | (held & (c.stop_difficulty == 2)))
        active = finished[self.adaptive[finished]]
        max_level = env.scene.terrain.cfg.terrain_generator.num_rows - 1
        self._advance(active, traversed[active], self.mastery, self.successes, self.failures,
                      successes_to_promote, failures_to_demote, max_level, enabled)
        # 平地/回放/尚未到达停车的回合不更新停驻技能。完成停驻后即按
        # 当时的质量计成绩,不因之后行走失败抹去已经完成的停驻能力。
        stop_active = finished[self.stop_adaptive[finished] & c.stop_required[finished]
                               & (c.stop_done[finished] | (c.phase[finished] != 0))]
        self._advance(stop_active, held[stop_active], self.stop_mastery, self.stop_successes, self.stop_failures,
                      stop_successes_to_promote, stop_failures_to_demote, max_level + 2, enabled)
        ascent, descent = c.route[finished] == 1, c.route[finished] == 2
        sampled = finished[c.stop_required[finished]]
        edge = sampled[c.stop_difficulty[sampled] == 2]
        mean = lambda values: values.float().mean() if values.numel() else 0.
        return {"level": self.mastery.float().mean(),
                "stop_level": self.stop_mastery.float().mean(),
                "success_rate": mean(success[finished]), "traversal_success": mean(traversed[finished]),
                "stop_success": mean(held[sampled]), "edge_stop_success": mean(held[edge]),
                "traversal_sample_count": float(len(active)), "stop_sample_count": float(len(stop_active)),
                "ascent_success": mean(success[finished[ascent]]),
                "descent_success": mean(success[finished[descent]])}

    @staticmethod
    def _advance(ids, passed, level, successes, failures, promote, demote, maximum, enabled):
        if promote < 1 or demote < 1:
            raise ValueError("Curriculum streak lengths must be positive")
        if not enabled:
            return
        successes[ids] = torch.where(passed, successes[ids] + 1, 0)
        failures[ids] = torch.where(~passed, failures[ids] + 1, 0)
        up, down = successes[ids] >= promote, failures[ids] >= demote
        level[ids] = (level[ids] + up.long() - down.long()).clamp(0, maximum)
        successes[ids[up | down]] = failures[ids[up | down]] = 0


def contacts(env, sensor_cfg):
    """读取指定 body 的接触传感器,返回 z 向法向力 > 1 N 的布尔接触矩阵。"""
    force = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return force[..., 2] > 1.0


def update_foot_contacts(env, feet_cfg, sensor_cfg):
    """仅由终止监视器调用;所有奖励读取同一份当前步缓存。"""
    c = env._bpx_stairs_command
    contact = contacts(env, sensor_cfg)
    if not hasattr(c, "foot_contacts"):
        c.foot_contacts = StairFootContacts(contact)
    velocity_z = env.scene["robot"].data.body_lin_vel_w[:, feet_cfg.body_ids, 2]
    c.foot_contacts.update(contact, velocity_z)
    return c.foot_contacts


def reward_contacts(env, sensor_cfg):
    """奖励用 20 ms 接触过滤;初次仿真前退化到原始接触。"""
    state = getattr(env._bpx_stairs_command, "foot_contacts", None)
    return contacts(env, sensor_cfg) if state is None else state.filtered


def stopped(env):
    """当前指令范数 < 0.02 的环境视为"受令停止"状态。"""
    return torch.linalg.vector_norm(env.command_manager.get_command("base_velocity"), dim=1) < .02


def critic_terrain_heights(env):
    """Critic 专用 12 点局部地面查询:机身高度减真实地面高度。"""
    robot = env.scene["robot"]
    offsets = torch.cartesian_prod(robot.data.root_pos_w.new_tensor([-.4, 0., .4, .8]),
                                   robot.data.root_pos_w.new_tensor([-.25, 0., .25]))
    yaw = robot.data.heading_w
    cosine, sine = yaw.cos()[:, None], yaw.sin()[:, None]
    rotated = torch.stack((cosine * offsets[:, 0] - sine * offsets[:, 1],
                           sine * offsets[:, 0] + cosine * offsets[:, 1]), dim=-1)
    xy = robot.data.root_pos_w[:, None, :2] + rotated
    return (robot.data.root_pos_w[:, 2, None] - ground_height(env, xy)).clamp(-1., 1.)


def critic_foot_support(env, sensor_cfg):
    """Critic 专用 8 维:四足瞬时承重标记与按 100 N 缩放的竖直力。"""
    force_z = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return torch.cat(((force_z > 1.).float(), (force_z / 100.).clamp(0., 2.)), dim=1)


def critic_stop_state(env):
    """28 维训练状态;时间/漂移/评分只进入 Critic,不进入 Actor。

    route/phase/difficulty/stage 各 3 维;4 个标记;5 个计时/计分量;
    2 维机体系漂移;距停车/出口的 2 个距离;3 个单次奖励锁存。
    """
    c = env._bpx_stairs_command
    robot = env.scene["robot"]
    xy = robot.data.root_pos_w[:, :2]
    delta = xy - c.anchor
    yaw = robot.data.heading_w
    drift = torch.stack((yaw.cos() * delta[:, 0] + yaw.sin() * delta[:, 1],
                         -yaw.sin() * delta[:, 0] + yaw.cos() * delta[:, 1]), dim=1)
    local_x = xy[:, 0] - env.scene.env_origins[:, 0]
    categorical = [torch.nn.functional.one_hot(value, num_classes=3).float()
                   for value in (c.route, c.phase, c.stop_difficulty, c.stop_stage)]
    flags = torch.stack((c.stop_required, c.stop_done, c.goal_reached, c.completed), dim=1).float()
    times = torch.stack((c.hold_elapsed, c.hold_time, c.hold_duration, c.hold_observed, c.hold_good), dim=1)
    distances = torch.stack((c.stop_x - local_x, c.goal_x - local_x), dim=1)
    rewarded = torch.stack((c.traversal_rewarded, c.stop_rewarded, c.completion_rewarded), dim=1).float()
    return torch.cat((*categorical, flags, (times / 3.).clamp(0., 1.),
                      (drift / .1).clamp(-5., 5.), (distances / 10.).clamp(-2., 2.), rewarded), dim=1)


def relative_support_height(env, feet_cfg, sensor_cfg):
    """躯干高度相对"支撑脚所站地面平均高度"的相对高度。

    有接触脚时取接触脚加权(0/1)的平均地面高度;
    无接触(腾空)时退化为四脚地面高度的平均。
    """
    feet = env.scene["robot"].data.body_pos_w[:, feet_cfg.body_ids]
    ground = ground_height(env, feet[..., :2])
    support = contacts(env, sensor_cfg).float()
    mean = (ground * support).sum(dim=1) / support.sum(dim=1).clamp(min=1.)
    mean = torch.where(support.sum(dim=1) > 0, mean, ground.mean(dim=1))
    return env.scene["robot"].data.root_pos_w[:, 2] - mean


def support_height_l2(env, feet_cfg, sensor_cfg, target, tolerance):
    """支撑相对高度对目标值的死区平方误差(容差内不罚)。"""
    error = (relative_support_height(env, feet_cfg, sensor_cfg) - target).abs()
    return (error - tolerance).clamp(min=0).square()


def tilt_deadzone_l2(env, roll_tolerance, pitch_tolerance):
    """横滚/俯仰死区平方惩罚:基于重力向量在机体系的投影分量。"""
    gravity = env.scene["robot"].data.projected_gravity_b
    return ((gravity[:, 1].abs() - roll_tolerance).clamp(min=0).square()
            + (gravity[:, 0].abs() - pitch_tolerance).clamp(min=0).square())


def progress_reward(env):
    """前进速度奖励:直接返回指令生成器维护的带符号前进进度。"""
    return env._bpx_stairs_command.progress


def stop_drift_l2(env, max_distance=.20):
    """受令停止期间相对锚点的漂移平方惩罚(带饱和上限)。"""
    xy = env.scene["robot"].data.root_pos_w[:, :2]
    # actor 只有 0.18 s 历史且没有世界位置观测。这里维持一个"局部保持"
    # 目标,同时用 max_distance 封顶旧累计漂移的代价;
    # 真正严格的 0.10 m 漂移判据保留在课程结算中。
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    error = (xy - env._bpx_stairs_command.anchor).square().sum(dim=1)
    return error.clamp(max=max_distance**2) * stopped(env)


def stop_velocity_l2(env):
    """旧日志兼容函数;新配置分别显式使用线/角速度归一化代价。"""
    robot = env.scene["robot"]
    return (robot.data.root_lin_vel_w.square().sum(dim=1)
            + .2 * robot.data.root_ang_vel_b.square().sum(dim=1)) * stopped(env)


def stop_linear_velocity_l2(env, speed_scale, max_normalized_error):
    """受令停止的三维线速度代价;归一化后封顶,权重由配置声明。"""
    if speed_scale <= 0 or max_normalized_error <= 0:
        raise ValueError("Stop velocity scale and error bound must be positive")
    speed_squared = env.scene["robot"].data.root_lin_vel_w.square().sum(dim=1)
    return (speed_squared / speed_scale**2).clamp(max=max_normalized_error) * stopped(env)


def stop_angular_velocity_l2(env, speed_scale, max_normalized_error):
    """受令停止的三轴角速度代价;不要求跨阶时回到默认关节姿态。"""
    if speed_scale <= 0 or max_normalized_error <= 0:
        raise ValueError("Stop velocity scale and error bound must be positive")
    speed_squared = env.scene["robot"].data.root_ang_vel_b.square().sum(dim=1)
    return (speed_squared / speed_scale**2).clamp(max=max_normalized_error) * stopped(env)


def stair_completion_reward(env):
    """完整任务单次奖励;返回事件/step_dt,抵消 RewardManager 的积分。

    出口仍结束未通过停驻的回合。接近区/踏面停车仅用于技能课程,
    要求停车的最终任务仍须通过边缘停驻,才领取本项联合成功奖励。
    此函数在所有终止项算完后运行,所以同帧物理失败会被排除。
    """
    c = env._bpx_stairs_command
    eligible = (c.completed & (c.route != 0)
                & (~c.stop_required | (stop_success(c) & (c.stop_difficulty == 2)))
                & ~env.termination_manager.terminated)
    event = eligible & ~c.completion_rewarded
    c.completion_rewarded |= eligible
    c.task_success[:] = eligible
    c.metrics["task_success"][:] = eligible.float()
    return event.float() / env.step_dt


def stair_traversal_reward(env):
    """物理驶出的独立单次奖励,不以停车质量为门槛。"""
    c = env._bpx_stairs_command
    eligible = c.completed & (c.route != 0) & ~env.termination_manager.terminated
    event = eligible & ~c.traversal_rewarded
    c.traversal_rewarded |= eligible
    return event.float() / env.step_dt


def stop_completion_reward(env):
    """在严格保持通过且恢复前进指令后立即给一次奖励。"""
    c = env._bpx_stairs_command
    eligible = stop_success(c) & (c.route != 0) & ~env.termination_manager.terminated
    event = eligible & ~c.stop_rewarded
    c.stop_rewarded |= eligible
    return event.float() / env.step_dt


def stop_quality_reward(env, sensor_cfg, linear_scale=.08, angular_scale=.20, drift_scale=.10):
    """停车当下的联合稳定质量,范围 [0, 1];允许非默认的跨阶姿态。"""
    if min(linear_scale, angular_scale, drift_scale) <= 0:
        raise ValueError("Stop quality scales must be positive")
    c = env._bpx_stairs_command
    data = env.scene["robot"].data
    error = (data.root_lin_vel_w.square().sum(dim=1) / linear_scale**2
             + data.root_ang_vel_b.square().sum(dim=1) / angular_scale**2
             + (data.root_pos_w[:, :2] - c.anchor).square().sum(dim=1) / drift_scale**2)
    supported = contacts(env, sensor_cfg).sum(dim=1) >= 3
    active = (c.stop_required & (c.phase == 2) & (c.hold_elapsed >= .3) & stopped(env)
              & supported & (data.projected_gravity_b[:, 2] < -.85)
              & ~env.termination_manager.terminated)
    return torch.exp(-error) * active


def touchdown_impact_l2(env, speed_scale, max_normalized_error):
    """首次落脚前的向下足速;事件代价不随控制步长缩小。"""
    if speed_scale <= 0 or max_normalized_error <= 0:
        raise ValueError("Touchdown velocity scale and error bound must be positive")
    state = env._bpx_stairs_command.foot_contacts
    cost = (state.touchdown_speed / speed_scale).square().clamp(max=max_normalized_error)
    return cost.sum(dim=1) / env.step_dt


def stop_joint_velocity_l2(env):
    """受令停止期间的关节速度平方惩罚(抑制原地关节微动)。"""
    return env.scene["robot"].data.joint_vel.square().sum(dim=1) * stopped(env)


def stop_support_reward(env, sensor_cfg):
    """受令停止且至少 3 脚支撑时的正奖励(鼓励稳定的三/四脚站立)。"""
    return (reward_contacts(env, sensor_cfg).sum(dim=1) >= 3).float() * stopped(env)


def stance_slip_l2(env, feet_cfg, sensor_cfg):
    """支撑脚打滑惩罚:接触中的脚的水平速度平方和。"""
    velocity = env.scene["robot"].data.body_lin_vel_w[:, feet_cfg.body_ids, :2]
    return (velocity.square().sum(dim=-1) * reward_contacts(env, sensor_cfg)).sum(dim=1)


def riser_collision(env, sensor_cfg, ratio):
    """踢面撞击计数:水平冲击力超过竖直法向力 ratio 倍的脚数。

    用于识别"脚直接撞上竖直踢面"而非正常踏上踏面。
    """
    force = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return (torch.linalg.vector_norm(force[..., :2], dim=-1) > ratio * force[..., 2].abs().clamp(min=1.)).float().sum(dim=1)


def clearance_samples(env, feet_cfg, sensor_cfg, lookahead, clearance, toe_radius):
    """返回净空不足量和激活掩码,供奖励与诊断共用。

    lookahead: 向前预看的水平距离,用于估计下一级踏面高度;
    clearance: 脚底应高出目标踏面的裕量;
    toe_radius: 脚(趾)碰撞半径,用于把目标面换算到脚心高度。
    """
    robot = env.scene["robot"]
    feet = robot.data.body_pos_w[:, feet_cfg.body_ids]
    current = ground_height(env, feet[..., :2])       # 当前脚下的踏面高度
    ahead_xy = feet[..., :2].clone()
    ahead_xy[..., 0] += lookahead
    ahead = ground_height(env, ahead_xy)              # 前方 lookahead 处的地面高度
    # 目标抬脚量取"当前踏面"与"前方踏面"的较大者再加裕量:
    # 向上摆腿要为下一级踏面留出净空;下行边缘处则使用当前踏面,
    # 避免奖励脚向下俯冲钻到上一级踢面之下。
    target = torch.maximum(current, ahead) + toe_radius + clearance
    deficit = (target - feet[..., 2]).clamp(0., .20)  # 最多计 0.20 m 不足量
    swing = ~reward_contacts(env, sensor_cfg)         # 与脚滑项共用同一份接触过滤
    foot_moving = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, feet_cfg.body_ids, :2], dim=-1) > .05
    walking = torch.linalg.vector_norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1) > .04
    active = swing & foot_moving & walking[:, None]
    return deficit, active, swing & walking[:, None], walking


def swing_clearance_l2(env, feet_cfg, sensor_cfg, lookahead, clearance, toe_radius):
    """行走中且移动的摆动脚,相对当前/前方较高踏面的净空不足。"""
    deficit, active, _, _ = clearance_samples(env, feet_cfg, sensor_cfg, lookahead, clearance, toe_radius)
    return (deficit.square() * active).sum(dim=1)


def near_ground_drag_samples(env, feet_cfg, sensor_cfg, toe_radius, margin, min_speed):
    """局部踏面附近移动的空中脚;支撑脚已由脚滑项覆盖。"""
    robot = env.scene["robot"]
    feet = robot.data.body_pos_w[:, feet_cfg.body_ids]
    sole_height = feet[..., 2] - toe_radius - ground_height(env, feet[..., :2])
    speed_squared = robot.data.body_lin_vel_w[:, feet_cfg.body_ids, :2].square().sum(dim=-1)
    walking = torch.linalg.vector_norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1) > .04
    dragging = ((sole_height <= margin) & (speed_squared > min_speed**2)
                & ~reward_contacts(env, sensor_cfg) & walking[:, None])
    return speed_squared, dragging


def near_ground_drag_l2(env, feet_cfg, sensor_cfg, toe_radius, margin, min_speed, speed_scale,
                        max_normalized_error):
    """可选近地拖脚代价;默认权重为零,先根据诊断判断是否需要。"""
    if speed_scale <= 0 or max_normalized_error <= 0:
        raise ValueError("Drag velocity scale and error bound must be positive")
    speed_squared, dragging = near_ground_drag_samples(env, feet_cfg, sensor_cfg, toe_radius, margin, min_speed)
    return ((speed_squared / speed_scale**2).clamp(max=max_normalized_error) * dragging).sum(dim=1)


def record_foot_diagnostics(env, feet_cfg, sensor_cfg, lookahead, clearance, toe_radius,
                            drag_margin, drag_min_speed):
    """记录净空激活、近地拖脚、踢面碰撞及失败前所处楼梯段。"""
    c = env._bpx_stairs_command
    deficit, active, swing, walking = clearance_samples(env, feet_cfg, sensor_cfg, lookahead, clearance, toe_radius)
    _, dragging = near_ground_drag_samples(env, feet_cfg, sensor_cfg, toe_radius, drag_margin, drag_min_speed)
    c.walking_time += walking.float() * env.step_dt
    c.swing_feet_time += swing.sum(dim=1) * env.step_dt
    c.clearance_active_time += active.sum(dim=1) * env.step_dt
    c.clearance_deficit_time += (active & (deficit > 0)).sum(dim=1) * env.step_dt
    c.near_ground_drag_time += dragging.sum(dim=1) * env.step_dt
    c.riser_strike_time += (riser_collision(env, sensor_cfg, 4.) > 0).float() * walking * env.step_dt
    edges, _, _, counts, *_ = terrain_table(env)
    terrain = env.scene.terrain
    row, col = terrain.terrain_levels, terrain.terrain_types
    count = counts[row, col]
    first_index = torch.where(c.route == 2, count, 0)
    last_index = torch.where(c.route == 2, 2 * count - 1, count - 1).clamp(min=0)
    stairs = c.route != 0
    first = torch.where(stairs, edges[row, col, first_index], 0.)
    last = torch.where(stairs, edges[row, col, last_index], 1.)
    local_x = env.scene["robot"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    fraction = ((local_x - first) / (last - first).clamp(min=.01)).clamp(0., 1.)
    stage = torch.where(fraction < .25, 0, torch.where(fraction > .75, 2, 1))
    c.last_stair_stage[:] = torch.where(stairs, stage, -1)


def vertical_excess_l2(env, allowance):
    """竖直速度超额平方惩罚:|vz| 超过 allowance 的部分。"""
    return (env.scene["robot"].data.root_lin_vel_w[:, 2].abs() - allowance).clamp(min=0).square()


def low_support_height(env, feet_cfg, sensor_cfg, minimum):
    """支撑相对高度低于最低值的布尔终止条件(判定躯干过度下沉/坍塌)。"""
    return relative_support_height(env, feet_cfg, sensor_cfg) < minimum


def stair_boundary(env, margin):
    """越出 tile 边界的布尔终止条件(x/y 方向各留 margin 裕量)。"""
    local = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    _, _, widths, *_ = terrain_table(env)
    terrain = env.scene.terrain
    width = widths[terrain.terrain_levels, terrain.terrain_types]
    length = terrain.cfg.terrain_generator.size[0]
    return (local[:, 0].abs() > length / 2 - margin) | (local[:, 1].abs() > width / 2 - margin)
