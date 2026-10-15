# -*- coding: utf-8 -*-
"""楼梯 Move/Hold 任务共用的场景、奖励和课程基线。

具体任务配置位于 bpx_stairs_move_env_cfg 与 bpx_stairs_hold_env_cfg；
场景、动作和观测契约由 bpx_stairs_common_cfg 统一定义。
"""

import isaaclab.envs.mdp as core
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .mdp import stairs
from .bpx_stairs_common_cfg import (
    FEET, BpxStairTerrainCfg, BpxStairsSceneCfg, BpxStairsActionsCfg,
    BpxStairsObservationsCfg, feet_asset, feet_sensor,
)


@configclass
class BpxStairsCommandsCfg:
    """速度指令:楼梯上慢速前进 + 空间触发停车;平地上全向指令回放。"""
    base_velocity = stairs.StairVelocityCommandCfg(
        forward_speed_range=(.10, .40), lateral_range=(-.02, .02), yaw_range=(-.05, .05),
        speed_resampling_range=(2., 5.), slew_rate=(.5, .4, .8),
        start_wait_range=(0., .5), stop_probability=.75, hold_time_range=(1., 3.),
        exit_distance=.85, lateral_gain=.4, heading_gain=.8,
        flat_x_range=(-.7, .7), flat_y_range=(-.25, .25), flat_yaw_range=(-.7, .7),
        flat_standing_probability=.2, resampling_time_range=(1000., 1000.), debug_vis=False,
    )
    # 各参数含义详见 StairVelocityCommandCfg 的字段注释。


@configclass
class BpxStairsEventsCfg:
    """事件:域随机化(摩擦/执行器增益)与根部/关节重置。"""
    physics_material = EventTerm(
        # 启动时对整机随机化刚体材质:静/动摩擦 0.5~1.2,分 64 个桶。
        func=core.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (.5, 1.2), "dynamic_friction_range": (.5, 1.2),
                "restitution_range": (0., 0.), "num_buckets": 64, "make_consistent": True},
    )
    actuator_gains = EventTerm(
        # 每次重置随机缩放 PD 增益 ±15%,覆盖真实电机刚度/阻尼的不确定性。
        func=core.randomize_actuator_gains, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "stiffness_distribution_params": (.85, 1.15),
                "damping_distribution_params": (.85, 1.15), "operation": "scale", "distribution": "uniform"},
    )
    reset_base = EventTerm(
        # 楼梯专用根部重置:25% 平地列;楼梯列中 25% 低难度回放;
        # 出生于首级边缘前 0.60~1.50 m、站高 0.40 m。
        func=stairs.reset_stair_root, mode="reset",
        params={"flat_probability": .25, "replay_probability": .25,
                "approach_range": (.60, 1.50), "height": .40},
    )
    reset_joints = EventTerm(
        # 关节重置:在默认姿态上加 ±0.04 rad 偏移、±0.15 rad/s 初速度。
        func=core.reset_joints_by_offset, mode="reset",
        params={"position_range": (-.04, .04), "velocity_range": (-.15, .15)},
    )


@configclass
class BpxStairsRewardsCfg:
    """显式楼梯奖励;通常为 step_dt 积分的速率,事件项在函数中除以 dt。

    不设默认姿态/对称性/平地高度奖励:不同的支撑高度必须都是可行的,
    只惩罚过大的竖直速度与过度倾斜。
    """

    # ---- 任务跟踪与姿态 shaping ----
    track_linear = RewTerm(func=core.track_lin_vel_xy_exp, weight=2.,        # 线速度跟踪(指数核)
                          params={"command_name": "base_velocity", "std": .20})
    track_yaw = RewTerm(func=core.track_ang_vel_z_exp, weight=.4,            # 偏航角速度跟踪
                       params={"command_name": "base_velocity", "std": .25})
    forward_progress = RewTerm(func=stairs.progress_reward, weight=.5)       # 前进进度(带符号)
    traversal_bonus = RewTerm(func=stairs.stair_traversal_reward, weight=2.)  # 独立物理驶出,单次 +2
    completion_bonus = RewTerm(func=stairs.stair_completion_reward, weight=2.)  # 最终联合验收,单次 +2
    support_height = RewTerm(func=stairs.support_height_l2, weight=-6.,      # 支撑高度死区保持 ~0.40 m
                             params={"feet_cfg": feet_asset(), "sensor_cfg": feet_sensor(),
                                     "target": .40, "tolerance": .07})
    moderate_tilt = RewTerm(func=stairs.tilt_deadzone_l2, weight=-.8,        # 横滚 ±0.15 / 俯仰 ±0.45 死区
                            params={"roll_tolerance": .15, "pitch_tolerance": .45})
    excessive_vertical_speed = RewTerm(func=stairs.vertical_excess_l2, weight=-.3, params={"allowance": .45})  # |vz| > 0.45 惩罚
    angular_motion = RewTerm(func=core.ang_vel_xy_l2, weight=-.04)           # 横向角速度正则
    # ---- 脚-楼梯交互 shaping ----
    feet_slip = RewTerm(func=stairs.stance_slip_l2, weight=-.4,              # 支撑脚打滑
                       params={"feet_cfg": feet_asset(), "sensor_cfg": feet_sensor()})
    riser_strike = RewTerm(func=stairs.riser_collision, weight=-.25,         # 撞击竖直踢面(水平力 > 4x 法向力)
                          params={"sensor_cfg": feet_sensor(), "ratio": 4.})
    swing_clearance = RewTerm(func=stairs.swing_clearance_l2, weight=-2.,    # 摆动脚抬脚净空不足
                             params={"feet_cfg": feet_asset(), "sensor_cfg": feet_sensor(),
                                     "lookahead": .12, "clearance": .04, "toe_radius": .039})
    touchdown_impact = RewTerm(
        func=stairs.touchdown_impact_l2, weight=-.02,
        params={"speed_scale": .50, "max_normalized_error": 9.},
    )  # 单次落脚按上一帧向下足速结算;0.50 m/s 每脚扣 0.02
    near_ground_drag = RewTerm(
        func=stairs.near_ground_drag_l2, weight=0.,  # 先记录诊断;确认需要后再显式启用
        params={"feet_cfg": feet_asset(), "sensor_cfg": feet_sensor(), "toe_radius": .039,
                "margin": .015, "min_speed": .05, "speed_scale": .10, "max_normalized_error": 9.},
    )
    # ---- 受令停止(hold)专项 ----
    hold_quality = RewTerm(func=stairs.stop_quality_reward, weight=.5,
                           params={"sensor_cfg": feet_sensor()})  # 联合稳定质量,每秒最多 +0.5
    hold_success_bonus = RewTerm(func=stairs.stop_completion_reward, weight=1.)  # 严格保持并重启,单次 +1
    hold_drift = RewTerm(func=stairs.stop_drift_l2, weight=-8., params={"max_distance": .20})  # 停车漂移(带饱和)
    hold_linear_velocity = RewTerm(
        func=stairs.stop_linear_velocity_l2, weight=-.25,
        params={"speed_scale": .08, "max_normalized_error": 9.},
    )  # 3D 线速度,0.08 m/s 扣 0.25/s,最多扣 2.25/s
    hold_angular_velocity = RewTerm(
        func=stairs.stop_angular_velocity_l2, weight=-.25,
        params={"speed_scale": .20, "max_normalized_error": 9.},
    )  # 3 轴角速度,0.20 rad/s 扣 0.25/s,最多扣 2.25/s
    hold_joint_motion = RewTerm(func=stairs.stop_joint_velocity_l2, weight=-.002)  # 停车时的关节速度
    hold_support = RewTerm(func=stairs.stop_support_reward, weight=.3, params={"sensor_cfg": feet_sensor()})  # 停车 >=3 脚支撑奖励
    # ---- 常规正则与安全 ----
    torque = RewTerm(func=core.joint_torques_l2, weight=-1.e-5)              # 关节力矩正则
    joint_acceleration = RewTerm(func=core.joint_acc_l2, weight=-1.e-7)      # 关节加速度正则
    action_change = RewTerm(func=core.action_rate_l2, weight=-.01)           # 动作变化率正则
    joint_limits = RewTerm(func=core.joint_pos_limits, weight=-1.)           # 关节限位惩罚
    non_foot_contact = RewTerm(                                              # 非脚部位(躯干/髋/大腿/小腿)触地惩罚
        func=core.undesired_contacts, weight=-.75,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso", ".*_hip_link", ".*_thigh_link", ".*_calf_link"]),
                "threshold": 5.},
    )
    failure = RewTerm(func=core.is_terminated, weight=-5.)                   # 故障终止的一次性惩罚


@configclass
class BpxStairsTerminationsCfg:
    # 监视器必须排在第一位:保证 reset 之前终局统计已经写好。
    completed_exit = DoneTerm(
        func=stairs.StairEpisodeMonitor, time_out=True,
        params={"feet_cfg": feet_asset(), "sensor_cfg": feet_sensor(), "final_hold": .20,
                "flat_episode_s": 20., "exit_speed_limit": .65,
                "diagnostic_clearance": {"lookahead": .12, "clearance": .04, "toe_radius": .039,
                                         "drag_margin": .015, "drag_min_speed": .05}},
    )
    time_out = DoneTerm(func=core.time_out, time_out=True)                   # 到达回合时长上限
    torso_contact = DoneTerm(func=core.illegal_contact,                      # 躯干触地(摔倒)
                             params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso"), "threshold": 5.})
    fallen_orientation = DoneTerm(func=core.bad_orientation, params={"limit_angle": .95})  # 姿态过度倾斜
    low_body = DoneTerm(func=stairs.low_support_height,                      # 躯干过度下沉(支撑高度 < 0.16 m)
                        params={"feet_cfg": feet_asset(), "sensor_cfg": feet_sensor(), "minimum": .16})
    tile_boundary = DoneTerm(func=stairs.stair_boundary, params={"margin": .15})  # 越出 tile 边界


@configclass
class BpxStairsCurriculumCfg:
    """通过 0-7 级与停驻 0-9 级独立进阶;最终仍验收边缘停驻+完整通过。"""
    terrain_mastery = CurrTerm(
        func=stairs.StairMasteryCurriculum,
        params={"successes_to_promote": 3, "failures_to_demote": 2,
                "max_linear_error": .20, "max_angular_error": .30,  # 成功判定的平均跟踪误差阈值
                "initial_level": 0, "initial_stop_level": 0, "enabled": True,
                "stop_successes_to_promote": 3, "stop_failures_to_demote": 2},
    )


@configclass
class BpxStairsBaseEnvCfg(ManagerBasedRLEnvCfg):
    # 空间停车调度及其掌握度统计从第 0 个回合就开始生效,
    # 因此不做初始回合长度随机化。
    randomize_initial_episode_length: bool = False
    scene: BpxStairsSceneCfg = BpxStairsSceneCfg(num_envs=2048, env_spacing=3.)
    observations: BpxStairsObservationsCfg = BpxStairsObservationsCfg()
    actions: BpxStairsActionsCfg = BpxStairsActionsCfg()
    commands: BpxStairsCommandsCfg = BpxStairsCommandsCfg()
    events: BpxStairsEventsCfg = BpxStairsEventsCfg()
    rewards: BpxStairsRewardsCfg = BpxStairsRewardsCfg()
    terminations: BpxStairsTerminationsCfg = BpxStairsTerminationsCfg()
    curriculum: BpxStairsCurriculumCfg = BpxStairsCurriculumCfg()

    def __post_init__(self):
        # 控制频率 50 Hz:物理步长 5 ms x decimation 4;
        # 回合上限 80 s;渲染每 4 个物理步一次。
        self.decimation = 4
        self.episode_length_s = 80.
        self.sim.dt = .005
        self.sim.render_interval = 4
        self.sim.physics_material = self.scene.terrain.physics_material  # 仿真材质与地形一致
        self.scene.contact_forces.update_period = .005  # 接触传感器每个物理步更新
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15  # 提高接触补丁上限(多脚+台阶密集接触)
