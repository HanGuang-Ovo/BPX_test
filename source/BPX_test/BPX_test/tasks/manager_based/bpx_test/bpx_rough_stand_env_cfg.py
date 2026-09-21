"""Blind rough standing: fully explicit task config; only the shared base contract is inherited."""
import math

from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from . import mdp
from .bpx_base_env_cfg import BPX_POLICY_JOINT_NAMES, BpxBaseEnvCfg, BpxCommonEventCfg
from .bpx_rough_env_cfg import BpxWaveTerrainCfg
from .mdp.rough_stand import (
    reset_rough_stand, reset_rough_stand_joints, rough_stand_last_action,
    relative_stand_height_l2, StandStability, StandingCurriculum,
)


BPX_FOOT_NAMES = ["fl_toe_link", "fr_toe_link", "hl_toe_link", "hr_toe_link"]


@configclass
# 崎岖站立指令：恒为零的占位速度指令（与平地站立一致），保持 48 维观测接口不变
class RoughStandCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
# 崎岖站立事件：平地/低级回放/交接 三层重置 + 小幅推力扰动。
# 公共域随机化（摩擦系数、执行器增益）继承自 BpxCommonEventCfg（基类，非平地任务）。
class RoughStandEventCfg(BpxCommonEventCfg):
    # 三层混合重置：25% 平地 / 25% 历史低级回放 / 25% 交接宽分布 / 25% 当前掌握级，
    # 按当地地面高度放置机身（替换平地的 reset_root_state_uniform）
    reset_base = EventTerm(
        func=reset_rough_stand,
        mode="reset",
        params={
            "flat_probability": .25,
            "replay_probability": .25,
            "handoff_probability": .25,
            "pose_range": {
                "roll": (-.05, .05), "pitch": (-.05, .05), "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-.05, .05), "y": (-.05, .05), "z": (-.03, .03),
                "roll": (-.08, .08), "pitch": (-.08, .08), "yaw": (-.08, .08),
            },
            "handoff_pose_range": {
                "roll": (-.10, .10), "pitch": (-.10, .10), "yaw": (-3.14, 3.14),
            },
            "handoff_velocity_range": {
                "x": (-.08, .08), "y": (-.08, .08), "z": (-.05, .05),
                "roll": (-.15, .15), "pitch": (-.15, .15), "yaw": (-.15, .15),
            },
        },
    )
    # 关节重置 + 合成交接时刻的"上一动作"观测
    reset_robot_joints = EventTerm(
        func=reset_rough_stand_joints,
        mode="reset",
        params={
            "position_range": (-.05, .05), "velocity_range": (-.05, .05),
            "handoff_position_range": (-.15, .15), "handoff_velocity_range": (-.30, .30),
            "action_scale": .5,
        },
    )
    # 周期推力扰动（间隔沿用平地站立的 5–8 s，幅度缩小到 ±0.1）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={
            "velocity_range": {
                "x": (-.1, .1), "y": (-.1, .1), "yaw": (-.1, .1),
            },
        },
    )


@configclass
# 崎岖站立终止条件：显式写出（不继承平地站立），前三项与平地一致，stand_drift 为崎岖专属
class RoughStandTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    # 站立稳定度统计 + 漂移失败终止（供 StandingCurriculum 读取统计）
    stand_drift = DoneTerm(func=StandStability, params={
        'settling_time': .5, 'max_speed': .15, 'max_yaw_rate': .25,
        'max_tilt': .1, 'max_drift': .20, 'failure_drift': .65,
        'contact_force_threshold': 5.,
        'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_toe_link'),
        'symmetry_tolerance': .20,
        'symmetry_joint_cfg': SceneEntityCfg(
            'robot', joint_names=BPX_POLICY_JOINT_NAMES, preserve_order=True,
        ),
        'support_center_tolerance': .05,
        'support_feet_cfg': SceneEntityCfg(
            'robot', body_names=BPX_FOOT_NAMES, preserve_order=True,
        ),
    })


@configclass
# 崎岖站立奖励：显式写出（不继承平地站立）。前 13 项与平地站立逐项一致，
# 其中 base_height_l2 直接使用"相对地面高度"版本（原先在 __post_init__ 中运行时替换），
# 最后两项为崎岖站立专属。之后可独立调整权重或增删奖励项。
class RoughStandRewardsCfg:
    # 存活奖励：不终止即持续给分
    alive = RewTerm(
        func=mdp.is_alive, 
        weight=0.5
    )

    # 零速跟踪奖励：站立指令恒为零，抑制机身平移（std 更小，要求更紧）
    track_zero_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity", 
            "std": math.sqrt(0.04)
        },
    )

    # 零偏航角速度跟踪奖励：抑制原地转动
    track_zero_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity", 
            "std": math.sqrt(0.04)
        },
    )

    # z 方向线速度惩罚：抑制上下颠簸
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0
    )

    # roll/pitch 角速度惩罚：抑制机身晃动
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2, 
        weight=-1.0
    )

    # 姿态平整惩罚：roll/pitch 趋向水平
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2, 
        weight=-10.0
    )

    # 站立高度惩罚：机身相对"当地地面"高度保持 0.40 m（崎岖地形适配）
    base_height_l2 = RewTerm(
        func=relative_stand_height_l2, 
        weight=-5.0, 
        params={
            "target_height": 0.40
        },
    )

    # 足端滑动惩罚：接触期间足端不应平移
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe_link"),
        },
    )

    # 关节速度 / 关节偏离默认角 / 动作变化率 / 关节力矩 正则化惩罚
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.01)
    joint_deviation_l1 = RewTerm(func=mdp.joint_deviation_l1, weight=-0.10)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)

    # 不期望接触惩罚：躯干、髋、大腿、小腿触地
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["torso", ".*_hip_link", ".*_thigh_link", ".*_calf_link"],
            ),
            "threshold": 1.0,
        },
    )

    # —— 以下两项为崎岖站立专属 ——
    # 左右关节镜像对称惩罚（0.10 rad 死区，误差按 0.20 比例计入）
    left_right_joint_symmetry = RewTerm(
        func=mdp.left_right_joint_symmetry_l2,
        weight=-1.0,
        params={
            "tolerance": .10,
            "scale": .20,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=BPX_POLICY_JOINT_NAMES, preserve_order=True,
            ),
        },
    )
    # 质心居中奖励：全身质心到四足支撑几何中心的距离（0.02 m 容差，指数衰减）
    whole_body_com_support = RewTerm(
        func=mdp.whole_body_com_support_exp,
        weight=1.0,
        params={
            "tolerance": .02,
            "std": .05,
            "asset_cfg": SceneEntityCfg("robot"),
            "feet_cfg": SceneEntityCfg(
                "robot", body_names=BPX_FOOT_NAMES, preserve_order=True,
            ),
        },
    )


@configclass
# 崎岖站立课程，用于训练阶段的难度递增（地形、初始状态、指令等）
class RoughStandCurriculumCfg:
    terrain_levels = CurrTerm(func=StandingCurriculum, params={
        'min_duration': 14.4, 'stable_fraction': .90,
        'minimum_four_feet_fraction': .90, 'minimum_symmetry_fraction': .80,
        'minimum_support_center_fraction': .80,
        'successes_to_promote': 2,
    })


@configclass
# 崎岖站立环境配置：只继承公共基类 BpxBaseEnvCfg（机器人 USD、48 维观测、12 维动作等
# 接口契约），不再依赖平地站立任务文件；指令、事件、奖励、终止、课程全部在本文件显式定义。
class BpxRoughStandEnvCfg(BpxBaseEnvCfg):
    commands: RoughStandCommandsCfg = RoughStandCommandsCfg()
    rewards: RoughStandRewardsCfg = RoughStandRewardsCfg()
    terminations: RoughStandTerminationsCfg = RoughStandTerminationsCfg()
    events: RoughStandEventCfg = RoughStandEventCfg()
    curriculum: RoughStandCurriculumCfg = RoughStandCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # 站立回合 15 秒（基类默认 20 秒，与平地站立一致）
        self.episode_length_s = 15.0
        # 8 级波浪起伏地形（与崎岖行走共用 BpxWaveTerrainCfg），重置从最低级开始
        self.scene.terrain.terrain_type = 'generator'
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=20260918, curriculum=True, size=(8., 8.), num_rows=8, num_cols=4,
            border_width=5., color_scheme='height', use_cache=False,
            sub_terrains={'bpx_waves': BpxWaveTerrainCfg()},
        )
        # 交接后首帧的"上一动作"观测用合成动作替换（配合 reset_rough_stand_joints）
        self.observations.policy.actions.func = rough_stand_last_action
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
