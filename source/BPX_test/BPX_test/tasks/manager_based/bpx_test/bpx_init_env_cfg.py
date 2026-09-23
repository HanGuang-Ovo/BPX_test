# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BPX 趴卧起身（Init）策略训练配置。

机器人从腹部朝地、机身低位、小腿自然贴地且足端朝前的趴卧状态开始，目标是抬升到
0.40 m 左右的四足稳定默认站姿。Init 使用单帧非对称观测：Actor 45 维，不提供
基座线速度；Critic 48 维，额外读取三轴线速度真值。12 维动作、关节顺序和
动作缩放仍与其他策略一致，导出后由 sim2sim Supervisor 切换。
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .bpx_base_env_cfg import BpxBaseEnvCfg, BpxCommonEventCfg, BpxObservationsCfg


MINIMUM_FRONT_FEET_WIDTH = 0.24
INIT_START_HEIGHT = 0.13
INIT_MAXIMUM_UPWARD_VELOCITY = 0.20
INIT_MAXIMUM_JOINT_TORQUE = 28.0
INIT_TORQUE_PENALTY_INITIAL_WEIGHT = -0.1
INIT_TORQUE_PENALTY_FINAL_WEIGHT = -1.0
INIT_TORQUE_CURRICULUM_START_SUCCESS_RATE = 0.70
INIT_TORQUE_CURRICULUM_FULL_SUCCESS_RATE = 0.90
INIT_TORQUE_CURRICULUM_MINIMUM_EPISODES = 4_096
INIT_TORQUE_CURRICULUM_WINDOW_SIZE = 8_192


@configclass
class BpxInitObservationsCfg:
    """单帧非对称观测：Actor 45 维，Critic 48 维。

    Actor 顺序：角速度(3)、重力投影(3)、零速度指令(3)、关节位置(12)、
    关节速度(12)、上一步动作(12)。Critic 在最前面增加机身系线速度真值(3)。
    其余观测沿用公共配置的噪声；不增加历史帧。
    """

    @configclass
    class PolicyCfg(BpxObservationsCfg.PolicyCfg):
        # 删除观测项，不用零填充，避免 Actor 依赖仿真线速度。
        base_lin_vel = None

    @configclass
    class CriticCfg(BpxObservationsCfg.PolicyCfg):
        # Critic 独有的三轴线速度 [vx, vy, vz]，不加观测噪声。
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class BpxInitCommandsCfg:
    """Init 期间不接收运动指令，但保留 command 的三个观测维度。"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 4.0),
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
class BpxInitEventCfg(BpxCommonEventCfg):
    """构造腹部朝地、小腿贴地且足端朝前的趴卧初始状态。

    BPX 默认根高度是 0.40 m，因此 z 偏移约 -0.27 m 后，机身保持低位。髋俯仰
    约 0.98 rad、膝约 -2.62 rad，使髋俯仰与膝角之和接近 -pi/2：小腿近似
    水平、足端沿机身前向伸出。少量随机化用于防止记忆唯一动作序列。
    """

    # 这些 reset event 的随机化范围应尽量小，避免机器人在起身前就翻倒或撞到地面。
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.268, -0.266),
                "roll": (-0.02, 0.02),
                "pitch": (-0.02, 0.02),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.02, 0.02),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )

    # hip横滚关节角度随机化
    reset_hip_roll_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "velocity_range": (-0.05, 0.05),
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_roll_joint"),
        },
    )
    # hip俯仰关节角度随机化
    reset_hip_pitch_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            # 默认值 0.70 + offset，得到约 [0.96, 1.00] rad。
            "position_range": (0.26, 0.30),
            "velocity_range": (-0.05, 0.05),
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_pitch_joint"),
        },
    )
    # 膝关节角度随机化
    reset_knee_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            # 默认值 -1.40 + offset，得到约 [-2.64, -2.60] rad。
            "position_range": (-1.24, -1.20),
            "velocity_range": (-0.05, 0.05),
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_joint"),
        },
    )


@configclass
class BpxInitRewardsCfg:
    """用状态反馈塑造抬升、四足支撑、正常站姿和稳定交接。"""


    """
    奖励项
    """
    # 机身抬升奖励
    # 机身高度从趴卧的 0.13 m 提升到 0.40 m，奖励逐渐增加。
    height_progress = RewTerm(
        func=mdp.upright_height_progress,
        weight=4.0,
        params={
            "start_height": INIT_START_HEIGHT, 
            "target_height": 0.40, 
            "orientation_std": 0.35
        },
    )

    # 关节姿态奖励
    # 机身抬得越高，四条腿恢复默认站姿的收益越大，直接压制只伸直两条腿的局部最优。
    joint_posture = RewTerm(
        func=mdp.joint_posture_progress_exp,
        weight=2.0,
        params={
            "start_height": INIT_START_HEIGHT,
            "target_height": 0.40,
            "std": 0.50,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    # 前足宽度使用机体坐标系计算，避免世界系 yaw 影响。奖励只规定最低宽度，不锁死
    # 起身轨迹；有符号宽度还能防止两只前脚交叉后利用绝对距离重新得分。
    front_feet_width = RewTerm(
        func=mdp.front_feet_width_progress,
        weight=2.0,
        params={
            "start_height": 0.14,
            "target_height": 0.32,
            "minimum_width": MINIMUM_FRONT_FEET_WIDTH,
            "front_feet_cfg": SceneEntityCfg(
                "robot",
                body_names=["fl_toe_link", "fr_toe_link"],
                preserve_order=True,
            ),
        },
    )

    # 全关节姿态项会稀释两个前髋横滚误差，因此在机身开始抬升后单独约束四个髋横滚。
    hip_roll_posture = RewTerm(
        func=mdp.joint_posture_progress_exp,
        weight=1.0,
        params={
            "start_height": 0.18,
            "target_height": 0.36,
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_roll_joint"),
        },
    )

    # 起始趴姿不强制足端接触；随实际抬升进度逐渐提高四足共同支撑的收益。
    feet_support = RewTerm(
        func=mdp.feet_support_progress,
        weight=2.0,
        params={
            "start_height": INIT_START_HEIGHT,
            "target_height": 0.40,
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
        },
    )

    # 起身任务的“成功判定”奖励
    # 成功奖励随时可以激活，但必须同时满足高度、姿态、四足接触和前足宽度。
    recovered = RewTerm(
        func=mdp.recovered_posture_with_contact,
        weight=10.0,
        params={
            "minimum_height": 0.36,
            "maximum_tilt": 0.1,       # 0.1 rad ≈ 5.7°，避免机身明显倾斜。
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
            "minimum_front_width": MINIMUM_FRONT_FEET_WIDTH,
            "front_feet_cfg": SceneEntityCfg(
                "robot",
                body_names=["fl_toe_link", "fr_toe_link"],
                preserve_order=True,
            ),
        },
    )

    # 四足完成起身后立即奖励低机身速度，不在起身过程中抑制必要运动。
    recovered_stability = RewTerm(
        func=mdp.recovered_stability_with_contact,
        weight=5.0,
        params={
            "minimum_height": 0.36,
            "maximum_tilt": 0.1,       # 0.1 rad ≈ 5.7°，避免机身明显倾斜。
            "linear_velocity_std": 0.30,
            "angular_velocity_std": 0.45,
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
            "minimum_front_width": MINIMUM_FRONT_FEET_WIDTH,
            "front_feet_cfg": SceneEntityCfg(
                "robot",
                body_names=["fl_toe_link", "fr_toe_link"],
                preserve_order=True,
            ),
        },
    )

    """
    惩罚项
    """
    # 趴姿初期允许足端寻找支点；高度超过 0.18 m 后逐步抑制贴地横向滑动。
    feet_slide = RewTerm(
        func=mdp.feet_slide_progress,
        weight=-0.15,
        params={
            "start_height": 0.18,
            "target_height": 0.36,
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["fl_toe_link", "fr_toe_link", "hl_toe_link", "hr_toe_link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["fl_toe_link", "fr_toe_link", "hl_toe_link", "hr_toe_link"],
                preserve_order=True,
            ),
        },
    )

    # 后仰约 4°、前倾约 5.7°时开始惩罚，前后超界量使用相同惩罚尺度。
    asymmetric_pitch_band_l2 = RewTerm(
        func=mdp.asymmetric_pitch_band_l2,
        weight=-10.0,
        params={
            "maximum_backward_pitch": 0.07,
            "maximum_forward_pitch": 0.10,
        },
    )

    # 低位允许调整支撑；随高度从 0.18 m 升至 0.36 m，逐步抑制世界系 x、y 方向漂移。
    base_lin_vel_xy_progress_l2 = RewTerm(
        func=mdp.base_lin_vel_xy_progress_l2,
        weight=-0.5,
        params={"start_height": 0.18, "target_height": 0.36},
    )

    # 只惩罚超过上限的世界系向上速度，限制快速弹起而不强迫策略跟踪固定速度。
    base_upward_velocity_limit_l2 = RewTerm(
        func=mdp.base_upward_velocity_limit_l2,
        weight=-2.0,
        params={"maximum_velocity": INIT_MAXIMUM_UPWARD_VELOCITY},
    )

    # “先学会站起”阶段保留轻量正则；动作变化率仍跳过复位后的首次突变，避免平滑惩罚压制探索。
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0
    )
    # 直接抑制关节高速运动，降低起身过程中腿部动作的激进程度。
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001
    )
    # 抑制关节速度突变，使起身动作更平滑；该量级与 locomotion 任务保持一致。
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-4.0e-7
    )
    # 抑制动作突变，使起身动作更平滑；该量级与 locomotion 任务保持一致。
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_after_time,
        weight=-0.005,
        params={"start_time_s": 0.10},
    )
    # 直接抑制关节力矩，降低起身过程中腿部动作的激进程度。
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-5
    )
    # 超过 28 N·m 后按超限量平方施加大惩罚，给 30 N·m 仿真硬上限预留保护裕量。
    joint_torque_limit_l2 = RewTerm(
        func=mdp.joint_torque_limit_l2,
        weight=INIT_TORQUE_PENALTY_INITIAL_WEIGHT,
        params={"maximum_torque": INIT_MAXIMUM_JOINT_TORQUE},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0
    )


@configclass
class BpxInitTerminationsCfg:
    """允许起身期间所有身体部位接触地面，只终止超时和关节越界。"""

    # 当前 contact_forces 是各刚体的净接触力，不能可靠区分腿间自碰撞和允许的腿部触地。
    # 前腿并拢先由足端宽度直接约束；以后若增加过滤接触传感器，再单独加入自碰撞项。
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True
    )
    joint_pos_out_of_limit = DoneTerm(
        func=mdp.joint_pos_out_of_limit
    )


@configclass
class BpxInitCurriculumCfg:
    """先学习可靠起身，再逐步加强关节力矩超限约束。"""

    joint_torque_limit = CurrTerm(
        func=mdp.modify_reward_weight_by_success_rate,
        params={
            "term_name": "joint_torque_limit_l2",
            "success_term_name": "recovered",
            "initial_weight": INIT_TORQUE_PENALTY_INITIAL_WEIGHT,
            "final_weight": INIT_TORQUE_PENALTY_FINAL_WEIGHT,
            "start_success_rate": INIT_TORQUE_CURRICULUM_START_SUCCESS_RATE,
            "full_success_rate": INIT_TORQUE_CURRICULUM_FULL_SUCCESS_RATE,
            "minimum_episodes": INIT_TORQUE_CURRICULUM_MINIMUM_EPISODES,
            "window_size": INIT_TORQUE_CURRICULUM_WINDOW_SIZE,
        },
    )


@configclass
class BpxInitEnvCfg(BpxBaseEnvCfg):
    """可注册并训练的 BPX 趴卧起身环境。"""

    commands: BpxInitCommandsCfg = BpxInitCommandsCfg()
    observations: BpxInitObservationsCfg = BpxInitObservationsCfg()
    rewards: BpxInitRewardsCfg = BpxInitRewardsCfg()
    terminations: BpxInitTerminationsCfg = BpxInitTerminationsCfg()
    events: BpxInitEventCfg = BpxInitEventCfg()
    curriculum: BpxInitCurriculumCfg = BpxInitCurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # 固定 4 秒回合用于学习起身后的稳定保持，不再规定完成时刻。
        self.episode_length_s = 4.0
