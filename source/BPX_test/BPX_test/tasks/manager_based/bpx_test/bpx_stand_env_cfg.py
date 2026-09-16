# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BPX 静止站立策略训练配置。

站立策略仍保留 3 维速度 command 输入，但训练中 command 始终为零。这样导出的
policy 与 locomotion policy 具有完全相同的观测/动作接口，可由 supervisor 直接切换。
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .bpx_base_env_cfg import BpxBaseEnvCfg, BpxCommonEventCfg


@configclass
class BpxStandCommandsCfg:
    """恒为零的占位速度指令，保持观测维度和名称不变。"""

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
class BpxStandEventCfg(BpxCommonEventCfg):
    """从默认站姿附近开始，并用小扰动训练抗干扰站立。"""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.10, 0.10),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-0.15, 0.15),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.10, 0.10), "velocity_range": (-0.05, 0.05)},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={
            "velocity_range": {
                "x": (-0.30, 0.30),
                "y": (-0.30, 0.30),
                "yaw": (-0.20, 0.20),
            }
        },
    )


@configclass
class BpxStandRewardsCfg:
    """优先抑制零指令下的机身运动、足端滑动和关节抖动。"""

    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    track_zero_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.04)},
    )
    track_zero_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.04)},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-5.0, params={"target_height": 0.40})
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe_link"),
        },
    )
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.01)
    joint_deviation_l1 = RewTerm(func=mdp.joint_deviation_l1, weight=-0.10)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
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


@configclass
class BpxStandTerminationsCfg:
    """站立失败后及时重置，集中采样直立状态附近的数据。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class BpxStandEnvCfg(BpxBaseEnvCfg):
    """可注册并训练的 BPX 静止站立环境。"""

    commands: BpxStandCommandsCfg = BpxStandCommandsCfg()
    rewards: BpxStandRewardsCfg = BpxStandRewardsCfg()
    terminations: BpxStandTerminationsCfg = BpxStandTerminationsCfg()
    events: BpxStandEventCfg = BpxStandEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 15.0
