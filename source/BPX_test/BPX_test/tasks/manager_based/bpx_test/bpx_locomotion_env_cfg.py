# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BPX 平地速度跟踪（行走）任务配置。

本文件承接重构前 ``bpx_test_env_cfg.py`` 的任务专属部分。参数保持原值，
因此旧的 locomotion checkpoint 与导出的 policy 不会因为此次文件拆分而改变接口。
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
class BpxLocomotionCommandsCfg:
    """随机采样机身前后、横向和偏航速度。"""

    base_velocity = mdp.StratifiedVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        # 10% 静止、25% 纯原地旋转、65% 三轴混合运动。
        rel_standing_envs=0.10,
        rel_pure_rotation_envs=0.25,
        pure_rotation_speed_range=(0.2, 1.0),
        heading_command=False,
        debug_vis=True,
        ranges=mdp.StratifiedVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class BpxLocomotionEventCfg(BpxCommonEventCfg):
    """行走任务的直立重置和周期推力。"""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.2, 0.2), "velocity_range": (-0.1, 0.1)},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class BpxLocomotionRewardsCfg:
    """速度跟踪、步态塑形和通用正则化奖励。"""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_deviation_l1 = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05)
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
class BpxLocomotionTerminationsCfg:
    """行走时跌倒或躯干碰地即结束 episode。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class BpxLocomotionEnvCfg(BpxBaseEnvCfg):
    """可注册并训练的 BPX 行走环境。"""

    commands: BpxLocomotionCommandsCfg = BpxLocomotionCommandsCfg()
    rewards: BpxLocomotionRewardsCfg = BpxLocomotionRewardsCfg()
    terminations: BpxLocomotionTerminationsCfg = BpxLocomotionTerminationsCfg()
    events: BpxLocomotionEventCfg = BpxLocomotionEventCfg()
