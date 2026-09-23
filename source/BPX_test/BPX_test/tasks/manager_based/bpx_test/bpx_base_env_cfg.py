# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BPX 多策略训练环境共享配置。

本模块只保存所有底层策略都必须遵守的“机器人接口契约”：

* 相同的机器人 USD、执行器参数和默认站姿；
* 相同的策略关节顺序；
* 公共 48 维观测布局（Init Actor 去掉线速度后为 45 维）和相同的 12 维动作含义；
* 相同的平坦场景、仿真步长和公共域随机化。

行走、站立、起身各自的 command、reset、reward 和 termination 不放在这里，
而是在对应的任务文件中定义。这样修改某个任务的训练目标时，不会无意中改变
其他策略的任务定义。
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


##
# 机器人和策略接口
##

# 由 URDF 转换得到的 BPX USD。使用相对当前包的位置计算绝对路径，避免依赖启动目录。
BPX_USD_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "BPX_structure", "usd", "bpx.usd")
)

# 这是训练、导出、MuJoCo 部署共同使用的唯一关节顺序。
# 不要为了让列表看起来更符合单腿顺序而重排；已有策略张量的每一维都依赖此顺序。
BPX_POLICY_JOINT_NAMES = [
    "fl_hip_roll_joint",
    "fr_hip_roll_joint",
    "hl_hip_roll_joint",
    "hr_hip_roll_joint",
    "fl_hip_pitch_joint",
    "fr_hip_pitch_joint",
    "hl_hip_pitch_joint",
    "hr_hip_pitch_joint",
    "fl_knee_joint",
    "fr_knee_joint",
    "hl_knee_joint",
    "hr_knee_joint",
]

# 所有策略共同使用的机器人定义和默认站姿。
BPX_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=BPX_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    # 机身高度约 0.40 m；动作值 0 对应下面这组默认关节角。
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": 0.7,
            ".*_knee_joint": -1.4,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit_sim=30.0,
            velocity_limit_sim=20.0,
            stiffness=40.0,
            damping=1.0,
        ),
    },
)


##
# 公共场景
##


@configclass
class BpxSceneCfg(InteractiveSceneCfg):
    """所有 BPX 平地任务共同使用的场景。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = BPX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 接触传感器既服务于奖励，也用于检测非法碰撞和足端滑动。
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


##
# 公共动作和观测
##


@configclass
class BpxActionsCfg:
    """共同的 12 维关节位置动作。

    实际目标角 = 默认关节角 + 0.5 * policy_action。
    supervisor 会在不同策略的 action 之间插值，因此所有策略必须保持相同的顺序、
    默认角和缩放。
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=BPX_POLICY_JOINT_NAMES,
        preserve_order=True,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
# 观测
class BpxObservationsCfg:
    """共同的 48 维 actor/critic 观测。"""
    """
    obs[0:3]    = base_lin_vel
    obs[3:6]    = base_ang_vel
    obs[6:9]    = projected_gravity
    obs[9:12]   = velocity_commands
    obs[12:24]  = joint_pos
    obs[24:36]  = joint_vel
    obs[36:48]  = actions
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """观测项按声明顺序拼接；顺序就是导出模型的输入契约。"""
    # 机身坐标系下的基座线速度      三维 [vx, vy, vz]
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, 
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
    # 机身坐标系下的基座角速度      三维 [wx, wy, wz]
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            noise=Unoise(n_min=-0.2, n_max=0.2)
        )
    # 机身坐标系下的重力方向        三维重力投影 [gx, gy, gz]
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, 
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
    # 机身坐标系下的速度命令        三维 [vx, vy, vz]
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )
    # 关节相对位置                十二维 [q1, q2, ..., q12]
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=BPX_POLICY_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
    # 关节相对速度                十二维 [dq1, dq2, ..., dq12]
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=BPX_POLICY_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
    # 上一时刻关节的动作            十二维 [a1, a2, ..., a12]
        actions = ObsTerm(
            func=mdp.last_action
        )
        # 表示训练时会启用上述观测噪声，并将所有观测项拼接成一个 48 维向量
        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
# 公共事件，如物理材质随机化、执行器增益随机化等
class BpxCommonEventCfg:
    """与具体任务无关、在仿真启动时执行的域随机化，4096个环境就有4096个随机化实例。"""

    # 物理材质随机化，主要是摩擦系数和恢复系数
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.4),                    # 静摩擦系数
            "dynamic_friction_range": (0.4, 1.0),                   # 动摩擦系数
            "restitution_range": (0.0, 0.0),                        # 恢复系数
            "num_buckets": 64,
        },
    )
    # 执行器增益随机化，主要是刚度和阻尼
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


##
# 公共环境骨架
##


@configclass
class BpxBaseEnvCfg(ManagerBasedRLEnvCfg):
    """只提供公共字段的环境基类，不应直接注册为 Gym 任务。"""

    scene: BpxSceneCfg = BpxSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: BpxObservationsCfg = BpxObservationsCfg()
    actions: BpxActionsCfg = BpxActionsCfg()
    events: BpxCommonEventCfg = BpxCommonEventCfg()

    def __post_init__(self) -> None:
        # 200 Hz 物理仿真，每 4 个物理步更新一次动作，即策略频率为 50 Hz。
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
