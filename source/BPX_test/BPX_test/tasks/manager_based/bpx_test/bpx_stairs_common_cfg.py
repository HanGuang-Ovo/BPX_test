# -*- coding: utf-8 -*-
"""楼梯任务共享的地形、机器人、450/528 维观测与动作契约。"""

import isaaclab.envs.mdp as core
import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .bpx_base_env_cfg import BPX_CFG, BPX_POLICY_JOINT_NAMES
from .mdp import stairs
from .stair_geometry import stair_terrain


# 四个脚(趾)链接名,顺序固定 —— 观测/奖励中按此顺序索引脚部数据。
FEET = ["fl_toe_link", "fr_toe_link", "hl_toe_link", "hr_toe_link"]


def feet_asset():
    """指向四个脚 body 的场景实体配置(preserve_order 保证顺序)。"""
    return SceneEntityCfg("robot", body_names=FEET, preserve_order=True)


def feet_sensor():
    """指向四个脚接触力传感器的场景实体配置(与 FEET 同序)。"""
    return SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)


@configclass
class BpxStairTerrainCfg(SubTerrainBaseCfg):
    """楼梯子地形配置;几何由 stair_geometry.stair_terrain 回调生成。"""
    function = stair_terrain
    flat: bool = False                                   # True 时生成平地 tile(回放/平地训练)
    levels: int = 8                                      # 课程难度等级数(映射到地形行)
    initial_height_range: tuple[float, float] = (.03, .05)  # 最低等级的台阶高度范围(接近平地)
    height_range: tuple[float, float] = (.05, .15)       # 最高等级的台阶高度范围(完整楼梯)
    depth_range: tuple[float, float] = (.25, .40)        # 踏面进深采样范围
    step_count_range: tuple[int, int] = (4, 8)           # 单侧台阶数采样范围
    width_range: tuple[float, float] = (1.2, 2.0)        # 楼梯有效宽度采样范围
    landing_length: float = 3.2                          # 顶部平台长度
    approach_margin: float = 1.55                        # 平地接近段最小长度(两端各留一份)


@configclass
class BpxStairsSceneCfg(InteractiveSceneCfg):
    """场景配置:课程地形生成器 + 延迟 PD 执行器的机器人 + 接触传感器。"""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", collision_group=-1,
        max_init_terrain_level=0,  # 初始难度行由 reset_stair_root 按掌握度自定,不走默认抽样
        terrain_generator=TerrainGeneratorCfg(
            # 固定 seed:stairs.py 依赖它逐位复现每块 tile 的几何(见 generator_layouts)。
            seed=20261014, curriculum=True, difficulty_range=(0., 1.),
            # tile 尺寸 20 m x 3 m;8 行难度 x 32 列(平地 25% + 楼梯 75%)。
            size=(20., 3.), num_rows=8, num_cols=32, border_width=3.,
            color_scheme="height", use_cache=False,
            sub_terrains={
                "flat_replay": BpxStairTerrainCfg(proportion=.25, flat=True),
                "stairs": BpxStairTerrainCfg(proportion=.75),
            },
        ),
        # 地面材质:静/动摩擦 1.0、无弹性;机器人侧再由事件做摩擦随机化。
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1., dynamic_friction=1., restitution=0.,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = BPX_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        # 显式的带延迟 PD 执行器用于建模真实电机回路的有限控制频率;
        # 增益/默认姿态/动作契约与部署侧保持一致(仿真到真实的关键)。
        actuators={"legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit=30., effort_limit_sim=30., velocity_limit=20., velocity_limit_sim=20.,
            stiffness=40., damping=1., min_delay=0, max_delay=4,  # 0~4 步随机控制延迟
        )},
    )
    # 全身接触力传感器:history_length=3 供滤波,track_air_time 记录腾空时长。
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750., color=(.9, .9, .9)),
    )


@configclass
class BpxStairsActionsCfg:
    """动作:关节位置比例缩放 0.5 + 默认姿态偏移 + 每回合固定目标误差(±0.02 rad)。"""
    joint_pos = stairs.StairPositionActionCfg(
        asset_name="robot", joint_names=BPX_POLICY_JOINT_NAMES,
        preserve_order=True, scale=.5, use_default_offset=True, target_error_range=(-.02, .02),
    )


@configclass
class BpxStairsObservationsCfg:
    """Actor 保持 450 维盲走输入;Critic 使用无噪声 480 维历史+48 维当前特权量。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测组:6 项基础观测,叠加均匀噪声,拼接 10 步历史。"""
        # 角速度
        base_ang_vel = ObsTerm(func=core.base_ang_vel, noise=Unoise(n_min=-.2, n_max=.2))       # 机体角速度 (3)
        projected_gravity = ObsTerm(func=core.projected_gravity, noise=Unoise(n_min=-.05, n_max=.05))  # 重力投影 (3)
        velocity_commands = ObsTerm(func=core.generated_commands, params={"command_name": "base_velocity"})  # 速度指令 (3)
        joint_pos = ObsTerm(                                                                    # 关节位置偏差 (12)
            func=core.joint_pos_rel, noise=Unoise(n_min=-.01, n_max=.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BPX_POLICY_JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(                                                                    # 关节速度偏差 (12)
            func=core.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BPX_POLICY_JOINT_NAMES, preserve_order=True)},
        )
        actions = ObsTerm(func=core.last_action)                                                # 上一步动作 (12)

        def __post_init__(self):
            self.enable_corruption = True      # 启用观测噪声
            self.concatenate_terms = True      # 各 term 拼接成单一扁平向量
            self.history_length = 10           # 保留 10 步历史(0.18 s @ 50 Hz)
            self.flatten_history_dim = True    # 历史维度展平 -> 45 * 10 = 450 维

    @configclass
    class CriticCfg(ObsGroup):
        """528 维;特权量不堆叠历史,Actor 的观测及部署接口独立保留。"""
        base_lin_vel = ObsTerm(func=core.base_lin_vel, history_length=10)
        base_ang_vel = ObsTerm(func=core.base_ang_vel, history_length=10)
        projected_gravity = ObsTerm(func=core.projected_gravity, history_length=10)
        velocity_commands = ObsTerm(func=core.generated_commands, history_length=10,
                                    params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=core.joint_pos_rel, history_length=10,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BPX_POLICY_JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=core.joint_vel_rel, history_length=10,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BPX_POLICY_JOINT_NAMES, preserve_order=True)},
        )
        actions = ObsTerm(func=core.last_action, history_length=10)
        terrain_context = ObsTerm(func=stairs.critic_terrain_heights)
        support_context = ObsTerm(func=stairs.critic_foot_support, params={"sensor_cfg": feet_sensor()})
        stop_context = ObsTerm(func=stairs.critic_stop_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = None         # 每项独立历史:480+12+8+28=528
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


