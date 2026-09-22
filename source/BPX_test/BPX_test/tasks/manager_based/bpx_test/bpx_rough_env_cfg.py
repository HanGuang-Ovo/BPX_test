"""Blind rough locomotion: rewards written out explicitly, 48-D single-frame observations."""
import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from . import mdp
from .bpx_locomotion_env_cfg import BpxLocomotionEnvCfg, BpxLocomotionTerminationsCfg
from .mdp.rough_curriculum import RoughTraversalBoundary, TraversabilityCurriculum
from .rough_terrain_geometry import wave_terrain


@configclass
# 波浪地形配置
class BpxWaveTerrainCfg(SubTerrainBaseCfg):
    function = wave_terrain
    horizontal_scale: float = 0.025
    amplitude_range: tuple[float, float] = (0.005, 0.04)
    wavelength_range: tuple[float, float] = (0.15, 0.30)
    levels: int = 8
    platform_width: float = 2.0
    transition_width: float = 0.5


@configclass
# 三维速度指令观测
class BpxRoughCommandsCfg:
    # Each component independently uniform. No standing or pure-rotation mixture.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0, heading_command=False, debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0),
        ),
    )


# 单帧基线；H=10 完整观测堆叠见独立的 bpx_rough_history_env_cfg.py。

@configclass
# 崎岖行走奖励：显式写出（不继承平地行走），当前数值与平地一致，保持策略契约不变，
# 之后可独立调整崎岖任务的奖励权重或增删奖励项。
class BpxRoughRewardsCfg:
    # x,y方向的线速度跟踪奖励，，
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # z方向的角速度惩罚
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # 足端腾空时间奖励，鼓励足端腾空时间长一些，避免贴地滑行
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )

    # z方向的线速度惩罚
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2, 
        weight=-2.0
    )

    # x,y方向的角速度惩罚
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2, 
        weight=-0.05
    )

    # 机器人姿态平整度惩罚
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2, 
        weight=-0.5
    )

    # 关节力矩惩罚
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-1.0e-5
    )

    # 关节加速度惩罚
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, 
        weight=-2.5e-7
    )

    # 动作变化率惩罚
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.01
    )

    # 关节偏离目标位置惩罚
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1, 
        weight=-0.05
        )

    # 不期望的接触惩罚
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
# 终止条件配置
class BpxRoughTerminationsCfg(BpxLocomotionTerminationsCfg):
    # Time-limit truncation (value bootstrap), not an extra failure penalty.
    terrain_boundary = DoneTerm(
        func=RoughTraversalBoundary, time_out=True,
        params={"rough_start": 1.95, "boundary": 3.4},
    )


@configclass
# 课程配置
class BpxRoughCurriculumCfg:
    terrain_levels = CurrTerm(
        func=TraversabilityCurriculum,
        params={"min_rough_distance": 1.0, "min_rough_time": 1.0,
                "max_linear_error": 0.35, "max_angular_error": 0.5,
                "successes_to_promote": 2, "failures_to_demote": 2},
    )


@configclass
# 盲人粗糙地形环境配置
class BpxRoughEnvCfg(BpxLocomotionEnvCfg):
    commands: BpxRoughCommandsCfg = BpxRoughCommandsCfg()
    rewards: BpxRoughRewardsCfg = BpxRoughRewardsCfg()
    terminations: BpxRoughTerminationsCfg = BpxRoughTerminationsCfg()
    curriculum: BpxRoughCurriculumCfg = BpxRoughCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=20260918, curriculum=True, size=(8.0, 8.0),
            num_rows=8, num_cols=4, border_width=5.0,
            color_scheme="height", use_cache=False,
            sub_terrains={"bpx_waves": BpxWaveTerrainCfg()},
        )
        # Flat 2x2 m platform; ±0.1 m center jitter leaves room for feet at any yaw.
        # Zero initial velocity prevents resets from immediately sliding off it.
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
