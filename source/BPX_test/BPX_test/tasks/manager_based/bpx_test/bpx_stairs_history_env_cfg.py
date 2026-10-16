"""从零开始的上下楼梯与平地混合课程。"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import MeshInvertedPyramidStairsTerrainCfg, MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .bpx_base_env_cfg import BPX_POLICY_JOINT_NAMES
from .bpx_rough_env_cfg import BpxRoughEnvCfg
from .mdp.mixed_stairs_curriculum import MixedStairCurriculum, MixedStairTraversalBoundary
from .mdp.mixed_stairs_commands import MixedStairVelocityCommandCfg
from .stairs_terrain_geometry import BpxDescentTerrainCfg
from .mdp.stairs_rewards import feet_air_time_excess, feet_contact_time_excess, feet_stumble


@configclass
class BpxStairsHistoryCommandsCfg:
    """低台阶前进起步，通过门槛后上下楼均开放后退；平地保留全方向。"""

    base_velocity = MixedStairVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(40.0, 40.0),
        # Flat-only strata: 10% standing, 25% pure yaw, 65% signed mixed motion
        # from the terrain distribution. Stairs use forward/backward translation only, and the base
        # class rel_standing_envs must remain 0 so stair episodes are never
        # muted as standing environments.
        rel_flat_standing_envs=0.10, rel_pure_rotation_envs=0.25,
        pure_rotation_speed_range=(0.2, 1.0),
        flat_lin_vel_x=(-1.0, 1.0), flat_lin_vel_y=(-0.5, 0.5), flat_ang_vel_z=(-1.0, 1.0),
        heading_command=False, debug_vis=False,
        initial_stage=0, initial_sideward_ramp=0,
        # Start with more flat episodes; MixedStairCurriculum automatically
        # changes to 40/40/20 after two qualified flat-quality windows.
        traversal_probabilities=(0.30, 0.30, 0.40),
        descent_initial_speed_range=(0.15, 0.30),
        descent_final_speed_range=(0.30, 0.60),
        backward_speed_range=(0.2, 0.4),
        # Stair forward keeps the positive-only range of the last trained run;
        # flat mixed motion uses the signed flat_* fields above.
        ranges=MixedStairVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.6), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class BpxStairsHistoryObservationsCfg:
    """Actor 每帧 45 维，Critic 每帧 48 维；均保留 10 帧历史。

    两组观测独立声明，不继承单帧任务的观测配置。只有 Critic 能直接读取
    基座线速度真值；其余观测的噪声规则一致，但两组分别采样噪声。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor 输入：不提供任何时刻的基座线速度，共 450 维。"""

        # 机身坐标系角速度 [wx, wy, wz]，3 维，均匀噪声范围 ±0.2。
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        # 机身坐标系重力投影，3 维，均匀噪声范围 ±0.05。
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 目标速度 [vx, vy, wz]，3 维；这是控制指令，不是真实线速度。
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        # 相对默认站姿的关节角，12 维；严格使用训练与部署共用的关节顺序。
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=BPX_POLICY_JOINT_NAMES, preserve_order=True,
                ),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 相对默认关节速度，12 维，均匀噪声范围 ±1.5。
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=BPX_POLICY_JOINT_NAMES, preserve_order=True,
                ),
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        # 上一策略周期的动作，12 维；当前观测对应 a_(t-1)，而非待预测的 a_t。
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # 新观测入历史缓存前加噪；每项从最旧到最新展开，再按声明顺序拼接。
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 输入：额外拥有基座线速度真值，共 480 维。"""

        # 唯一额外特权：机身坐标系基座线速度 [vx, vy, vz]，3 维，不加噪声。
        # 沿用 mdp.base_lin_vel 的 root_lin_vel_b 定义。
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # 机身坐标系角速度，3 维；噪声与 Actor 相同。
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        # 机身坐标系重力投影，3 维；噪声与 Actor 相同。
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 目标速度 [vx, vy, wz]，3 维，无噪声。
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        # 相对默认站姿的关节角，12 维；关节顺序、噪声均与 Actor 相同。
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=BPX_POLICY_JOINT_NAMES, preserve_order=True,
                ),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 相对默认关节速度，12 维；关节顺序、噪声均与 Actor 相同。
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=BPX_POLICY_JOINT_NAMES, preserve_order=True,
                ),
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        # 上一策略周期的动作，12 维，无噪声。
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # 保留其他状态的观测噪声，仅线速度读取真值；历史布局与 Actor 一致。
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class BpxStairsHistoryRewardsCfg:
    """楼梯独立奖励：放宽水平姿态约束，保留站姿并约束脚滑和踢面。

    奖励计算可使用仿真真值；这不会将真值作为观测输入传给 Actor。
    正权重用于奖励，负权重用于惩罚。
    """

    # 水平面 x、y 方向线速度的指数跟踪奖励，误差尺度为 0.5 m/s。
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # 绕 z 轴的偏航角速度指数跟踪奖励，误差尺度为 0.5 rad/s。
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # 足端腾空时间奖励，以 0.4 秒为阈值，鼓励抬脚迈步。
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )
    # 每足连续腾空超过 0.6 s 后渐增惩罚，1.0 s 达到单足上限。
    feet_air_time_excess = RewTerm(
        func=feet_air_time_excess,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
            "command_name": "base_velocity",
            "max_air_time": 0.6,
            "ramp_time": 0.4,
            "command_threshold": 0.1,
        },
    )
    # 每足连续支撑超过 1.0 s 后渐增惩罚，1.5 s 达到单足上限。
    feet_contact_time_excess = RewTerm(
        func=feet_contact_time_excess,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
            "command_name": "base_velocity",
            "max_contact_time": 1.0,
            "ramp_time": 0.5,
            "command_threshold": 0.1,
        },
    )
    # 惩罚竖直线速度平方，抑制机身上下跳动。
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # 惩罚横滚、俯仰角速度平方，抑制机身快速摇摆。
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 惩罚机身偏离水平姿态。
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.1)
    # 惩罚关节力矩平方，减少过大的驱动力矩。
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # 惩罚关节加速度平方，抑制关节运动突变。
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # 惩罚相邻策略周期动作差的平方，使输出更平滑。
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # 惩罚关节角相对默认站姿的绝对偏差。
    joint_deviation_l1 = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05)
    # 支撑脚水平滑动；传感器与刚体严格保持相同的四足顺序。
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link", preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe_link", preserve_order=True),
        },
    )
    # 轻量踢面惩罚：任意足水平力 > 4 倍竖直力，且 > 5 N。
    feet_stumble = RewTerm(
        func=feet_stumble,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link"),
            "force_ratio": 4.0,
            "min_horizontal_force": 5.0,
        },
    )
    # 惩罚机身、髋部、大腿、小腿等非足端部位的接触，力阈值为 1 N。
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
class BpxStairsHistoryCurriculumCfg:
    """每环境独立维护上楼/下楼×移动模式的等级，平地固定为零级。"""

    terrain_levels = CurrTerm(
        func=MixedStairCurriculum,
        params={
            "initial_up_levels": (0, 0),
            "command_min_episodes": 128,
            "command_success_threshold": 0.8,
            "command_success_windows": 2,
            "command_min_level": 1,
            "flat_success_threshold": 0.90,
            "flat_success_windows": 2,
            # Whole-episode quality after reset settling; brief excursions do
            # not invalidate an otherwise well tracked flat episode.
            "flat_min_quality_time": 2.0,
            "flat_min_stable_fraction": 0.90,
            "flat_max_unstable_time": 0.5,
            "statistics_window": 256,
            "min_stair_time": 1.0,
            "max_linear_error": 0.35,
            "max_descent_linear_error": 0.10,
            "max_angular_error": 0.5,
            "min_speed_ratio": 0.6,
            "successes_to_promote": 2,
            "failures_to_demote": 2,
        },
    )


@configclass
class BpxStairsHistoryEnvCfg(BpxRoughEnvCfg):
    """中央低台向外上楼、中央高台向外下楼，以及纯平地回合。"""

    observations: BpxStairsHistoryObservationsCfg = BpxStairsHistoryObservationsCfg()
    commands: BpxStairsHistoryCommandsCfg = BpxStairsHistoryCommandsCfg()
    rewards: BpxStairsHistoryRewardsCfg = BpxStairsHistoryRewardsCfg()
    curriculum: BpxStairsHistoryCurriculumCfg = BpxStairsHistoryCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.debug_vis = False
        # 0.15 m/s、随机朝向时也有时间走完约 5 m 的对角线路径。
        self.episode_length_s = 40.0
        self.terminations.terrain_boundary.func = MixedStairTraversalBoundary
        self.terminations.terrain_boundary.params = {
            "rough_start": 1.0, "boundary": 4.5, "completion_radius": 3.5,
            "landing_duration": 0.4, "height_tolerance": 0.15,
            "max_tilt": 0.35, "max_vertical_speed": 0.2, "max_angular_speed": 1.0,
            "contact_force": 5.0,
            # Legacy dwell remains diagnostic; flat success uses episode quality.
            "flat_completion_duration": 1.0,
            "flat_quality_warmup": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe_link", preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe_link", preserve_order=True),
        }
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=20261016, curriculum=True, difficulty_range=(0.0, 1.0),
            size=(10.0, 10.0), num_rows=8, num_cols=6, border_width=5.0,
            color_scheme="height", use_cache=False,
            # Equal proportions give columns [up, up, down, down, flat, flat].
            # The command curriculum initially samples episodes at 30/30/40%;
            # restore 40/40/20% after flat locomotion has stabilized.
            sub_terrains={
                "up_stairs": MeshInvertedPyramidStairsTerrainCfg(
                    proportion=1.0, step_height_range=(0.02, 0.15),
                    step_width=0.2, platform_width=2.2, border_width=2.0, holes=False,
                ),
                "down_stairs": BpxDescentTerrainCfg(
                    proportion=1.0, step_height_range=(0.03, 0.15),
                    step_width=0.2, platform_width=2.2, border_width=2.0, holes=False,
                ),
                "flat": MeshPlaneTerrainCfg(proportion=1.0),
            },
        )
        # 保留中央实际 2×2 m 平台和 11 个踢面，边缘半径 1.0…3.0 m。
        # 外圈平台扩宽至半径 5 m，3.5 m 后稳定行走 0.4 s 才完成。
