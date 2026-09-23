"""崎岖行走历史任务：显式定义非对称观测和奖励，便于独立调整。"""

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .bpx_base_env_cfg import BPX_POLICY_JOINT_NAMES
from .bpx_rough_env_cfg import BpxRoughEnvCfg


@configclass
class BpxRoughHistoryObservationsCfg:
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

        # 观测历史缓存的布局：每项按声明顺序展开，再按时间顺序拼接，最旧在前，最新在后。每帧包含 450 维观测。
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
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel
        )
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
class BpxRoughHistoryRewardsCfg:
    """历史任务独立奖励配置，当前参数与原崎岖行走任务一致。

    奖励计算可使用仿真真值；这不会将真值作为观测输入传给 Actor。
    正权重用于奖励，负权重用于惩罚。
    """


    """
    奖励
    """
    # 水平面 x、y 方向线速度的指数跟踪奖励，误差尺度为 0.5 m/s。
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
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

    """
    惩罚
    """
    # 惩罚竖直线速度平方，抑制机身上下跳动。
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2, 
        weight=-2.0
    )
    # 惩罚横滚、俯仰角速度平方，抑制机身快速摇摆。
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2, 
        weight=-0.05
    )
    # 惩罚机身偏离水平姿态。
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2, 
        weight=-0.5
    )
    # 惩罚关节力矩平方，减少过大的驱动力矩。
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-1.0e-5
    )
    # 惩罚关节加速度平方，抑制关节运动突变。
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, 
        weight=-2.5e-7
    )
    # 惩罚相邻策略周期动作差的平方，使输出更平滑。
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.01
    )
    # 惩罚关节角相对默认站姿的绝对偏差。
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1, 
        weight=-0.05
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
class BpxRoughHistoryEnvCfg(BpxRoughEnvCfg):
    """使用本文件的观测与奖励，复用崎岖任务的地形、指令、动作和课程等配置。

    H=10 包含当前帧，在 50 Hz 下首末帧相隔 0.18 秒；各观测项分别维护历史，
    按项展开后拼接。复位后的首帧填满该环境的历史缓存。
    """

    observations: BpxRoughHistoryObservationsCfg = BpxRoughHistoryObservationsCfg()
    rewards: BpxRoughHistoryRewardsCfg = BpxRoughHistoryRewardsCfg()
