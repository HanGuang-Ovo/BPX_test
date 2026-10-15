"""楼梯停驻：微调站立 Actor；冻结 Move 仅负责产生真实交接状态。"""

import isaaclab.envs.mdp as core
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .bpx_stairs_common_cfg import feet_asset, feet_sensor
from .bpx_stairs_base_env_cfg import BpxStairsBaseEnvCfg, BpxStairsEventsCfg
from .mdp import stairs
from .mdp import stairs_skills as hold


def hold_reward(function, weight, **kwargs):
    return RewTerm(func=hold.HoldReward, weight=weight,
                   params={"function": function, "kwargs": kwargs})


@configclass
class BpxStairsHoldCommandsCfg:
    base_velocity = hold.StairHoldCommandCfg()


@configclass
class BpxStairsHoldEventsCfg(BpxStairsEventsCfg):
    reset_base = EventTerm(func=hold.reset_stair_hold_root, mode="reset",
                          params={"replay_probability": .20, "approach_range": (.60, 1.50), "height": .40})


@configclass
class BpxStairsHoldRewardsCfg:
    component_quality = hold_reward(hold.hold_component_quality, 2., sensor_cfg=feet_sensor())
    joint_quality = hold_reward(stairs.stop_quality_reward, .5, sensor_cfg=feet_sensor())
    success = hold_reward(hold.hold_success_reward, 2.)
    linear_velocity = hold_reward(stairs.stop_linear_velocity_l2, -.25,
                                  speed_scale=.08, max_normalized_error=25.)
    angular_velocity = hold_reward(stairs.stop_angular_velocity_l2, -.25,
                                   speed_scale=.20, max_normalized_error=25.)
    drift = hold_reward(stairs.stop_drift_l2, -8., max_distance=.20)
    support_height = hold_reward(stairs.support_height_l2, -6.,
                                 feet_cfg=feet_asset(), sensor_cfg=feet_sensor(), target=.40, tolerance=.07)
    moderate_tilt = hold_reward(stairs.tilt_deadzone_l2, -.8, roll_tolerance=.15, pitch_tolerance=.45)
    feet_slip = hold_reward(stairs.stance_slip_l2, -.4, feet_cfg=feet_asset(), sensor_cfg=feet_sensor())
    # Necessary corrective steps are allowed during the fixed settling interval.
    joint_motion = hold_reward(hold.settled_joint_motion, -.002, settle_time=.30)
    torque = hold_reward(core.joint_torques_l2, -1.e-5)
    joint_limits = hold_reward(core.joint_pos_limits, -1.)
    action_change = hold_reward(core.action_rate_l2, -.01)
    non_foot_contact = hold_reward(core.undesired_contacts, -.75,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=[
            "torso", ".*_hip_link", ".*_thigh_link", ".*_calf_link"]), threshold=5.)
    failure = hold_reward(core.is_terminated, -5.)


@configclass
class BpxStairsHoldTerminationsCfg:
    # Hold finishes at its fixed deadline, regardless of whether it succeeded.
    # No dependence on traversal, exit position, or a restart command.
    hold_finished = DoneTerm(func=hold.StairHoldEpisodeMonitor, time_out=True,
                             params={"feet_cfg": feet_asset(), "sensor_cfg": feet_sensor(), "settle_time": .30})
    time_out = DoneTerm(func=core.time_out, time_out=True)
    torso_contact = DoneTerm(func=core.illegal_contact,
                            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso"), "threshold": 5.})
    fallen_orientation = DoneTerm(func=core.bad_orientation, params={"limit_angle": .95})
    low_body = DoneTerm(func=stairs.low_support_height,
        params={"feet_cfg": feet_asset(), "sensor_cfg": feet_sensor(), "minimum": .16})
    tile_boundary = DoneTerm(func=stairs.stair_boundary, params={"margin": .15})


@configclass
class BpxStairsHoldCurriculumCfg:
    hold_mastery = CurrTerm(func=hold.StairHoldCurriculum,
        params={"initial_level": 0, "successes_to_promote": 3, "failures_to_demote": 2, "enabled": True})


@configclass
class BpxStairsHoldEnvCfg(BpxStairsBaseEnvCfg):
    hold_policy_training: bool = True
    # This actor generates warmup states only. Hold initialization is configured
    # independently in agents/bpx_stairs_hold_ppo_cfg.py.
    # Replace with a trained StairsMove checkpoint when advancing to high stairs.
    move_checkpoint: str | None = "logs/rsl_rl/bpx_rough_history_asymmetric/2026-09-23_11-17-22/model_4999.pt"
    commands: BpxStairsHoldCommandsCfg = BpxStairsHoldCommandsCfg()
    events: BpxStairsHoldEventsCfg = BpxStairsHoldEventsCfg()
    rewards: BpxStairsHoldRewardsCfg = BpxStairsHoldRewardsCfg()
    terminations: BpxStairsHoldTerminationsCfg = BpxStairsHoldTerminationsCfg()
    curriculum: BpxStairsHoldCurriculumCfg = BpxStairsHoldCurriculumCfg()
