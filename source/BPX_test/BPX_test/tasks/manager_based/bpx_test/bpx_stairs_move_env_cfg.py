"""连续上下楼梯：独立移动策略，不安排停车、不考核停驻。"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from .bpx_stairs_base_env_cfg import (
    BpxStairsBaseEnvCfg, BpxStairsCommandsCfg, BpxStairsRewardsCfg,
)
from .mdp.stairs_skills import StairMoveCurriculum


@configclass
class BpxStairsMoveCommandsCfg(BpxStairsCommandsCfg):
    def __post_init__(self):
        self.base_velocity.stop_probability = 0.
        self.base_velocity.flat_standing_probability = 0.


@configclass
class BpxStairsMoveRewardsCfg(BpxStairsRewardsCfg):
    # Only physical traversal earns an exit bonus. No reward depends on Hold.
    completion_bonus = None
    hold_quality = None
    hold_success_bonus = None
    hold_drift = None
    hold_linear_velocity = None
    hold_angular_velocity = None
    hold_joint_motion = None
    hold_support = None


@configclass
class BpxStairsMoveCurriculumCfg:
    terrain_mastery = CurrTerm(
        func=StairMoveCurriculum,
        params={"initial_level": 0, "successes_to_promote": 3, "failures_to_demote": 2,
                "max_linear_error": .20, "max_angular_error": .30, "enabled": True},
    )


@configclass
class BpxStairsMoveEnvCfg(BpxStairsBaseEnvCfg):
    commands: BpxStairsMoveCommandsCfg = BpxStairsMoveCommandsCfg()
    rewards: BpxStairsMoveRewardsCfg = BpxStairsMoveRewardsCfg()
    curriculum: BpxStairsMoveCurriculumCfg = BpxStairsMoveCurriculumCfg()
