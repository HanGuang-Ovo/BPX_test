"""Same PPO/network as flat locomotion, separate experiment directory."""
from isaaclab.utils import configclass
from .bpx_locomotion_ppo_cfg import BpxLocomotionPPORunnerCfg


@configclass
class BpxRoughPPORunnerCfg(BpxLocomotionPPORunnerCfg):
    experiment_name = "bpx_rough"
