"""Same PPO and MLP as rough locomotion, with a separate history experiment."""

from isaaclab.utils import configclass

from .bpx_rough_ppo_cfg import BpxRoughPPORunnerCfg


@configclass
class BpxRoughHistoryPPORunnerCfg(BpxRoughPPORunnerCfg):
    experiment_name = "bpx_rough_history_asymmetric"

