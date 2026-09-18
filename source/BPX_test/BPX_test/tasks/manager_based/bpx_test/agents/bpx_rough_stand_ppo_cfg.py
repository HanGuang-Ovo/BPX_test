"""Stand-compatible network, separate rough standing logs."""
from isaaclab.utils import configclass
from .bpx_stand_ppo_cfg import BpxStandPPORunnerCfg


@configclass
class BpxRoughStandPPORunnerCfg(BpxStandPPORunnerCfg):
    experiment_name = 'bpx_rough_stand'
