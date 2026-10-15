"""Independent Move optimizer/checkpoints, initialized from walking actor only."""

from isaaclab.utils import configclass
from .bpx_stairs_base_ppo_cfg import BpxStairsBasePPORunnerCfg


@configclass
class BpxStairsMovePPORunnerCfg(BpxStairsBasePPORunnerCfg):
    experiment_name = "bpx_stairs_move"
    run_name = "stairs_move_v1"
    finetune_reset_critic: bool = True
