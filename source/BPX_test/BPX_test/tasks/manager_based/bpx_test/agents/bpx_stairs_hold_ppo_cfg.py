"""Warm-start Hold from rough standing; frozen Move is configured on the env."""

from isaaclab.utils import configclass
from .bpx_stairs_base_ppo_cfg import BpxStairsBasePPORunnerCfg


@configclass
class BpxStairsHoldPPORunnerCfg(BpxStairsBasePPORunnerCfg):
    experiment_name = "bpx_stairs_hold"
    run_name = "stairs_hold_v2"
    finetune_checkpoint: str | None = (
        "logs/rsl_rl/bpx_rough_stand/2026-09-20_13-01-51/model_1998.pt"
    )
    # The source uses a 48-D privileged single frame. Its compatible non-velocity
    # inputs initialize the latest frame of the blind 450-D history actor.
    finetune_actor_input_adapter: str | None = "stand48_to_blind450"
    finetune_reset_critic: bool = True
    num_steps_per_env = 96

    def __post_init__(self):
        super().__post_init__()
        # Source std is approximately .016-.046. Start near that exploration
        # scale instead of raising every joint to the generic stair default .08.
        self.policy.init_noise_std = .03
