"""Rough locomotion with ten frames of the complete 48-D observation."""

from isaaclab.utils import configclass

from .bpx_rough_env_cfg import BpxRoughEnvCfg


@configclass
class BpxRoughHistoryEnvCfg(BpxRoughEnvCfg):
    """480-D shared actor/critic input; no privileged critic observations.

    Isaac Lab flattens each term's history oldest-to-newest, then concatenates
    terms. H=10 includes the current frame (0.18 s span at 50 Hz). Each frame
    contains the previous action, not the action that the policy will predict.
    After reset the first observation fills that environment's history.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 10
        self.observations.policy.flatten_history_dim = True

