"""Use frozen Move actions during warmup, student Hold actions after handoff."""

from pathlib import Path

import torch
from torch import nn
from isaaclab.envs import ManagerBasedRLEnv


def load_move_actor(checkpoint, device):
    """Load the shared 450 -> [128,128,128] ELU -> 12 deployment contract."""
    if not checkpoint or not Path(checkpoint).expanduser().is_file():
        raise FileNotFoundError(
            f"Hold warmup needs a Move checkpoint: {checkpoint!r}. "
            "Set --move_checkpoint or env.move_checkpoint to a compatible walking model.")
    saved = torch.load(Path(checkpoint).expanduser(), map_location=device, weights_only=True)
    if "obs_norm_state_dict" in saved:
        raise ValueError("Hold warmup requires an unnormalized 450-D ELU Move actor")
    actor = nn.Sequential(nn.Linear(450, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(),
                          nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 12)).to(device)
    actor.load_state_dict({key.removeprefix("actor."): value
                           for key, value in saved["model_state_dict"].items() if key.startswith("actor.")})
    actor.eval().requires_grad_(False)
    return actor


class BpxStairsHoldEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        self._move_actor = None
        super().__init__(cfg, **kwargs)
        self._bpx_hold_action_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def step(self, action):
        # Snapshot BEFORE stepping: phase/reset can change inside super().step.
        # This mask corresponds to the actions stored by PPO, including the last
        # Hold action of an episode. Never derive it from post-reset observations.
        mask = (self._bpx_stairs_command.phase == 2).clone()
        self._bpx_hold_action_mask = mask
        executed = action.to(self.device).clone()
        if (~mask).any():
            if self._move_actor is None:
                self._move_actor = load_move_actor(self.cfg.move_checkpoint, self.device)
            with torch.no_grad():
                executed[~mask] = self._move_actor(self.obs_buf["policy"][~mask])
        obs, reward, terminated, time_outs, extras = super().step(executed)
        extras["hold_action_mask"] = mask
        return obs, reward, terminated, time_outs, extras
