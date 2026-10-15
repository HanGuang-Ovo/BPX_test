"""RSL-RL 2.3 PPO with frozen-Move warmup excluded from every gradient update.

Keep complete simulator trajectories for GAE. Hold -> reset is a done boundary;
warmup precedes Hold, so invalid teacher advantages cannot flow into Hold. Only
student samples enter advantage normalization, actor/value/entropy/KL losses.
"""

from copy import deepcopy

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage


class HoldRolloutStorage(RolloutStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid = torch.zeros(self.num_transitions_per_env, self.num_envs,
                                 device=self.device, dtype=torch.bool)

    def compute_returns(self, last_values, gamma, lam, normalize_advantage=True):
        super().compute_returns(last_values, gamma, lam, normalize_advantage=False)
        if normalize_advantage and self.valid.any():
            selected = self.advantages[self.valid]
            self.advantages[self.valid] = (selected - selected.mean()) / (selected.std(unbiased=False) + 1.e-8)
        self.advantages[~self.valid] = 0.

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        indices = self.valid.flatten().nonzero().flatten()
        critic = self.privileged_observations if self.privileged_observations is not None else self.observations
        tensors = (self.observations, critic, self.actions, self.values, self.advantages,
                   self.returns, self.actions_log_prob, self.mu, self.sigma)
        flat = [value.flatten(0, 1) for value in tensors]
        for _ in range(num_epochs):
            order = indices[torch.randperm(len(indices), device=self.device)]
            for batch in torch.tensor_split(order, num_mini_batches):
                if len(batch):
                    yield (*(value[batch] for value in flat), (None, None), None, None)


class HoldPPO(PPO):
    def init_storage(self, training_type, num_envs, num_transitions_per_env,
                     actor_obs_shape, critic_obs_shape, actions_shape):
        if self.policy.is_recurrent or self.rnd or self.symmetry or self.is_multi_gpu:
            raise ValueError("Masked Hold PPO supports feedforward, single-device PPO without RND/symmetry")
        self.storage = HoldRolloutStorage(training_type, num_envs, num_transitions_per_env,
                                          actor_obs_shape, critic_obs_shape, actions_shape, device=self.device)

    def process_env_step(self, rewards, dones, infos):
        if "hold_action_mask" not in infos:
            raise RuntimeError("Hold PPO requires the pre-step hold_action_mask from BpxStairsHoldEnv")
        self.storage.valid[self.storage.step].copy_(infos["hold_action_mask"].to(self.device))
        super().process_env_step(rewards, dones, infos)

    def update(self):
        count = int(self.storage.valid.sum())
        fraction = count / self.storage.valid.numel()
        if count == 0:
            self.storage.clear()
            result = {"value_function": 0., "surrogate": 0., "entropy": 0.}
        else:
            configured_batches = self.num_mini_batches
            self.num_mini_batches = min(configured_batches, count)
            try:
                result = super().update()
            finally:
                self.num_mini_batches = configured_batches
        result.update(hold_samples=count, hold_sample_fraction=fraction)
        return result


class HoldOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        algorithm = deepcopy(train_cfg["algorithm"])
        algorithm.pop("class_name")
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.alg = HoldPPO(self.alg.policy, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg, **algorithm)
        self.alg.init_storage(self.training_type, env.num_envs, self.num_steps_per_env,
                              [env.num_obs], [env.num_privileged_obs], [env.num_actions])
