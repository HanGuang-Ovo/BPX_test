"""Real simulator smoke check for one split task (run each in a fresh process)."""

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=["BPX-Stairs-Move-v0", "BPX-Stairs-Hold-v0"], required=True)
parser.add_argument("--move_checkpoint", default=None)
parser.add_argument("--steps", type=int, default=250)

# Match the stair train/play bootstrap before Isaac Sim loads Warp extensions.
import warp
import warp.types
for name in ("array", "indexedarray"):
    if not hasattr(warp.types, name):
        setattr(warp.types, name, getattr(warp, name))

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import sys
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import BPX_test.tasks  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent / "rsl_rl"))
from hold_ppo import HoldOnPolicyRunner


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 4
    cfg.scene.terrain.terrain_generator.num_cols = 8
    cfg.sim.device = args.device
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**16
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**16
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**16
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**16
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**14
    cfg.sim.physx.gpu_heap_capacity = 2**24
    cfg.sim.physx.gpu_temp_buffer_capacity = 2**22
    hold_task = getattr(cfg, "hold_policy_training", False)
    if hold_task and args.move_checkpoint:
        cfg.move_checkpoint = args.move_checkpoint
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        wrapped = RslRlVecEnvWrapper(env)
        agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent.device = env.device
        agent.num_steps_per_env = 8
        actor_obs, extras = wrapped.get_observations()
        assert actor_obs.shape == (4, 450)
        assert extras["observations"]["critic"].shape == (4, 528)
        assert wrapped.num_actions == 12
        if hold_task:
            assert list(env.termination_manager.active_terms)[0] == "hold_finished"
            assert "traversal_bonus" not in env.reward_manager.active_terms
            # Exercise both stationary reset and actual frozen-Move warmup.
            env._bpx_stair_curriculum.level[:] = torch.tensor([0, 1, 2, 3], device=env.device)
            env.event_manager.get_term_cfg("reset_base").params["replay_probability"] = 0.
            env._bpx_stair_curriculum.cfg.params["enabled"] = False
            actor_obs, _ = wrapped.reset()
            runner = HoldOnPolicyRunner(wrapped, agent.to_dict(), device=env.device)
        else:
            assert all(not name.startswith("hold_") for name in env.reward_manager.active_terms)
            assert env._bpx_stairs_command.cfg.stop_probability == 0.
            runner = OnPolicyRunner(wrapped, agent.to_dict(), device=env.device)
        seen = torch.zeros(2, device=env.device, dtype=torch.bool)
        for index in range(args.steps):
            with torch.inference_mode():
                policy = runner.alg.policy
                action = policy.act_inference(actor_obs)
                before = (env._bpx_stairs_command.phase == 2).clone()
                actor_obs, reward, done, extras = wrapped.step(action)
                assert torch.isfinite(actor_obs).all() and torch.isfinite(reward).all()
                if hold_task:
                    torch.testing.assert_close(extras["hold_action_mask"], before)
                    assert not reward[~before].any()
                    seen[0] |= (~before).any()
                    seen[1] |= before.any()
        if hold_task:
            assert seen.all(), "Expected both frozen warmup and trainable Hold steps"
        # A logging-free runner.learn() is not supported by the installed RSL-RL
        # version. Exercise one real collection/update directly instead.
        if hold_task:
            env._bpx_stair_curriculum.level[:] = 0
            wrapped.reset()
        actor_obs, extras = wrapped.get_observations()
        critic_obs = extras["observations"]["critic"]
        for _ in range(agent.num_steps_per_env):
            with torch.inference_mode():
                action = runner.alg.act(actor_obs, critic_obs)
                actor_obs, reward, done, extras = wrapped.step(action)
                critic_obs = extras["observations"]["critic"]
                runner.alg.process_env_step(reward, done, extras)
        with torch.inference_mode():
            runner.alg.compute_returns(critic_obs)
        losses = runner.alg.update()
        assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
        if hold_task:
            assert losses["hold_samples"] > 0
        print(f"PASS: {args.task}: 450/528 observations, 12 actions, finite steps, PPO update")
    finally:
        env.close()


try:
    main()
finally:
    app.close()
