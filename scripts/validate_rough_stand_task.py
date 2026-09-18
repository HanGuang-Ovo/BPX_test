"""Run inside Isaac Lab: python -u scripts/validate_rough_stand_task.py --headless."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import torch
import gymnasium as gym
import BPX_test.tasks
from BPX_test.tasks.manager_based.bpx_test.bpx_stand_env_cfg import BpxStandEnvCfg
from BPX_test.tasks.manager_based.bpx_test.bpx_rough_stand_env_cfg import BpxRoughStandEnvCfg
from BPX_test.tasks.manager_based.bpx_test.mdp.rough_stand import ground_heights, relative_stand_height_l2, reset_rough_stand


def main():
    cfg, flat = BpxRoughStandEnvCfg(), BpxStandEnvCfg()
    assert cfg.observations.to_dict() == flat.observations.to_dict()
    assert cfg.actions.to_dict() == flat.actions.to_dict()
    assert cfg.commands.to_dict() == flat.commands.to_dict()
    rewards = cfg.rewards.to_dict()
    rewards['base_height_l2'] = flat.rewards.to_dict()['base_height_l2']
    assert rewards == flat.rewards.to_dict()
    assert flat.scene.terrain.terrain_type == 'plane'
    cfg.scene.num_envs = 4
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.events.reset_base.params['flat_probability'] = 0.
    env = gym.make('BPX-Stand-Rough-v0', cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs['policy'].shape == (4, 48)
        assert torch.all(env.command_manager.get_command('base_velocity') == 0)
        ids = torch.arange(4, device=env.device)
        terrain, tracker = env.scene.terrain, env._bpx_stand_stability
        pos = env.scene['robot'].data.root_pos_w
        local = pos - env.scene.env_origins
        assert torch.all(local[:, 0].abs() == 2.5)
        heights = ground_heights(env, pos[:, :2])
        assert torch.all(heights.std(dim=1) > 0)
        torch.testing.assert_close(pos[:, 2], heights.amax(dim=1) + .42)
        torch.testing.assert_close(relative_stand_height_l2(env, .4), (pos[:, 2] - heights.mean(dim=1) - .4).square())
        reset_rough_stand(env, ids, flat_probability=1.)
        env.scene.update(env.step_dt)
        assert not tracker.rough.any()
        assert torch.all(ground_heights(env, tracker.spawn_xy).abs() < 1e-6)
        reset_rough_stand(env, ids, flat_probability=0.)

        def finish(stable=True, fallen=False, rough=True, duration=14.5):
            env.episode_length_buf[:] = 750
            tracker.duration[:] = duration
            tracker.stable_time[:] = duration if stable else duration * .5
            tracker.rough[:] = rough
            env.termination_manager.terminated[:] = fallen
            env._reset_idx(ids)
            assert torch.all(tracker.duration == 0)
        finish()
        assert torch.all(terrain.terrain_levels == 0)
        finish()
        assert torch.all(terrain.terrain_levels == 1)
        finish(rough=False)
        finish(rough=False)
        assert torch.all(terrain.terrain_levels == 1)
        finish(fallen=True)
        assert torch.all(terrain.terrain_levels == 0)
        finish(duration=1.)
        finish()
        assert torch.all(terrain.terrain_levels == 0)
        finish()
        assert torch.all(terrain.terrain_levels == 1)
        finish(stable=False)
        assert torch.all(terrain.terrain_levels == 0)
        terrain.terrain_levels[:] = 7
        terrain.update_env_origins(ids, torch.zeros_like(ids), torch.zeros_like(ids))
        finish()
        finish()
        assert torch.all(terrain.terrain_levels == 7)
        state = env.scene['robot'].data.root_state_w.clone()
        state[:, :2] = tracker.spawn_xy + torch.tensor([.7, 0.], device=env.device)
        env.scene['robot'].write_root_state_to_sim(state, env_ids=ids)
        env.scene.update(env.step_dt)
        done = tracker(env, **cfg.terminations.stand_drift.params)
        assert torch.all(done)
        env.reset()
        for _ in range(10):
            obs, reward, _, _, _ = env.step(torch.zeros((4, 12), device=env.device))
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        print('PASS: 48-D interface, zero command, reward isolation, rough/flat reset, relative height, curriculum, drift failure, physics steps', flush=True)
    finally:
        env.close()

try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
else:
    app.close()
