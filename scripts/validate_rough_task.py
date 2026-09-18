"""Validate rough geometry, unchanged policy contract and real manager reset ordering.

Run in the Isaac Lab environment: python scripts/validate_rough_task.py --headless
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import BPX_test.tasks
from BPX_test.tasks.manager_based.bpx_test.bpx_locomotion_env_cfg import BpxLocomotionEnvCfg
from BPX_test.tasks.manager_based.bpx_test.bpx_rough_env_cfg import BpxRoughEnvCfg
from BPX_test.tasks.manager_based.bpx_test.rough_terrain_geometry import wave_height_field, wave_terrain, terrain_amplitude


def main():
    cfg = BpxRoughEnvCfg()
    flat = BpxLocomotionEnvCfg()
    assert cfg.rewards.to_dict() == flat.rewards.to_dict()
    assert cfg.observations.to_dict() == flat.observations.to_dict()
    assert cfg.actions.to_dict() == flat.actions.to_dict()
    assert flat.scene.terrain.terrain_type == 'plane'
    assert flat.events.reset_base.params['pose_range']['x'] == (-.5, .5)
    assert cfg.commands.base_velocity.rel_standing_envs == 0.0
    assert cfg.commands.base_velocity.class_type.__name__ == 'UniformVelocityCommand'
    terrain_cfg = cfg.scene.terrain.terrain_generator.sub_terrains['bpx_waves'].copy()
    terrain_cfg.size = (8., 8.)
    terrain_cfg.seed = 20260918
    for level in range(8):
        difficulty = (level + .5) / 8
        z = wave_height_field(difficulty, terrain_cfg)
        assert np.isfinite(z).all()
        assert np.abs(z).max() <= .005 * (level + 1) + 1e-10
        # central 2x2 m square, including its boundary, must be exactly level zero
        assert np.all(z[120:201, 120:201] == 0)
        assert np.all(z[[0, -1], :] == 0) and np.all(z[:, [0, -1]] == 0)
        np.testing.assert_array_equal(z, wave_height_field(difficulty, terrain_cfg))
    assert np.isclose(terrain_amplitude(.999, 8, (.005, .04)), .04)
    meshes, origin = wave_terrain(.999, terrain_cfg)
    np.testing.assert_array_equal(origin, [4, 4, 0])
    assert np.all(meshes[0].face_normals[:, 2] > 0)
    print('PASS: geometry, reproducibility, rewards/actions/observations, uniform commands')

    cfg.seed = 42
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    env = gym.make('BPX-Locomotion-Rough-v0', cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs['policy'].shape == (4, 48)
        ids = torch.arange(4, device=env.device)
        terrain = env.scene.terrain
        tracker = env._bpx_rough_traversal
        assert torch.all(terrain.terrain_levels == 0)
        local = env.scene['robot'].data.root_pos_w - env.scene.env_origins
        assert torch.all(local[:, :2].abs() <= .101)
        torch.testing.assert_close(local[:, 2], torch.full((4,), .4, device=env.device))

        def finish(success=True, fallen=False):
            env.episode_length_buf[:] = 500
            tracker.duration[:] = 10
            tracker.rough_time[:] = 2 if success else 0
            tracker.rough_distance[:] = 1.2 if success else 0
            tracker.linear_error[:] = .1
            tracker.angular_error[:] = .1
            env.termination_manager.terminated[:] = fallen
            env._reset_idx(ids)
            assert torch.all(tracker.rough_distance == 0)
            assert torch.all(tracker.rough_time == 0)

        finish()
        assert torch.all(terrain.terrain_levels == 0)
        finish()
        assert torch.all(terrain.terrain_levels == 1)
        finish(success=False)
        assert torch.all(terrain.terrain_levels == 1)  # remaining on flat is not success
        finish(fallen=True)
        assert torch.all(terrain.terrain_levels == 0)  # falling overrides apparent traversal
        terrain.terrain_levels[:] = 7
        terrain.update_env_origins(ids, torch.zeros_like(ids), torch.zeros_like(ids))
        finish()
        finish()
        assert torch.all(terrain.terrain_levels == 7)  # no random wrap at maximum
        # Check counters and bootstrap truncation before a neighbouring tile.
        state = env.scene['robot'].data.root_state_w.clone()
        state[:, :3] = env.scene.env_origins + torch.tensor([3.5, 0, .4], device=env.device)
        env.scene['robot'].write_root_state_to_sim(state, env_ids=ids)
        env.scene.update(env.step_dt)
        env.episode_length_buf[:] = 1
        done = tracker(env, rough_start=1.95, boundary=3.4)
        assert torch.all(done) and torch.all(tracker.rough_distance == 0)
        assert cfg.terminations.terrain_boundary.time_out
        print('PASS: real manager ordering, flat resets, promote/demote/top clamp, boundary truncation')
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback
    import sys
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
else:
    import sys
    sys.stdout.flush()
    app.close()
