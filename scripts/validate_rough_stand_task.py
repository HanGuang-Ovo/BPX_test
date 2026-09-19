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
from BPX_test.tasks.manager_based.bpx_test.mdp.rough_stand import (
    ground_heights, relative_stand_height_l2, reset_rough_stand,
    reset_rough_stand_joints, rough_stand_last_action,
)


def main():
    cfg, flat = BpxRoughStandEnvCfg(), BpxStandEnvCfg()
    observations = cfg.observations.to_dict()
    observations['policy']['actions'] = flat.observations.to_dict()['policy']['actions']
    assert observations == flat.observations.to_dict()
    assert cfg.actions.to_dict() == flat.actions.to_dict()
    assert cfg.commands.to_dict() == flat.commands.to_dict()
    rewards = cfg.rewards.to_dict()
    rewards.pop('left_right_joint_symmetry')
    rewards['base_height_l2'] = flat.rewards.to_dict()['base_height_l2']
    assert rewards == flat.rewards.to_dict()
    assert flat.scene.terrain.terrain_type == 'plane'
    cfg.scene.num_envs = 4
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.events.reset_base.params.update(
        flat_probability=0., replay_probability=0., handoff_probability=0.,
    )
    env = gym.make('BPX-Stand-Rough-v0', cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs['policy'].shape == (4, 48)
        assert torch.all(env.command_manager.get_command('base_velocity') == 0)
        ids = torch.arange(4, device=env.device)
        terrain, tracker = env.scene.terrain, env._bpx_stand_stability
        curriculum = env._bpx_stand_curriculum
        pos = env.scene['robot'].data.root_pos_w
        local = pos - env.scene.env_origins
        assert torch.all(local[:, 0].abs() == 2.5)
        heights = ground_heights(env, pos[:, :2])
        assert torch.all(heights.std(dim=1) > 0)
        torch.testing.assert_close(pos[:, 2], heights.amax(dim=1) + .42)
        torch.testing.assert_close(relative_stand_height_l2(env, .4), (pos[:, 2] - heights.mean(dim=1) - .4).square())
        reset_rough_stand(
            env, ids, flat_probability=1., replay_probability=0., handoff_probability=0.,
        )
        env.scene.update(env.step_dt)
        assert not tracker.rough.any()
        assert torch.all(ground_heights(env, tracker.spawn_xy).abs() < 1e-6)
        reset_rough_stand(
            env, ids, flat_probability=0., replay_probability=0., handoff_probability=1.,
        )
        reset_rough_stand_joints(env, ids)
        env.episode_length_buf[:] = 0
        assert tracker.rough.all() and tracker.adaptive.all() and tracker.handoff.all()
        assert torch.all(rough_stand_last_action(env).abs().sum(dim=1) > 0)

        curriculum.mastery_level[:] = torch.tensor([0, 1, 4, 7], device=env.device)
        reset_rough_stand(
            env, ids, flat_probability=0., replay_probability=1., handoff_probability=0.,
        )
        assert not tracker.adaptive.any()
        assert torch.all(tracker.sampled_level <= curriculum.mastery_level)
        assert torch.all(tracker.sampled_level[curriculum.mastery_level > 0]
                         < curriculum.mastery_level[curriculum.mastery_level > 0])
        curriculum.mastery_level[:] = 0
        reset_rough_stand(
            env, ids, flat_probability=0., replay_probability=0., handoff_probability=0.,
        )

        def finish(
            stable=True, contact=True, symmetric=True, fallen=False,
            adaptive=True, duration=14.5,
        ):
            env.episode_length_buf[:] = 750
            tracker.duration[:] = duration
            tracker.stable_time[:] = duration if stable else duration * .5
            tracker.four_feet_time[:] = duration if contact else duration * .5
            tracker.symmetry_time[:] = duration if symmetric else duration * .5
            tracker.symmetry_error_sum[:] = duration * (.05 if symmetric else .30)
            tracker.rough[:] = adaptive
            tracker.adaptive[:] = adaptive
            env.termination_manager.terminated[:] = fallen
            env._reset_idx(ids)
            assert torch.all(tracker.duration == 0) and torch.all(tracker.four_feet_time == 0)
            assert torch.all(tracker.symmetry_time == 0) and torch.all(tracker.symmetry_error_sum == 0)
        finish()
        assert torch.all(curriculum.mastery_level == 0)
        finish()
        assert torch.all(curriculum.mastery_level == 1)
        finish(adaptive=False)
        finish(adaptive=False)
        assert torch.all(curriculum.mastery_level == 1)
        finish(fallen=True)
        assert torch.all(curriculum.mastery_level == 0)
        finish(duration=1.)
        finish()
        assert torch.all(curriculum.mastery_level == 0)
        finish()
        assert torch.all(curriculum.mastery_level == 1)
        finish(stable=False)
        assert torch.all(curriculum.mastery_level == 0)
        finish()
        finish()
        assert torch.all(curriculum.mastery_level == 1)
        finish(contact=False)
        assert torch.all(curriculum.mastery_level == 0)
        finish()
        finish()
        assert torch.all(curriculum.mastery_level == 1)
        finish(symmetric=False)
        assert torch.all(curriculum.mastery_level == 0)
        curriculum.mastery_level[:] = 7
        finish()
        finish()
        assert torch.all(curriculum.mastery_level == 7)
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
        print(
            'PASS: 48-D interface, stratified/handoff reset, four-foot and symmetry '
            'curriculum, relative height, drift failure, physics steps',
            flush=True,
        )
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
