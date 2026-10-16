"""Validate mixed stair geometry, real resets, checkpoint transfer and PPO updates.

Run in Isaac Lab with --finetune CHECKPOINT --headless --device cpu.
"""

import argparse
from pathlib import Path
import sys
import tempfile
import traceback
from unittest.mock import patch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--finetune", required=True, help="Checkpoint to validate weight transfer from.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

import BPX_test.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from BPX_test.tasks.manager_based.bpx_test.agents.bpx_stairs_history_ppo_cfg import BpxStairsHistoryPPORunnerCfg
from BPX_test.tasks.manager_based.bpx_test.bpx_rough_history_env_cfg import BpxRoughHistoryEnvCfg
from BPX_test.tasks.manager_based.bpx_test.bpx_stairs_history_env_cfg import BpxStairsHistoryEnvCfg

sys.path.insert(0, str(Path(__file__).resolve().parent / "rsl_rl"))
from curriculum_runner import CurriculumOnPolicyRunner


def main():
    baseline = BpxRoughHistoryEnvCfg()
    cfg = BpxStairsHistoryEnvCfg()
    for name in ("observations", "actions", "events"):
        assert getattr(cfg, name).to_dict() == getattr(baseline, name).to_dict(), name
    assert cfg.episode_length_s == 40
    assert cfg.commands.base_velocity.resampling_time_range == (40., 40.)
    assert cfg.commands.base_velocity.traversal_probabilities == (.3, .3, .4)
    assert cfg.commands.base_velocity.initial_stage == 0
    assert cfg.scene.robot.to_dict() == baseline.scene.robot.to_dict()
    assert cfg.sim.to_dict() == baseline.sim.to_dict()
    generator = cfg.scene.terrain.terrain_generator
    for name in ("up_stairs", "down_stairs"):
        sub = generator.sub_terrains[name].copy()
        sub.size = generator.size
        assert sub.step_width == .2
        for difficulty in (0., .124999, .5, .875, 1.):
            meshes, origin = sub.function(difficulty, sub)
            assert len(meshes) == 45
            np.testing.assert_allclose(meshes[-1].extents[:2], [2., 2.])
            np.testing.assert_allclose(meshes[-1].bounds[1, 2], origin[2])
            height = abs(origin[2]) / 11
            assert .019999 <= height <= .150001
            if name == "down_stairs":
                assert origin[2] > 0
                if difficulty < .125:
                    assert .03 - 1.e-8 <= height <= .05 + 1.e-8
                if difficulty >= .875:
                    assert .13 - 1.e-8 <= height <= .15 + 1.e-8
            else:
                assert origin[2] < 0
            assert all(mesh.is_watertight for mesh in meshes)
            for k in range(10):
                np.testing.assert_allclose(meshes[4 + 4*k].extents[1], .2)
    print("PASS: 450/480 interfaces preserved, 20-cm stairs, 3–5 cm first descent row and wide landings", flush=True)

    cfg.seed = 42
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    env = gym.make("BPX-Locomotion-Stairs-History-v0", cfg=cfg).unwrapped
    wrapped = None
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (4, 450)
        assert obs["critic"].shape == (4, 480)
        terrain = env.scene.terrain
        # Check native column allocation and actual signed central elevations.
        assert torch.all(terrain.terrain_origins[:, :2, 2] < 0)
        assert torch.all(terrain.terrain_origins[:, 2:4, 2] > 0)
        assert torch.all(terrain.terrain_origins[:, 4:, 2] == 0)
        assert torch.all((terrain.terrain_origins[0, 2:4, 2] / 11 >= .03)
                         & (terrain.terrain_origins[0, 2:4, 2] / 11 <= .05))
        command = env.command_manager.get_term("base_velocity")
        tracker = env._bpx_rough_traversal
        course = env.curriculum_manager._term_cfgs[0].func
        ids = torch.arange(4, device=env.device)
        course.levels[:, 0] = 5
        course.levels[:, 1] = 0
        kinds = torch.tensor([0, 1, 2, 1], device=env.device)
        original_prepare = command.prepare_next_episode

        def prepare(selected):
            original_prepare(selected)
            command.next_traversal[selected] = kinds[selected]
            command.next_mode[selected] = 0

        env.episode_length_buf[:] = 0
        with patch.object(command, "prepare_next_episode", side_effect=prepare):
            env._reset_idx(ids)
        torch.testing.assert_close(command.episode_traversal, kinds)
        torch.testing.assert_close(terrain.terrain_types // 2, kinds)
        torch.testing.assert_close(terrain.terrain_levels, torch.tensor([5, 0, 0, 0], device=env.device))
        torch.testing.assert_close(env.scene.env_origins,
                                   terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
        local = env.scene["robot"].data.root_pos_w - env.scene.env_origins
        torch.testing.assert_close(local[:, 2], torch.full((4,), .4, device=env.device))
        down = kinds == 1
        assert torch.all((command.command[down, 0] >= .15) & (command.command[down, 0] <= .30))
        assert torch.all(command.command[down, 1:] == 0)
        print("PASS: real high-platform descent spawn, independent up/down levels, flat resets and low-speed commands", flush=True)

        # Exercise old-episode accounting through the real curriculum/events/
        # command reset order; next episode changes kind and restored difficulty.
        env.episode_length_buf[:] = 500
        tracker.duration[:] = 10
        tracker.rough_time[:] = 3
        tracker.linear_error[:] = .5
        tracker.angular_error[:] = .5
        tracker.along_command_integral[:] = 2
        tracker.command_speed_integral[:] = 2.5
        tracker.speed_integral[:] = 2
        tracker.settled[:] = True
        # Flat completion now survives a brief terminal excursion. Feed its
        # episode quality through real manager reset/logging, not final dwell.
        tracker.settled[command.episode_traversal == 2] = False
        tracker.flat_quality_time[:] = 9
        tracker.flat_stable_time[:] = 8.9
        tracker.flat_max_unstable_time[:] = .04
        env.termination_manager.terminated[:] = False
        env.termination_manager.get_term("time_out")[:] = False
        kinds[:] = torch.tensor([1, 0, 2, 0], device=env.device)
        with patch.object(command, "prepare_next_episode", side_effect=prepare):
            env._reset_idx(ids)
        assert env.extras["log"]["Curriculum/terrain_levels/down/success_rate"] == 1
        assert env.extras["log"]["Curriculum/terrain_levels/down/terrain_level"] == 0
        assert env.extras["log"]["Curriculum/terrain_levels/up/terrain_level"] == 5
        assert env.extras["log"]["Curriculum/terrain_levels/flat/success_rate"] == 1
        assert env.extras["log"]["Curriculum/terrain_levels/flat/stable_fraction"] > .98
        assert torch.all(tracker.landing_time == 0)
        assert torch.all(tracker.settled == 0)
        for name in ("flat_quality_time", "flat_stable_time", "flat_unstable_time", "flat_max_unstable_time"):
            assert torch.all(getattr(tracker, name) == 0)
        print("PASS: original-terrain accounting, flat episode quality despite terminal excursion, buffer resets", flush=True)

        # Partial reset batches containing flat episodes must combine id-indexed
        # tensors with each other, not with full-size command tensors (this
        # crashed real 1024-env runs on the first partial reset batch).
        moving_before = int(course.flat_attempts[0])
        standing_before = int(course.flat_attempts[1])
        env.episode_length_buf[:] = 0
        env.episode_length_buf[torch.tensor([1, 3], device=env.device)] = 500
        command.episode_traversal[:] = torch.tensor([0, 2, 0, 2], device=env.device)
        command.episode_standing[:] = False
        command.episode_standing[3] = True
        kinds[:] = torch.tensor([0, 2, 0, 2], device=env.device)  # next episode stays flat
        with patch.object(command, "prepare_next_episode", side_effect=prepare):
            env._reset_idx(torch.tensor([1, 3], device=env.device))
        assert int(course.flat_attempts[0]) == moving_before + 1
        assert int(course.flat_attempts[1]) == standing_before + 1
        print("PASS: partial reset batch with moving and standing flat episodes feeds the gate", flush=True)

        for _ in range(90):
            obs, reward, _, _, _ = env.step(torch.zeros(4, 12, device=env.device))
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
        print("PASS: 90 real CPU physics/control steps with finite rewards and observations", flush=True)

        agent = BpxStairsHistoryPPORunnerCfg()
        assert agent.algorithm.learning_rate == 3.0e-4
        assert agent.algorithm.schedule == "adaptive"
        agent.device = args.device
        agent.num_steps_per_env = 24
        wrapped = RslRlVecEnvWrapper(env)
        with tempfile.TemporaryDirectory(prefix="bpx_mixed_validation_") as temp:
            runner = CurriculumOnPolicyRunner(wrapped, agent.to_dict(), log_dir=temp, device=args.device)
            runner.load(args.finetune, load_optimizer=False)
            runner.current_learning_iteration = 0
            saved = torch.load(args.finetune, map_location=args.device, weights_only=False)
            for name, value in runner.alg.policy.state_dict().items():
                torch.testing.assert_close(value, saved["model_state_dict"][name], rtol=0, atol=0)
            assert len(runner.alg.optimizer.state) == 0
            assert runner.alg.optimizer.param_groups[0]["lr"] == 3.0e-4
            obs, _ = wrapped.get_observations()
            assert torch.isfinite(runner.get_inference_policy()(obs)).all()
            print("PASS: exact checkpoint weight transfer, fresh optimizer and 3e-4 learning rate", flush=True)
            runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
            # Verify the actual TensorBoard file, not only the writer proxy.
            import re
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            runner.writer.flush()
            events = EventAccumulator(temp)
            events.Reload()
            scalar_tags = events.Tags()["scalars"]
            assert scalar_tags and len(scalar_tags) <= 26
            assert all(re.fullmatch(agent.tensorboard_scalar_filter, tag) for tag in scalar_tags)
            assert {"Loss/value_function", "Loss/learning_rate", "Policy/mean_noise_std"} <= set(scalar_tags)
            assert "Perf/total_fps" not in scalar_tags
            print(f"PASS: real TensorBoard writer keeps only core scalars ({len(scalar_tags)} tags)", flush=True)
            assert len(runner.alg.optimizer.state) > 0
            assert all(torch.isfinite(value).all() for value in runner.alg.policy.state_dict().values())
            course.levels[:, 0] = 6
            course.levels[:, 1, 0] = 2
            # The adaptive schedule may move the learning rate during learning;
            # resume must restore whatever rate the checkpoint saved.
            expected_lr = runner.alg.optimizer.param_groups[0]["lr"]
            checkpoint = str(Path(temp) / "resume_test.pt")
            runner.save(checkpoint)
            state = course.state_dict()
            course.levels[:] = 0
            runner.load(checkpoint)
            torch.testing.assert_close(course.levels, state["levels"])
            torch.testing.assert_close(course.counts, state["counts"])
            assert torch.all(command.command[command.episode_traversal == 1, 0] >= .15)
            assert runner.alg.optimizer.param_groups[0]["lr"] == expected_lr
            print("PASS: two real PPO updates and checkpoint resume restores up/down curriculum without spurious episodes", flush=True)
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()
