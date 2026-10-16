"""CPU regression tests for descent, skill retention and settled landing gates."""

import ast
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

TASK_DIR = Path(__file__).resolve().parents[1] / "source/BPX_test/BPX_test/tasks/manager_based/bpx_test"


class FakeTerrain:
    def __init__(self, count):
        self.terrain_levels = torch.zeros(count, dtype=torch.long)
        self.max_terrain_level = 8

    def update_env_origins(self, ids, up, down):
        self.terrain_levels[ids] = (self.terrain_levels[ids] + up.long() - down.long()).clamp(0, 7)


class FakeScene(dict):
    def __init__(self, count):
        super().__init__()
        self.terrain = FakeTerrain(count)
        self.env_origins = torch.zeros(count, 3)
        self.env_origins[:, 2] = -0.6
        default = torch.zeros(count, 13)
        default[:, 2] = 0.4
        self["robot"] = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor([[3.3, 0.0, 0.4]]).repeat(count, 1),
            default_root_state=default,
            root_lin_vel_b=torch.zeros(count, 3), root_ang_vel_b=torch.zeros(count, 3),
        ))



def command_cfg(stage=0):
    return SimpleNamespace(
        heading_command=False, rel_standing_envs=0., initial_stage=stage,
        initial_sideward_ramp=0,
        ranges=SimpleNamespace(lin_vel_x=(.3, .6)),
        backward_speed_range=(.2, .4))





def load_modules():
    managers = ModuleType("isaaclab.managers")
    managers.ManagerTermBase = type("Base", (), {"__init__": lambda self, cfg, env: None})
    managers.SceneEntityCfg = type("EntityCfg", (), {})
    mdp = ModuleType("isaaclab.envs.mdp")

    class UniformCommand:
        def __init__(self, cfg, env):
            self.cfg, self.device = cfg, env.device
            self.vel_command_b = torch.zeros(env.num_envs, 3)

        def _resample_command(self, ids):
            self.vel_command_b[ids, 0] = torch.empty(len(ids)).uniform_(*self.cfg.ranges.lin_vel_x)

    mdp.UniformVelocityCommand = UniformCommand
    mdp.UniformVelocityCommandCfg = type("Cfg", (), {})
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    package = ModuleType("mixed_test_mdp")
    package.__path__ = [str(TASK_DIR / "mdp")]
    with patch.dict(sys.modules, {"isaaclab.managers": managers, "isaaclab.envs.mdp": mdp,
                                  "isaaclab.utils": utils, "mixed_test_mdp": package}):
        command_module = importlib.import_module("mixed_test_mdp.mixed_stairs_commands")
        course_module = importlib.import_module("mixed_test_mdp.mixed_stairs_curriculum")
    return command_module, course_module


def mixed_cfg():
    cfg = command_cfg(1)
    cfg.initial_sideward_ramp = 0
    cfg.traversal_probabilities = (.4, .4, .2)
    cfg.descent_initial_speed_range = (.15, .30)
    cfg.descent_final_speed_range = (.30, .60)
    cfg.rel_standing_envs = .0
    cfg.rel_flat_standing_envs = .10
    cfg.rel_pure_rotation_envs = .25
    cfg.pure_rotation_speed_range = (.2, 1.0)
    cfg.flat_lin_vel_x = (-1.0, 1.0)
    cfg.flat_lin_vel_y = (-0.5, 0.5)
    cfg.flat_ang_vel_z = (-1.0, 1.0)
    return cfg


class MixedTests(unittest.TestCase):
    def setUp(self):
        self.commands, self.module = load_modules()
        self.env = SimpleNamespace(num_envs=4, device="cpu", step_dt=.02,
                                   episode_length_buf=torch.zeros(4, dtype=torch.long), scene=FakeScene(4))
        self.command = self.commands.MixedStairVelocityCommand(mixed_cfg(), self.env)
        self.env.command_manager = SimpleNamespace(get_term=lambda name: self.command,
                                                   get_command=lambda name: self.command.vel_command_b)
        self.env.scene.terrain.terrain_types = torch.zeros(4, dtype=torch.long)
        self.timeout = torch.zeros(4, dtype=torch.bool)
        self.env.termination_manager = SimpleNamespace(terminated=torch.zeros(4, dtype=torch.bool),
                                                      get_term=lambda name: self.timeout)
        self.params = dict(initial_up_levels=(3, 7), statistics_window=16,
                           successes_to_promote=2, failures_to_demote=2, min_stair_time=1.,
                           max_linear_error=.35, max_descent_linear_error=.10,
                           max_angular_error=.5, min_speed_ratio=.6)
        self.course = self.module.MixedStairCurriculum(SimpleNamespace(params=self.params), self.env)
        tracker = self.module.MixedStairTraversalBoundary(None, self.env)
        tracker.duration[:] = 10
        tracker.rough_time[:] = 3
        tracker.along_command_integral[:] = 2
        tracker.command_speed_integral[:] = 2.5
        tracker.speed_integral[:] = 2
        tracker.settled[:] = True
        tracker.flat_quality_time[:] = 10
        tracker.flat_stable_time[:] = 10
        self.tracker = tracker

    def step(self, ids=None):
        return self.course(self.env, ids, **self.params)

    def test_episode_mix_forward_backward_stairs_and_low_speed(self):
        torch.manual_seed(42)
        env = SimpleNamespace(num_envs=20000, device="cpu")
        command = self.commands.MixedStairVelocityCommand(mixed_cfg(), env)
        ids = torch.arange(env.num_envs)
        command.prepare_next_episode(ids)
        command.next_descent_speed_range[:] = torch.tensor([.15, .30])
        command._resample_command(ids)
        for kind, expected in enumerate((.4, .4, .2)):
            self.assertAlmostEqual((command.episode_traversal == kind).float().mean().item(), expected, delta=.015)
        down = command.episode_traversal == 1
        self.assertEqual(set(command.episode_mode[down].tolist()), {0, 2})
        self.assertTrue(torch.all(command.vel_command_b[down, 0].abs() >= .15))
        self.assertTrue(torch.all(command.vel_command_b[down, 0].abs() <= .30))
        self.assertTrue(torch.all(command.vel_command_b[down, 1:] == 0))
        for kind in (0, 1):
            mask = command.episode_traversal == kind
            self.assertEqual(set(command.episode_mode[mask].tolist()), {0, 2})
            self.assertTrue(torch.all(command.vel_command_b[mask, 1:] == 0))
            self.assertTrue(torch.all(command.vel_command_b[mask & (command.episode_mode == 2), 0] < 0))
        self.assertEqual(set(command.episode_mode[command.episode_traversal == 2].tolist()), {0, 1, 2, 3})
        old = command.vel_command_b.clone()
        command._resample_command(ids)
        torch.testing.assert_close(command.vel_command_b, old)

    def test_reset_selects_columns_and_levels_before_command_consumption(self):
        self.step()
        pending = self.command.next_traversal
        terrain = self.env.scene.terrain
        torch.testing.assert_close(terrain.terrain_types // 2, pending)
        self.assertTrue(torch.all(terrain.terrain_levels[pending == 1] == 0))
        self.assertTrue(torch.all(terrain.terrain_levels[pending == 2] == 0))
        if torch.any(pending == 0):
            self.assertTrue(torch.all(terrain.terrain_levels[pending == 0] >= 3))
        self.command._resample_command(torch.arange(4))
        torch.testing.assert_close(self.command.episode_traversal, pending)
        self.assertEqual(self.course.counts.sum(), 0)

    def completed(self, kind, success=True, level=0):
        self.env.episode_length_buf[:] = 500
        self.command.episode_traversal[:] = kind
        self.command.episode_mode[:] = 0
        self.env.scene.terrain.terrain_levels[:] = level
        self.tracker.settled[:] = success
        self.tracker.flat_quality_time[:] = 10
        self.tracker.flat_stable_time[:] = 10 if success else 0

    def test_down_promotes_independently_and_statistics_use_old_episode(self):
        up = self.course.levels[:, 0].clone()
        self.completed(1)
        logs = self.step()
        self.completed(1)
        self.step()
        self.assertTrue(torch.all(self.course.levels[:, 1, 0] == 1))
        torch.testing.assert_close(self.course.levels[:, 0], up)
        self.assertEqual(logs["down/success_rate"], 1)
        self.assertEqual(logs["up/window_samples"], 0)
        self.assertEqual(logs["down/forward/actual_speed"], .2)
        self.assertTrue(torch.all(self.course.levels[:, 2] == 0))

    def test_airborne_boundary_does_not_promote_and_demotes_after_two(self):
        self.completed(1, success=False, level=4)
        self.step()
        self.completed(1, success=False, level=4)
        logs = self.step()
        self.assertTrue(torch.all(self.course.levels[:, 1, 0] == 3))
        self.assertEqual(logs["down/success_rate"], 0)

    def test_low_speed_stalling_cannot_validate_descent(self):
        self.completed(1)
        self.tracker.along_command_integral[:] = 0
        self.step()
        self.assertTrue(torch.all(self.course.success_streak[:, 1] == 0))
        self.tracker.along_command_integral[:] = 2
        self.tracker.linear_error[:] = 1.1  # .11 m/s is too large in .15–.30 range.
        self.completed(1)
        logs = self.step()
        self.assertEqual(logs["down/success_rate"], 0)

    def test_top_clamps_flat_stays_zero_subset_and_state_restore(self):
        self.completed(1, level=7)
        self.step([0, 2])
        self.completed(1, level=7)
        self.step([0, 2])
        torch.testing.assert_close(self.course.levels[:, 1, 0], torch.tensor([7, 0, 7, 0]))
        self.completed(2, level=0)
        self.step()
        self.completed(2, level=0)
        self.step()
        self.assertTrue(torch.all(self.course.levels[:, 2] == 0))
        saved = self.course.state_dict()
        self.course.levels[:] = 0
        self.course.load_state_dict(saved)
        torch.testing.assert_close(self.course.levels, saved["levels"])
        self.step([])
        saved["levels"] = torch.zeros(2, 3, 4)
        with self.assertRaisesRegex(ValueError, "num_envs"):
            self.course.load_state_dict(saved)

    def test_scratch_direction_gate_requires_each_terrain_and_fresh_phase(self):
        self.command.stage = 0
        self.command.sideward_ramp = 0
        self.command.episode_stage[:] = 0
        self.command.episode_sideward_ramp[:] = 0
        self.env.scene.terrain.terrain_levels[:] = 1
        ids = torch.arange(4)
        success = torch.ones(4, dtype=torch.bool)
        def record(kind):
            self.command.episode_traversal[:] = kind
            for mode in (range(4) if kind == 2 else [0]):
                self.command.episode_mode[:] = mode
                self.course._update_command_stage(self.env, ids, success, 4, .8, 2, 1)
        record(0)
        record(2)
        self.assertEqual(self.command.stage, 0)
        record(1)
        self.assertEqual(self.command.stage, 0)
        for kind in (0, 1, 2):
            record(kind)
        self.assertEqual(self.command.stage, 1)
        self.assertEqual(self.command.sideward_ramp, 0)
        record(0)  # Running episodes from stage zero cannot validate stage one.
        self.assertEqual(self.course.command_attempts.sum(), 0)
        saved = self.course.state_dict()
        self.command.stage = 3
        self.course.load_state_dict(saved)
        self.assertEqual(self.command.stage, 1)
        self.assertEqual(self.command.sideward_ramp, 0)

    def test_forward_backward_is_final_stage(self):
        self.command.stage = 1
        self.command.sideward_ramp = 0
        self.command.episode_stage[:] = 1
        ids = torch.arange(4)
        self.env.scene.terrain.terrain_levels[:] = 1
        for _ in range(4):
            for kind in range(3):
                for mode in (range(4) if kind == 2 else [0, 2]):
                    self.command.episode_traversal[:] = kind
                    self.command.episode_mode[:] = mode
                    self.course._update_command_stage(
                        self.env, ids, torch.ones(4, dtype=torch.bool), 4, .8, 2, 1)
        self.assertEqual((self.command.stage, self.command.sideward_ramp), (1, 0))

    def test_backward_descent_has_independent_level(self):
        self.completed(1, level=2)
        self.command.episode_mode[:] = 2
        self.step()
        self.completed(1, level=2)
        self.command.episode_mode[:] = 2
        self.step()
        self.assertTrue(torch.all(self.course.levels[:, 1, 2] == 3))
        self.assertTrue(torch.all(self.course.levels[:, 1, 0] == 0))

    def test_legacy_curriculum_requires_finetune(self):
        state = self.course.state_dict()
        del state["schema_version"]
        with self.assertRaisesRegex(ValueError, "--finetune"):
            self.course.load_state_dict(state)

    def test_v2_resume_preserves_progress_but_rebuilds_changed_quality_windows(self):
        self.course.levels[:, 0, 0] = 5
        self.course.levels[:, 1, 0] = 3
        self.course.history[:] = .25
        self.course.counts[:] = 12
        self.course.cursors[:] = 4
        self.course.success_streak[:] = 1
        self.course.failure_streak[:] = 1
        self.course.command_attempts[:] = 50
        self.course.command_successes[:] = 40
        self.course.flat_attempts[:] = 50
        self.course.flat_successes[:] = 40
        self.course.flat_good_windows.fill_(1)
        self.course.command_phase[:] = torch.tensor([1, 0, 1])
        self.course.sampling_phase.fill_(1)
        old = self.course.state_dict()
        old["schema_version"].fill_(2)
        old["history"] = old["history"][..., :9].clone()
        original = {name: value.clone() for name, value in old.items()}
        with self.assertWarnsRegex(UserWarning, "v2.*v3"):
            self.course.load_state_dict(old)
        self.assertEqual(int(self.course.schema_version), 3)
        torch.testing.assert_close(self.course.levels, old["levels"])
        torch.testing.assert_close(self.course.history[:2, :, :, :9], old["history"][:2])
        self.assertFalse(self.course.history[2].any())
        self.assertTrue(torch.all(self.course.counts[:2] == 12))
        self.assertFalse(self.course.counts[2].any())
        self.assertFalse(self.course.cursors[2].any())
        self.assertTrue(torch.all(self.course.success_streak[:, :2] == 1))
        self.assertFalse(self.course.success_streak[:, 2].any())
        self.assertFalse(self.course.failure_streak[:, 2].any())
        for name in ("command_attempts", "command_successes", "flat_attempts",
                     "flat_successes", "flat_good_windows"):
            self.assertFalse(getattr(self.course, name).any(), name)
        self.assertEqual(self.command.stage, 1)
        self.assertEqual(self.command.traversal_probabilities, (.4, .4, .2))
        self.assertEqual(int(self.course.command_phase[2]), 0)
        for name in old:
            torch.testing.assert_close(old[name], original[name])
        # The migrated v3 state must itself round-trip without clearing new data.
        self.course.counts[2] = 5
        saved = self.course.state_dict()
        self.course.load_state_dict(saved)
        torch.testing.assert_close(self.course.counts, saved["counts"])

    def test_invalid_resume_does_not_partially_mutate_curriculum(self):
        before = self.course.state_dict()
        broken = self.course.state_dict()
        broken["schema_version"].fill_(2)
        broken["history"] = broken["history"][..., :9].clone()
        broken["counts"] = torch.zeros((2, 4), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "形状不匹配"):
            self.course.load_state_dict(broken)
        for name, value in self.course.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_scratch_starts_low_and_forward_on_all_terrains(self):
        self.command.stage = 0
        self.command.sideward_ramp = 0
        params = {**self.params, "initial_up_levels": (0, 0)}
        self.course = self.module.MixedStairCurriculum(SimpleNamespace(params=params), self.env)
        self.course(self.env, None, **params)
        self.assertTrue(torch.all(self.env.scene.terrain.terrain_levels == 0))
        self.assertTrue(torch.all(self.command.next_mode == 0))

    def test_partial_reset_batch_with_flat_episode_updates_flat_gate(self):
        # Real training resets a subset of environments; kinds/flags indexed by
        # those ids must combine safely with each other, not with full-size
        # command tensors (1024-env runs crashed on the first partial batch).
        self.env.episode_length_buf[:] = 500
        self.command.episode_traversal[:] = 2
        self.command.episode_mode[:] = 0
        self.command.episode_standing[:] = torch.tensor([True, False, True, False])
        self.env.scene.terrain.terrain_levels[:] = 0
        self.tracker.settled[:] = True
        self.step([1, 3])
        self.assertEqual(int(self.course.flat_attempts[0]), 2)  # two moving flat episodes
        self.assertEqual(int(self.course.flat_attempts[1]), 0)
        self.assertEqual(int(self.course.flat_successes[0]), 2)

    def test_flat_gate_needs_both_categories_two_windows_and_restores(self):
        command, course = self.command, self.course
        command.set_traversal_probabilities((.3, .3, .4))
        kinds = torch.full((4,), 2, dtype=torch.long)
        success = torch.ones(4, dtype=torch.bool)
        moving_fails = torch.tensor([False, False, True, True])
        command.episode_standing[:] = torch.tensor([True, True, False, False])

        def window(outcome=success):
            course._update_flat_sampling(kinds, outcome, command.episode_standing, 128, .9, 2)

        for _ in range(64):
            window()
        self.assertEqual(int(course.flat_good_windows), 1)
        # Standing alone is not enough: a window with failing motion resets the streak.
        for _ in range(64):
            window(moving_fails)
        self.assertEqual(int(course.flat_good_windows), 0)
        self.assertEqual(course.sampling_phase.item(), 0)
        self.assertEqual(command.traversal_probabilities, (.3, .3, .4))
        # Two consecutive qualified windows promote the sampling ratio.
        for _ in range(128):
            window()
        self.assertEqual(course.sampling_phase.item(), 1)
        self.assertEqual(command.traversal_probabilities, (.4, .4, .2))
        # Checkpoint restore reinstalls the ratio that matches the phase.
        course.load_state_dict(course.state_dict())
        self.assertEqual(command.traversal_probabilities, (.4, .4, .2))
        course.sampling_phase.fill_(0)
        course.load_state_dict(course.state_dict())
        self.assertEqual(command.traversal_probabilities, (.3, .3, .4))

    def test_flat_stair_episodes_never_inherit_standing_mask(self):
        command = self.command
        ids = torch.arange(4)
        command.prepare_next_episode(ids)
        command.next_traversal[:] = torch.tensor([0, 1, 2, 2])
        command.is_standing_env[:] = True  # Simulate a stale base-class mask.
        command.cfg.rel_flat_standing_envs = 1.0
        command.cfg.rel_pure_rotation_envs = 0.0
        command._resample_command(ids)
        # Stair episodes never keep the standing mask; every flat episode stands.
        self.assertTrue(torch.all(command.is_standing_env[:2] == False))
        self.assertTrue(torch.all(command.episode_standing[:2] == False))
        self.assertTrue(torch.all(command.is_standing_env[2:] == True))
        self.assertTrue(torch.all(command.episode_standing[2:] == True))

    def landing_fixture(self):
        robot = self.env.scene["robot"].data
        self.env.scene.env_origins[:] = 0
        robot.root_pos_w[:] = torch.tensor([3.6, 0, .4])
        robot.body_pos_w = torch.tensor([[[3.35, -.2, .025], [3.85, .2, .025],
                                         [3.35, .2, .025], [3.85, -.2, .025]]]).repeat(4, 1, 1)
        robot.projected_gravity_b = torch.tensor([[0., 0., -1.]]).repeat(4, 1)
        robot.root_lin_vel_w = torch.zeros(4, 3)
        forces = torch.zeros(4, 4, 3)
        forces[:, :2, 2] = 40
        self.env.scene.sensors = {"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))}
        entity = SimpleNamespace(name="contact_forces", body_ids=slice(None))
        self.landing_params = dict(rough_start=1., boundary=4.5, completion_radius=3.5,
                                   landing_duration=.4, height_tolerance=.15, max_tilt=.35,
                                   max_vertical_speed=.2, max_angular_speed=1., contact_force=5.,
                                   sensor_cfg=entity, asset_cfg=entity)
        self.tracker.reset()
        return robot, forces

    def test_landing_needs_continuous_supported_dwell_not_airborne_crossing(self):
        robot, forces = self.landing_fixture()
        for _ in range(19):
            self.assertFalse(self.tracker(self.env, **self.landing_params).any())
        self.assertTrue(self.tracker(self.env, **self.landing_params).all())
        forces[:] = 0
        self.assertFalse(self.tracker(self.env, **self.landing_params).any())
        self.assertTrue(torch.all(self.tracker.landing_time == 0))
        robot.root_pos_w[:, 0] = 4.6
        self.assertTrue(self.tracker(self.env, **self.landing_params).all())
        self.assertFalse(self.tracker.settled.any())

    def test_landing_rejects_straddling_last_step_tilt_impact_and_bad_height(self):
        robot, _ = self.landing_fixture()
        robot.body_pos_w[0, 0, 0] = 2.9
        robot.projected_gravity_b[1, 2] = -.8
        robot.root_lin_vel_w[2, 2] = -.8
        robot.root_pos_w[3, 2] = .8
        for _ in range(30):
            self.assertFalse(self.tracker(self.env, **self.landing_params).any())

    def test_flat_stability_does_not_end_episode_but_boundary_does(self):
        robot, _ = self.landing_fixture()
        self.command.episode_traversal[:] = 2
        robot.root_pos_w[:, :2] = 0
        # A moving command is deliberately not tracked: still allow time to learn.
        self.command.vel_command_b[:] = torch.tensor([.6, 0., 0.])
        for _ in range(200):
            self.assertFalse(self.tracker(self.env, **self.landing_params).any())
        self.assertTrue(self.tracker.settled.all())
        robot.root_pos_w[:, 0] = 4.6
        self.assertTrue(self.tracker(self.env, **self.landing_params).all())

    def test_flat_quality_tolerates_brief_terminal_excursion_not_sustained_or_frequent_instability(self):
        robot, _ = self.landing_fixture()
        self.command.episode_traversal[:] = 2
        self.command.vel_command_b[:] = torch.tensor([.4, 0., 0.])
        robot.root_lin_vel_b[:, 0] = .4
        robot.root_pos_w[:, :2] = 0
        for step in range(200):
            robot.root_lin_vel_w[:, 2] = 0
            if step >= 198:  # Just two terminal samples exceed a quality limit.
                robot.root_lin_vel_w[0, 2] = .21
            if step >= 170:  # Longest interruption .6s, but still 80% stable.
                robot.root_lin_vel_w[1, 2] = .21
            if step % 5 == 0:  # Frequent short interruptions: only 80% stable.
                robot.root_lin_vel_w[2, 2] = .21
            if step < 50:  # Reset transients do not count against quality.
                robot.root_lin_vel_w[3, 2] = .8
            self.assertFalse(self.tracker(self.env, **self.landing_params).any())
        self.assertFalse(self.tracker.settled[:3].any())
        self.env.episode_length_buf[:] = 200
        logs = self.step()
        self.assertEqual(float(logs["flat/success_rate"]), .5)
        self.assertEqual(int(self.course.flat_successes[0]), 2)
        torch.testing.assert_close(self.course.success_streak[:, 2, 0], torch.tensor([1, 0, 0, 1]))
        self.assertAlmostEqual(float(logs["flat/quality_time"]), 3., places=4)
        self.assertIn("flat/stable_fraction", logs)
        self.assertNotIn("up/stable_fraction", logs)

    def test_flat_rejects_long_interruption_even_with_high_overall_stability(self):
        robot, _ = self.landing_fixture()
        self.command.episode_traversal[:] = 2
        self.command.vel_command_b[:] = torch.tensor([.4, 0., 0.])
        robot.root_lin_vel_b[:, 0] = .4
        robot.root_pos_w[:, :2] = 0
        for step in range(550):
            robot.root_lin_vel_w[:, 2] = .21 if 250 <= step < 280 else 0
            self.tracker(self.env, **self.landing_params)
        self.env.episode_length_buf[:] = 550
        self.assertTrue(torch.all(self.tracker.flat_stable_time / self.tracker.flat_quality_time > .9))
        self.assertEqual(float(self.step()["flat/success_rate"]), 0)

    def test_flat_minimum_observation_tracking_progress_and_physical_failure_still_required(self):
        self.completed(2)
        self.tracker.flat_quality_time[0] = 1.9
        self.tracker.flat_stable_time[0] = 1.9
        self.tracker.along_command_integral[1] = 0
        self.tracker.linear_error[2] = 4  # .4m/s mean error > .35.
        self.env.termination_manager.terminated[3] = True
        self.assertEqual(float(self.step()["flat/success_rate"]), 0)

    def test_flat_quality_threshold_boundaries(self):
        self.completed(2)
        self.tracker.flat_quality_time[:] = torch.tensor([10., 10., 10., 2.])
        self.tracker.flat_stable_time[:] = torch.tensor([9., 8.99, 9.9, 1.8])
        self.tracker.flat_max_unstable_time[:] = torch.tensor([.5, .1, .52, .2])
        self.assertEqual(float(self.step()["flat/success_rate"]), .5)
        torch.testing.assert_close(self.course.success_streak[:, 2, 0], torch.tensor([1, 0, 0, 1]))

    def test_flat_quality_rejects_invalid_configuration(self):
        for params in ({"flat_min_quality_time": 0}, {"flat_min_stable_fraction": 0},
                       {"flat_min_stable_fraction": 1.1}, {"flat_max_unstable_time": -.1}):
            with self.subTest(params=params), self.assertRaisesRegex(ValueError, "平地质量"):
                self.course(self.env, [], **self.params, **params)
        self.landing_fixture()
        with self.assertRaisesRegex(ValueError, "flat_quality_warmup"):
            self.tracker(self.env, **self.landing_params, flat_quality_warmup=-.1)

    def test_flat_quality_subset_reset_preserves_other_episodes(self):
        for name in ("flat_quality_time", "flat_stable_time", "flat_unstable_time", "flat_max_unstable_time"):
            getattr(self.tracker, name)[:] = torch.tensor([1., 2., 3., 4.])
        self.tracker.reset([1, 3])
        for name in ("flat_quality_time", "flat_stable_time", "flat_unstable_time", "flat_max_unstable_time"):
            torch.testing.assert_close(getattr(self.tracker, name), torch.tensor([1., 0., 3., 0.]))

    def test_yaw_success_requires_actual_signed_rotation(self):
        for target, actual, expected in ((.4, 0., 0.), (.4, -.4, 0.),
                                         (.4, .1, 0.), (.4, .4, 1.),
                                         (-.4, -.4, 1.), (-.4, .4, 0.),
                                         (.4, 1.0, 0.)):
            with self.subTest(target=target, actual=actual):
                self.setUp()
                robot, _ = self.landing_fixture()
                self.command.episode_traversal[:] = 2
                self.command.episode_mode[:] = 3
                self.command.vel_command_b[:] = torch.tensor([0., 0., target])
                robot.root_lin_vel_b[:] = 0
                robot.root_ang_vel_b[:, 2] = actual
                self.env.episode_length_buf[:] = 2000
                for _ in range(200):
                    self.tracker(self.env, **self.landing_params)
                self.timeout[:] = True
                logs = self.step()
                self.assertEqual(float(logs["flat/turning/success_rate"]), expected)

    def test_standing_uses_tighter_zero_command_tolerances(self):
        for actual_vx, actual_yaw, expected in ((0., 0., 1.), (0., .3, 0.), (.11, 0., 0.)):
            self.setUp()
            robot, _ = self.landing_fixture()
            self.command.episode_traversal[:] = 2
            self.command.episode_standing[:] = True
            self.command.vel_command_b[:] = 0
            robot.root_lin_vel_b[:] = 0
            robot.root_lin_vel_b[:, 0] = actual_vx
            robot.root_ang_vel_b[:, 2] = actual_yaw
            self.env.episode_length_buf[:] = 2000
            for _ in range(200):
                self.tracker(self.env, **self.landing_params)
            logs = self.step()
            self.assertEqual(float(logs["flat/forward/success_rate"]), expected)

    def test_velocity_tracker_includes_terminal_sample_and_subset_reset(self):
        robot, _ = self.landing_fixture()
        self.command.vel_command_b[:] = torch.tensor([.4, 0., 0.])
        robot.root_lin_vel_b[:, 0] = .3
        for _ in range(20):
            done = self.tracker(self.env, **self.landing_params)
        self.assertTrue(done.all())
        torch.testing.assert_close(self.tracker.along_command_integral,
                                   torch.full((4,), .3 * .4))
        self.tracker.reset([1, 3])
        torch.testing.assert_close(self.tracker.velocity_integral[:, 0],
                                   torch.tensor([.12, 0., .12, 0.]))
        torch.testing.assert_close(self.tracker.command_speed_integral,
                                   torch.tensor([.16, 0., .16, 0.]))

    def test_command_subset_reset_preserves_other_episodes(self):
        ids = torch.arange(4)
        self.command.prepare_next_episode(ids)
        self.command.next_descent_speed_range[:] = torch.tensor([.15, .3])
        self.command._resample_command(ids)
        previous = self.command.vel_command_b.clone()
        self.command.stage = 0
        self.command.prepare_next_episode([1, 3])
        self.command.next_traversal[[1, 3]] = 0
        self.command._resample_command([1, 3])
        torch.testing.assert_close(self.command.vel_command_b[[0, 2]], previous[[0, 2]])
        self.assertTrue(torch.all(self.command.episode_mode[[1, 3]] == 0))
        current = self.command.vel_command_b.clone()
        self.command._resample_command(ids)
        torch.testing.assert_close(self.command.vel_command_b, current)

    def test_yaw_accumulators_reset_only_selected_environments(self):
        self.tracker.along_yaw_integral[:] = 2
        self.tracker.command_yaw_integral[:] = 3
        self.tracker.reset([1, 3])
        torch.testing.assert_close(self.tracker.along_yaw_integral, torch.tensor([2., 0., 2., 0.]))
        torch.testing.assert_close(self.tracker.command_yaw_integral, torch.tensor([3., 0., 3., 0.]))


class StairRewardsTests(unittest.TestCase):
    def setUp(self):
        managers = ModuleType("isaaclab.managers")
        managers.SceneEntityCfg = type("Cfg", (), {})
        spec = importlib.util.spec_from_file_location("stairs_rewards", TASK_DIR / "mdp/stairs_rewards.py")
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"isaaclab.managers": managers}):
            spec.loader.exec_module(self.module)

    def timer_env(self, air, contact=None, commands=None):
        air = torch.tensor(air, dtype=torch.float)
        contact = torch.zeros_like(air) if contact is None else torch.tensor(contact, dtype=torch.float)
        commands = (torch.tensor([[.4, 0., 0.]]).repeat(air.shape[0], 1) if commands is None
                    else torch.tensor(commands, dtype=torch.float))
        data = SimpleNamespace(current_air_time=air, current_contact_time=contact)
        env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": SimpleNamespace(data=data)}),
                              command_manager=SimpleNamespace(get_command=lambda name: commands))
        cfg = SimpleNamespace(name="contact_forces", body_ids=slice(None))
        return env, cfg, data

    def test_air_cost_has_grace_ramp_per_foot_and_cap(self):
        env, cfg, _ = self.timer_env([[.6, 0, 0, 0], [.8, 0, 0, 0],
                                     [1., 20., .1, .3], [5., 5., 5., 5.]])
        value = self.module.feet_air_time_excess(env, cfg, "base_velocity")
        torch.testing.assert_close(value, torch.tensor([0., .5, 2., 4.]))

    def test_stance_cost_allows_short_stair_support(self):
        contact = [[1., 0, 0, 0], [1.25, 0, 0, 0], [1.5, 20., .1, .3], [5., 5., 5., 5.]]
        env, cfg, _ = self.timer_env(torch.zeros(4, 4).tolist(), contact)
        value = self.module.feet_contact_time_excess(env, cfg, "base_velocity")
        torch.testing.assert_close(value, torch.tensor([0., .5, 2., 4.]))

    def test_time_cost_gates_all_translation_directions_and_stationary_commands(self):
        commands = [[.4, 0, 0], [-.4, 0, 0], [0, .4, 0], [0, -.4, 0],
                    [.4, 0, .1], [0, 0, 0], [0, 0, .5], [.1, 0, 0], [.08, .08, 0]]
        times = [[5.] * 4] * len(commands)
        env, cfg, _ = self.timer_env(times, times, commands)
        # No actual-speed gate: a stalled policy with a movement command still pays.
        for function in (self.module.feet_air_time_excess, self.module.feet_contact_time_excess):
            torch.testing.assert_close(function(env, cfg, "base_velocity"),
                                       torch.tensor([4., 4., 4., 4., 4., 0., 0., 0., 4.]))

    def test_time_cost_selects_feet_and_tracks_each_transition_independently(self):
        env, cfg, data = self.timer_env([[20., .8, 20., 0.]], [[20., 0., 20., 1.25]])
        cfg.body_ids = [1, 3]
        torch.testing.assert_close(self.module.feet_air_time_excess(env, cfg, "base_velocity"),
                                   torch.tensor([.5]))
        torch.testing.assert_close(self.module.feet_contact_time_excess(env, cfg, "base_velocity"),
                                   torch.tensor([.5]))
        data.current_air_time[:, 1] = 0  # This foot lands; the other foot's stance cost persists.
        self.assertEqual(self.module.feet_air_time_excess(env, cfg, "base_velocity").item(), 0)
        self.assertEqual(self.module.feet_contact_time_excess(env, cfg, "base_velocity").item(), .5)

    def test_time_cost_rejects_invalid_thresholds_and_disabled_sensor_timers(self):
        env, cfg, data = self.timer_env([[0., 0., 0., 0.]])
        for kwargs in ({"max_air_time": 0.}, {"ramp_time": 0.}, {"command_threshold": -.1}):
            with self.assertRaises(ValueError):
                self.module.feet_air_time_excess(env, cfg, "base_velocity", **kwargs)
        data.current_air_time = None
        with self.assertRaisesRegex(ValueError, "track_air_time"):
            self.module.feet_air_time_excess(env, cfg, "base_velocity")

    def test_stumble_ignores_normal_support_and_force_noise(self):
        forces = torch.zeros(4, 4, 3)
        forces[0, 0] = torch.tensor([3., 0., 50.])  # Normal support.
        forces[1, 2] = torch.tensor([10., 0., 1.])  # A vertical face hit.
        forces[2, 0] = torch.tensor([0.01, 0., 0.])  # Tiny force noise.
        forces[3, 0] = torch.tensor([10., 0., -4.])  # Use absolute vertical force.
        sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))
        env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor}))
        result = self.module.feet_stumble(env, SimpleNamespace(name="contact_forces", body_ids=slice(None)))
        torch.testing.assert_close(result, torch.tensor([0., 1., 0., 0.]))


class CopiedContractTests(unittest.TestCase):
    def test_observation_definitions_are_exact_copies(self):
        original = ast.parse((TASK_DIR / "bpx_rough_history_env_cfg.py").read_text())
        stairs = ast.parse((TASK_DIR / "bpx_stairs_history_env_cfg.py").read_text())
        # Rewards are intentionally independent and may be tuned for stairs.
        for suffix in ("ObservationsCfg",):
            source = next(n for n in original.body if isinstance(n, ast.ClassDef)
                          and n.name == "BpxRoughHistory" + suffix)
            copied = next(n for n in stairs.body if isinstance(n, ast.ClassDef)
                          and n.name == "BpxStairsHistory" + suffix)
            source.name = copied.name
            self.assertEqual(ast.dump(source), ast.dump(copied))



if __name__ == "__main__":
    unittest.main()
