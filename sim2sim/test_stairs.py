"""Solid stair geometry, contact queries, and PD-init handoff integration checks."""
from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
import csv
import io
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from bpx_sim2sim.config import load_config
from bpx_sim2sim.policy import ZeroPolicy
from bpx_sim2sim.robot import BpxMujocoRobot
from bpx_sim2sim.runner import Sim2SimRunner
from bpx_sim2sim.supervisor import BehaviorSupervisor
from bpx_sim2sim.terrain import add_course_labels, load_model


CONFIG = Path(__file__).parent / "config/bpx_stairs_history.toml"


class StairCourseTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIG)
        self.model = load_model(self.cfg)
        self.data = mujoco.MjData(self.model)
        self.robot = BpxMujocoRobot(self.model, self.data, self.cfg)
        self.robot.reset()

    def move(self, x, y, z):
        self.data.qpos[self.robot.base_qpos_address:self.robot.base_qpos_address + 3] = (x, y, z)
        mujoco.mj_forward(self.model, self.data)

    def test_tilt_opt_out_applies_to_optional_fall_check(self):
        self.assertFalse(self.cfg.supervisor.terminate_on_tilt)
        runner = Sim2SimRunner(self.cfg, ZeroPolicy(12))
        start = runner.robot.base_qpos_address
        runner.data.qpos[start:start + 3] = (0., 0., 2.)
        runner.data.qpos[start + 3:start + 7] = (np.cos(.6), np.sin(.6), 0., 0.)
        mujoco.mj_forward(runner.model, runner.data)
        self.assertAlmostEqual(runner.robot.tilt_angle(), 1.2)
        self.assertIsNone(runner._fall_reason())
        runner.config = replace(self.cfg, supervisor=replace(self.cfg.supervisor, terminate_on_tilt=True))
        self.assertEqual(runner._fall_reason(), "bad_orientation")
        runner.config = self.cfg
        runner.data.qpos[start:start + 3] = (0., 0., .1)
        mujoco.mj_forward(runner.model, runner.data)
        self.assertEqual(runner._fall_reason(), "base_height")

    def test_six_fifteen_cm_risers_and_twenty_cm_treads_in_both_directions(self):
        for x, height in ((0, 0), (1.99, 0), (2.1, .15), (2.3, .3), (2.5, .45),
                          (2.7, .6), (2.9, .75), (3.1, .9), (4.99, .9), (5.1, .75),
                          (5.3, .6), (5.5, .45), (5.7, .3), (5.9, .15), (6.01, 0)):
            self.move(x, 0, height + .4)
            self.assertAlmostEqual(self.robot.ground_height(), height, places=7, msg=str(x))
            self.assertAlmostEqual(self.robot.base_height(), .4, places=7)
        self.move(4, 1.01, .4)
        self.assertAlmostEqual(self.robot.ground_height(), 0, places=7)
        self.assertEqual(len(self.robot.ground_geom_ids), 7)
        self.assertEqual(self.model.nhfield, 0)

    def test_risers_are_vertical_solid_colliders(self):
        group = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
        for z, expected in ((.05, .2), (.15001, .4)):
            distance = mujoco.mj_ray(self.model, self.data, np.array([1.8, 0, z]),
                                     np.array([1., 0, 0]), group, 1, -1, None)
            self.assertAlmostEqual(distance, expected, places=6)

    def test_stair_tops_count_as_ground_for_feet_and_torso(self):
        for x, height in ((0, 0), (4, .9), (7, 0)):
            self.robot.reset()
            self.data.qpos[self.robot.joint_qpos_addresses] = self.robot.default_joint_position
            self.move(x, 0, height + .32)
            self.assertGreater(self.robot.feet_in_contact(), 0)
            self.move(x, 0, height + .10)
            self.assertTrue(self.robot.torso_touches_floor())

    def test_robot_dynamics_and_observations_match_flat_start(self):
        cfg = replace(self.cfg, terrain=replace(self.cfg.terrain, kind="flat"))
        flat = load_model(cfg)
        robot = BpxMujocoRobot(flat, mujoco.MjData(flat), cfg)
        robot.reset()
        for field in ("body_mass", "body_inertia", "jnt_range", "actuator_ctrlrange"):
            np.testing.assert_array_equal(getattr(self.model, field), getattr(flat, field))
        np.testing.assert_array_equal(self.robot.observation(np.zeros(3), np.zeros(12)),
                                      robot.observation(np.zeros(3), np.zeros(12)))

    def test_viewer_labels_and_logged_regions(self):
        class Viewer:
            user_scn = mujoco.MjvScene(self.model, maxgeom=10)
            opt = mujoco.MjvOption()
            cam = mujoco.MjvCamera()

            def lock(self):
                return nullcontext()

        viewer = Viewer()
        add_course_labels(viewer, self.cfg)
        self.assertEqual(viewer.user_scn.ngeom, 5)
        self.assertEqual(viewer.user_scn.geoms[1].label, "UP 6 x 15 cm")
        self.assertEqual(viewer.user_scn.geoms[2].label, "TOP 90 cm")
        for x, label in ((0, "APPROACH"), (2.2, "ASCENT"), (4, "TOP LANDING"),
                         (5.4, "DESCENT"), (7, "EXIT")):
            self.move(x, 0, .8)
            self.assertIn(label, self.robot.terrain_region())

    def test_pd_init_hands_off_before_accepting_motion(self):
        # Exercise the real state machine/controller without depending on model artifacts.
        runner = Sim2SimRunner(self.cfg, ZeroPolicy(12), BehaviorSupervisor(self.cfg.supervisor))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / "init.csv"
            result = runner.run((.25, 0, 0), init_request_source=lambda: True,
                                duration=8, realtime=False, log_path=path)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(result.termination_reason, "duration")
        self.assertIn("init_pd", {row["behavior_policy"] for row in rows})
        self.assertIn("walk", {row["behavior_mode"] for row in rows})
        init_rows = [row for row in rows if row["behavior_policy"] == "init_pd"]
        self.assertTrue(init_rows)
        for row in init_rows:
            self.assertEqual(float(row["command_vx"]), 0)
        self.assertTrue(np.isfinite(runner.data.qpos).all())


if __name__ == "__main__":
    unittest.main()
