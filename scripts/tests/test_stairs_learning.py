"""CPU tensor regressions for stair learning; Isaac Sim is not required.

Execute the production definitions with only Isaac's manager/action shells
replaced. All geometry queries, command transitions, rewards and curricula run
the real source and real torch operations. The simulator-level companion is
scripts/validate_stairs_skills.py.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
STAIRS = ROOT / "source/BPX_test/BPX_test/tasks/manager_based/bpx_test/mdp/stairs.py"


class ManagerShell:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.device, self.num_envs = env.device, env.num_envs

    def reset(self, ids):
        info = {key: value[ids].float().mean().item() for key, value in self.metrics.items()}
        for value in self.metrics.values():
            value[ids] = 0
        self._resample_command(ids)
        return info


def load_production():
    namespace = dict(torch=torch, np=np, CommandTerm=ManagerShell, ManagerTermBase=ManagerShell,
                     CommandTermCfg=object, JointPositionAction=object, JointPositionActionCfg=object,
                     configclass=lambda cls: cls,
                     quat_from_euler_xyz=lambda roll, pitch, yaw: torch.stack(
                         (torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll),
                          torch.zeros_like(roll)), dim=-1))
    tree = ast.parse(STAIRS.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(STAIRS), "exec"), namespace)
    return SimpleNamespace(**namespace)


stairs = load_production()


class Scene(dict):
    pass


def make_env(n=4):
    origins = torch.zeros(n, 3)
    root = origins.clone()
    root[:, 2] = .4
    state = torch.zeros(n, 13)
    state[:, :3], state[:, 3] = root, 1.
    feet = root[:, None, :].repeat(1, 4, 1)
    feet[:, :, 0] += torch.tensor([.3, .3, -.3, -.3])
    feet[:, :, 1] += torch.tensor([.2, -.2, .2, -.2])
    feet[:, :, 2] = .04
    data = SimpleNamespace(root_pos_w=root, heading_w=torch.zeros(n), default_root_state=state,
                           root_lin_vel_w=torch.zeros(n, 3), root_lin_vel_b=torch.zeros(n, 3),
                           root_ang_vel_b=torch.zeros(n, 3), body_pos_w=feet,
                           body_lin_vel_w=torch.zeros(n, 4, 3), joint_vel=torch.zeros(n, 12),
                           projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1))

    def write_root(state, env_ids):
        data.root_pos_w[env_ids] = state[:, :3]

    scene = Scene(robot=SimpleNamespace(data=data, write_root_state_to_sim=write_root))
    scene.env_origins = origins
    forces = torch.zeros(n, 4, 3)
    forces[:, :, 2] = 25.
    scene.sensors = {"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))}
    terrain_cfg = SimpleNamespace(num_rows=8, num_cols=2, size=(20., 3.))
    scene.terrain = SimpleNamespace(cfg=SimpleNamespace(terrain_generator=terrain_cfg),
                                   terrain_levels=torch.zeros(n, dtype=torch.long),
                                   terrain_types=torch.ones(n, dtype=torch.long), env_origins=origins,
                                   terrain_origins=torch.zeros(8, 2, 3))
    edges = torch.full((8, 2, 4), float("inf"))
    edges[:, 1] = torch.tensor([-1., -.5, .5, 1.])
    deltas = torch.zeros_like(edges)
    deltas[:, 1] = torch.tensor([.04, .04, -.04, -.04])
    counts = torch.zeros(8, 2, dtype=torch.long)
    counts[:, 1] = 2
    table = edges, deltas, torch.full((8, 2), 3.), counts, torch.tensor([0]), torch.tensor([1])
    env = SimpleNamespace(num_envs=n, device="cpu", step_dt=.02, scene=scene, _bpx_stair_table=table,
                          episode_length_buf=torch.full((n,), 50),
                          termination_manager=SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool)))
    curriculum_cfg = SimpleNamespace(params={"initial_level": 0, "initial_stop_level": 0})
    stairs.StairMasteryCurriculum(curriculum_cfg, env)
    c = stairs.StairVelocityCommand(stairs.StairVelocityCommandCfg(), env)
    c.route[:] = 1
    c.stop_difficulty[:] = 2
    c.stop_required[:] = True
    c.next_stop_difficulty[:] = 2
    c.next_stop_required[:] = True
    c.hold_duration[:] = 1.5
    c.goal_x[:] = 2.
    c.stop_x[:] = 0.
    env.command_manager = SimpleNamespace(get_command=lambda name: c.command)
    return env


ENTITY = SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2, 3])
CURRICULUM = dict(successes_to_promote=3, failures_to_demote=2,
                  max_linear_error=.2, max_angular_error=.3)


def test_privileged_observations_are_finite_local_and_include_current_hold_score():
    env = make_env()
    heights = stairs.critic_terrain_heights(env)
    assert heights.shape == (4, 12)
    # Positions ahead of the first stair see higher ground; translated tile
    # origins must give identical local observations, regardless of origin.z.
    env.scene.env_origins[1] = torch.tensor([20., 3., .9])
    env.scene["robot"].data.root_pos_w[1, :2] += torch.tensor([20., 3.])
    torch.testing.assert_close(stairs.critic_terrain_heights(env)[1], heights[0])
    support = stairs.critic_foot_support(env, ENTITY)
    assert support.shape == (4, 8)
    torch.testing.assert_close(support[0], torch.tensor([1., 1., 1., 1., .25, .25, .25, .25]))
    c = env._bpx_stairs_command
    c.hold_observed[:] = 1.
    before = stairs.critic_stop_state(env)
    assert before.shape == (4, 28) and torch.isfinite(before).all()
    command = c.command.clone()
    c.hold_good[0] = .8
    after = stairs.critic_stop_state(env)
    assert not torch.equal(before[0], after[0])
    torch.testing.assert_close(before[1:], after[1:])
    torch.testing.assert_close(command, c.command)


def test_terrain_scan_and_drift_rotate_with_heading():
    env = make_env(2)
    env.scene["robot"].data.heading_w[1] = torch.pi / 2
    heights = stairs.critic_terrain_heights(env)
    assert not torch.equal(heights[0], heights[1])
    c = env._bpx_stairs_command
    c.anchor[:, 0] = -.05
    # Drift occupies features 21:23 in the documented 28-D snapshot.
    drift = stairs.critic_stop_state(env)[:, 21:23]
    torch.testing.assert_close(drift[0], torch.tensor([.5, 0.]))
    torch.testing.assert_close(drift[1], torch.tensor([0., -.5]), atol=1.e-6, rtol=0.)


def test_failed_holds_do_not_block_traversal_progress():
    env = make_env()
    c, curriculum = env._bpx_stairs_command, env._bpx_stair_curriculum
    curriculum.adaptive[:] = curriculum.stop_adaptive[:] = True
    c.completed[:] = c.stop_done[:] = True
    c.duration[:] = 10.
    c.hold_observed[:] = 1.
    for _ in range(3):
        info = curriculum(env, torch.arange(4), **CURRICULUM)
    assert torch.all(curriculum.mastery == 1) and torch.all(curriculum.stop_mastery == 0)
    assert info["traversal_success"] == 1 and info["success_rate"] == info["stop_success"] == 0
    # Completing a good hold remains a stop skill even if traversal later fails.
    c.completed[:] = False
    c.hold_good[:] = 1.
    env.termination_manager.terminated[:] = True
    for _ in range(3):
        curriculum(env, torch.arange(4), **CURRICULUM)
    assert torch.all(curriculum.mastery == 0) and torch.all(curriculum.stop_mastery == 1)


def test_exempt_replay_flat_and_unreached_holds_do_not_promote_stop_skill():
    env = make_env()
    c, curriculum = env._bpx_stairs_command, env._bpx_stair_curriculum
    curriculum.mastery[:] = curriculum.stop_mastery[:] = 1
    curriculum.adaptive[:] = curriculum.stop_adaptive[:] = True
    c.route[0] = 0                      # flat
    curriculum.adaptive[1] = curriculum.stop_adaptive[1] = False  # replay
    c.stop_required[2] = False         # exempt
    c.phase[:] = 0                    # last env never reached its hold
    c.completed[:] = True
    c.duration[:] = 10.
    for _ in range(3):
        curriculum(env, torch.arange(4), **CURRICULUM)
    torch.testing.assert_close(curriculum.stop_mastery, torch.ones(4, dtype=torch.long))
    assert curriculum.mastery[0] == curriculum.mastery[1] == 1
    assert curriculum.mastery[2] == curriculum.mastery[3] == 2


def test_disabled_curriculum_does_not_change_levels_or_streaks():
    env = make_env()
    curriculum = env._bpx_stair_curriculum
    curriculum.adaptive[:] = curriculum.stop_adaptive[:] = True
    for _ in range(4):
        curriculum(env, torch.arange(4), **CURRICULUM, enabled=False)
    for field in ("mastery", "stop_mastery", "successes", "failures", "stop_successes", "stop_failures"):
        assert not getattr(curriculum, field).any()


def test_stop_curriculum_controls_terrain_and_position_without_changing_schedule():
    torch.manual_seed(4)
    env = make_env()
    c, curriculum = env._bpx_stairs_command, env._bpx_stair_curriculum
    curriculum.mastery[:] = 3
    curriculum.stop_mastery[:] = torch.tensor([0, 1, 2, 5])
    c.cfg.stop_probability = 1.
    ids = torch.arange(4)
    stairs.reset_stair_root(env, ids, 0., 0., (.6, 1.5), .4)
    torch.testing.assert_close(env.scene.terrain.terrain_levels, torch.tensor([0, 0, 0, 3]))
    torch.testing.assert_close(curriculum.adaptive, torch.tensor([False, False, False, True]))
    assert curriculum.stop_adaptive.all() and c.stop_required.all()
    route = c.route.clone()
    c._resample_command(ids)
    assert c.stop_required.all() and torch.equal(route, c.route)
    first = torch.where(c.route == 2, .5, -1.)
    assert abs(float(c.stop_x[0] - first[0]) + .5) <= .031
    selected_index = torch.where(c.stop_stage == 0, 0, 1) + torch.where(c.route == 2, 2, 0)
    selected_edge = env._bpx_stair_table[0][env.scene.terrain.terrain_levels, 1, selected_index]
    assert 0 < c.stop_x[1] - selected_edge[1] <= .2 + 1.e-6
    assert torch.all((c.stop_x[2:] - selected_edge[2:]).abs() <= .08)


def test_new_reset_plan_does_not_overwrite_terminal_hold_diagnostics():
    env = make_env()
    c = env._bpx_stairs_command
    c.stop_done[:] = True
    c.hold_observed[:] = c.hold_good[:] = 1.
    c._update_metrics()
    c.cfg.stop_probability = 0.
    stairs.reset_stair_root(env, torch.arange(4), 0., 0., (.6, 1.5), .4)
    assert c.stop_required.all() and not c.next_stop_required.any()
    info = c.reset(slice(None))
    assert info["hold_sample_count"] == 4 and info["hold_success_count"] == 4
    assert info["stop_success"] == 1 and info["stop_difficulty"] == 2
    assert not c.stop_required.any() and not c.stop_difficulty.any()


@pytest.mark.parametrize("dt", [.02, .01])
def test_three_event_rewards_are_once_only_and_independent_of_step_size(dt):
    env = make_env(6)
    env.step_dt = dt
    c = env._bpx_stairs_command
    c.completed[:] = True
    c.stop_done[:] = True
    c.hold_observed[:] = 1.
    c.hold_good[:] = 1.
    c.hold_good[1] = .79             # failed quality, still physically traversed
    c.stop_difficulty[2] = 0         # easy hold cannot pass final edge task
    c.stop_required[3] = False      # exempt traversal
    c.route[4] = 0                  # flat gets no stair events
    env.termination_manager.terminated[5] = True
    torch.testing.assert_close(stairs.stair_traversal_reward(env) * dt * 2.,
                               torch.tensor([2., 2., 2., 2., 0., 0.]))
    torch.testing.assert_close(stairs.stop_completion_reward(env) * dt,
                               torch.tensor([1., 0., 1., 0., 0., 0.]))
    torch.testing.assert_close(stairs.stair_completion_reward(env) * dt * 2.,
                               torch.tensor([2., 0., 0., 2., 0., 0.]))
    for reward in (stairs.stair_traversal_reward, stairs.stop_completion_reward, stairs.stair_completion_reward):
        assert not reward(env).any()
    # Only reset envs can receive their event again.
    c.reset(torch.tensor([0, 2]))
    assert not c.traversal_rewarded[[0, 2]].any() and c.traversal_rewarded[1]
    assert not c.stop_rewarded[[0, 2]].any()
    assert not c.completion_rewarded[0] and c.completion_rewarded[3]


def test_stop_event_is_available_before_exit_but_requires_finished_strict_hold():
    env = make_env()
    c = env._bpx_stairs_command
    c.hold_observed[:] = torch.tensor([1., 1., .4, 1.])
    c.hold_good[:] = torch.tensor([.8, 1., .4, .79])
    c.stop_done[:] = torch.tensor([True, False, True, True])
    assert not c.completed.any()
    torch.testing.assert_close(stairs.stop_completion_reward(env) * env.step_dt,
                               torch.tensor([1., 0., 0., 0.]))


def test_real_command_monitor_and_reward_sequence_can_finish_a_successful_stop():
    env = make_env()
    c = env._bpx_stairs_command
    c.start_wait[:] = -1.
    c.speed_timer[:] = 1000.
    c.speed[:] = c.vel_command_b[:, 0] = .4
    total = torch.zeros(4)
    for _ in range(180):
        stairs.StairEpisodeMonitor.__call__(None, env, ENTITY, ENTITY, .2, 20.)
        total += stairs.stop_completion_reward(env) * env.step_dt
        c._update_command()
    torch.testing.assert_close(total, torch.ones(4))
    assert c.stop_done.all() and stairs.stop_success(c).all()
    assert not c.completed.any() and (c.command[:, 0] > 0).all()
    torch.testing.assert_close(c.hold_good, c.hold_observed)


def test_quality_is_joint_bounded_and_uses_raw_support():
    env = make_env(8)
    c, data = env._bpx_stairs_command, env.scene["robot"].data
    c.phase[:] = 2
    c.hold_elapsed[:] = .5
    data.root_lin_vel_w[1, 0] = .08
    data.root_ang_vel_b[2, 0] = .20
    data.root_pos_w[3, 0] = .10
    forces = env.scene.sensors[ENTITY.name].data.net_forces_w
    forces[4, 2:, 2] = 0.           # two feet only, even if cached contacts say four
    c.foot_contacts = stairs.StairFootContacts(torch.ones(8, 4, dtype=torch.bool))
    c.foot_contacts.filtered[:] = True
    data.projected_gravity_b[5, 2] = -.8
    c.phase[6] = 0
    c.vel_command_b[7, 0] = .1
    quality = stairs.stop_quality_reward(env, ENTITY)
    torch.testing.assert_close(quality[:4], torch.tensor([1., np.exp(-1), np.exp(-1), np.exp(-1)]).float())
    assert not quality[4:].any() and torch.all((quality >= 0) & (quality <= 1))
    c.hold_elapsed[0] = .2
    assert stairs.stop_quality_reward(env, ENTITY)[0] == 0


def test_full_task_and_curriculum_do_not_accept_easy_stops_as_edge_stops():
    env = make_env()
    c, curriculum = env._bpx_stairs_command, env._bpx_stair_curriculum
    c.completed[:] = c.stop_done[:] = True
    c.hold_observed[:] = c.hold_good[:] = 1.
    c.duration[:] = 10.
    c.stop_difficulty[:] = torch.tensor([0, 1, 2, 2])
    env.termination_manager.terminated[3] = True
    expected = torch.tensor([False, False, True, False])
    stairs.stair_completion_reward(env)
    torch.testing.assert_close(c.task_success, expected)
    info = curriculum(env, torch.arange(4), **CURRICULUM)
    assert info["success_rate"] == .25 and info["traversal_success"] == .75


def test_policy_and_critic_history_and_noise_contract():
    # Execute the real observation config with lightweight term containers;
    # the real ObservationManager is checked by the integration script.
    class Term(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(history_length=kwargs.pop("history_length", 0),
                             noise=kwargs.pop("noise", None), **kwargs)

    class Core:
        def __getattr__(self, name):
            return lambda *args: None

    source = ROOT / "source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_stairs_common_cfg.py"
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == "BpxStairsObservationsCfg")
    ns = dict(configclass=lambda cls: cls, ObsGroup=object, ObsTerm=Term, core=Core(), stairs=stairs,
              Unoise=lambda **kwargs: kwargs, SceneEntityCfg=lambda *args, **kwargs: None,
              BPX_POLICY_JOINT_NAMES=[], feet_sensor=lambda: ENTITY)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    cfg = ns["BpxStairsObservationsCfg"]()
    cfg.policy.__post_init__()
    cfg.critic.__post_init__()
    assert cfg.policy.enable_corruption and cfg.policy.history_length == 10
    assert not cfg.critic.enable_corruption and cfg.critic.history_length is None
    terms = {key: value for key, value in vars(type(cfg.critic)).items() if isinstance(value, Term)}
    for name, term in terms.items():
        assert term.noise is None
        assert term.history_length == (0 if name.endswith("_context") else 10)
    assert list(key for key, value in vars(type(cfg.policy)).items() if isinstance(value, Term)) == [
        "base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"]


