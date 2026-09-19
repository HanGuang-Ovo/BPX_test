"""Blind rough standing with the unchanged Stand policy interface."""
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from . import mdp
from .bpx_base_env_cfg import BPX_POLICY_JOINT_NAMES
from .bpx_stand_env_cfg import BpxStandEnvCfg, BpxStandRewardsCfg, BpxStandTerminationsCfg
from .bpx_rough_env_cfg import BpxWaveTerrainCfg
from .mdp.rough_stand import (
    reset_rough_stand, reset_rough_stand_joints, rough_stand_last_action,
    relative_stand_height_l2, StandStability, StandingCurriculum,
)


@configclass
class RoughStandTerminationsCfg(BpxStandTerminationsCfg):
    stand_drift = DoneTerm(func=StandStability, params={
        'settling_time': .5, 'max_speed': .15, 'max_yaw_rate': .25,
        'max_tilt': .1, 'max_drift': .20, 'failure_drift': .65,
        'contact_force_threshold': 5.,
        'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_toe_link'),
        'symmetry_tolerance': .20,
        'symmetry_joint_cfg': SceneEntityCfg(
            'robot', joint_names=BPX_POLICY_JOINT_NAMES, preserve_order=True,
        ),
    })


@configclass
class RoughStandRewardsCfg(BpxStandRewardsCfg):
    left_right_joint_symmetry = RewTerm(
        func=mdp.left_right_joint_symmetry_l2,
        weight=-1.0,
        params={
            'tolerance': .10,
            'asset_cfg': SceneEntityCfg(
                'robot', joint_names=BPX_POLICY_JOINT_NAMES, preserve_order=True,
            ),
        },
    )


@configclass
class RoughStandCurriculumCfg:
    terrain_levels = CurrTerm(func=StandingCurriculum, params={
        'min_duration': 14.4, 'stable_fraction': .90,
        'minimum_four_feet_fraction': .90, 'minimum_symmetry_fraction': .80,
        'successes_to_promote': 2,
    })


@configclass
class BpxRoughStandEnvCfg(BpxStandEnvCfg):
    rewards: RoughStandRewardsCfg = RoughStandRewardsCfg()
    terminations: RoughStandTerminationsCfg = RoughStandTerminationsCfg()
    curriculum: RoughStandCurriculumCfg = RoughStandCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = 'generator'
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=20260918, curriculum=True, size=(8., 8.), num_rows=8, num_cols=4,
            border_width=5., color_scheme='height', use_cache=False,
            sub_terrains={'bpx_waves': BpxWaveTerrainCfg()},
        )
        self.events.reset_base.func = reset_rough_stand
        self.events.reset_base.params = {
            'flat_probability': .25,
            'replay_probability': .25,
            'handoff_probability': .25,
            'pose_range': {
                'roll': (-.05, .05), 'pitch': (-.05, .05), 'yaw': (-3.14, 3.14),
            },
            'velocity_range': {
                'x': (-.05, .05), 'y': (-.05, .05), 'z': (-.03, .03),
                'roll': (-.08, .08), 'pitch': (-.08, .08), 'yaw': (-.08, .08),
            },
            'handoff_pose_range': {
                'roll': (-.10, .10), 'pitch': (-.10, .10), 'yaw': (-3.14, 3.14),
            },
            'handoff_velocity_range': {
                'x': (-.08, .08), 'y': (-.08, .08), 'z': (-.05, .05),
                'roll': (-.15, .15), 'pitch': (-.15, .15), 'yaw': (-.15, .15),
            },
        }
        self.events.reset_robot_joints.func = reset_rough_stand_joints
        self.events.reset_robot_joints.params = {
            'position_range': (-.05, .05), 'velocity_range': (-.05, .05),
            'handoff_position_range': (-.15, .15), 'handoff_velocity_range': (-.30, .30),
            'action_scale': .5,
        }
        self.observations.policy.actions.func = rough_stand_last_action
        self.events.push_robot.params['velocity_range'] = {
            'x': (-.1, .1), 'y': (-.1, .1), 'yaw': (-.1, .1),
        }
        self.rewards.base_height_l2.func = relative_stand_height_l2
        self.rewards.base_height_l2.params = {'target_height': .40}
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
