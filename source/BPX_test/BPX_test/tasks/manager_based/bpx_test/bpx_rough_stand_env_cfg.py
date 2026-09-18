"""Blind rough standing with the unchanged Stand policy interface."""
from isaaclab.managers import CurriculumTermCfg as CurrTerm, TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from .bpx_stand_env_cfg import BpxStandEnvCfg, BpxStandTerminationsCfg
from .bpx_rough_env_cfg import BpxWaveTerrainCfg
from .mdp.rough_stand import reset_rough_stand, relative_stand_height_l2, StandStability, StandingCurriculum


@configclass
class RoughStandTerminationsCfg(BpxStandTerminationsCfg):
    stand_drift = DoneTerm(func=StandStability, params={
        'settling_time': .5, 'max_speed': .15, 'max_yaw_rate': .25,
        'max_tilt': .25, 'max_drift': .20, 'failure_drift': .65,
    })


@configclass
class RoughStandCurriculumCfg:
    terrain_levels = CurrTerm(func=StandingCurriculum, params={
        'min_duration': 14.4, 'stable_fraction': .90, 'successes_to_promote': 2,
    })


@configclass
class BpxRoughStandEnvCfg(BpxStandEnvCfg):
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
        self.events.reset_base.params = {'flat_probability': .1}
        self.events.reset_robot_joints.params = {'position_range': (-.03, .03), 'velocity_range': (0., 0.)}
        self.events.push_robot.params['velocity_range'] = {
            'x': (-.1, .1), 'y': (-.1, .1), 'yaw': (-.1, .1),
        }
        self.rewards.base_height_l2.func = relative_stand_height_l2
        self.rewards.base_height_l2.params = {'target_height': .40}
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
