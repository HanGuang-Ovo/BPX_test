"""Blind rough locomotion: unchanged flat rewards, 48-D single-frame observations."""
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from . import mdp
from .bpx_locomotion_env_cfg import BpxLocomotionEnvCfg, BpxLocomotionTerminationsCfg
from .mdp.rough_curriculum import RoughTraversalBoundary, TraversabilityCurriculum
from .rough_terrain_geometry import wave_terrain


@configclass
class BpxWaveTerrainCfg(SubTerrainBaseCfg):
    function = wave_terrain
    horizontal_scale: float = 0.025
    amplitude_range: tuple[float, float] = (0.005, 0.04)
    wavelength_range: tuple[float, float] = (0.15, 0.30)
    levels: int = 8
    platform_width: float = 2.0
    transition_width: float = 0.5


@configclass
class BpxRoughCommandsCfg:
    # Each component independently uniform. No standing or pure-rotation mixture.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0, heading_command=False, debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class BpxRoughTerminationsCfg(BpxLocomotionTerminationsCfg):
    # Time-limit truncation (value bootstrap), not an extra failure penalty.
    terrain_boundary = DoneTerm(
        func=RoughTraversalBoundary, time_out=True,
        params={"rough_start": 1.95, "boundary": 3.4},
    )


@configclass
class BpxRoughCurriculumCfg:
    terrain_levels = CurrTerm(
        func=TraversabilityCurriculum,
        params={"min_rough_distance": 1.0, "min_rough_time": 1.0,
                "max_linear_error": 0.35, "max_angular_error": 0.5,
                "successes_to_promote": 2},
    )


@configclass
class BpxRoughEnvCfg(BpxLocomotionEnvCfg):
    commands: BpxRoughCommandsCfg = BpxRoughCommandsCfg()
    terminations: BpxRoughTerminationsCfg = BpxRoughTerminationsCfg()
    curriculum: BpxRoughCurriculumCfg = BpxRoughCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=20260918, curriculum=True, size=(8.0, 8.0),
            num_rows=8, num_cols=4, border_width=5.0,
            color_scheme="height", use_cache=False,
            sub_terrains={"bpx_waves": BpxWaveTerrainCfg()},
        )
        # Flat 2x2 m platform; ±0.1 m center jitter leaves room for feet at any yaw.
        # Zero initial velocity prevents resets from immediately sliding off it.
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
