"""Native solid stairs with a 3–5 cm first descent row and a wide landing."""

import numpy as np

from isaaclab.terrains import MeshPyramidStairsTerrainCfg
from isaaclab.terrains.trimesh.mesh_terrains import pyramid_stairs_terrain
from isaaclab.utils import configclass


def descent_stairs_terrain(difficulty, cfg):
    # TerrainGenerator samples difficulty uniformly *within* each row. A band
    # per row keeps all of row zero in [3, 5] cm, rather than [3, 4.5] cm.
    scaled = np.clip(difficulty, 0.0, np.nextafter(1.0, 0.0)) * cfg.levels
    row = min(int(scaled), cfg.levels - 1)
    lower = cfg.initial_height_range[0] + row / (cfg.levels - 1) * (
        cfg.final_height_range[0] - cfg.initial_height_range[0])
    upper = cfg.initial_height_range[1] + row / (cfg.levels - 1) * (
        cfg.final_height_range[1] - cfg.initial_height_range[1])
    native_cfg = cfg.copy()
    height = lower + (scaled - row) * (upper - lower)
    native_cfg.step_height_range = (height, height)
    return pyramid_stairs_terrain(0.0, native_cfg)


@configclass
class BpxDescentTerrainCfg(MeshPyramidStairsTerrainCfg):
    function = descent_stairs_terrain
    levels: int = 8
    initial_height_range: tuple[float, float] = (0.03, 0.05)
    final_height_range: tuple[float, float] = (0.13, 0.15)
