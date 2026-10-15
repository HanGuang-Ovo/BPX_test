"""Geometry regressions, runnable without Isaac Sim or a GPU."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "source/BPX_test/BPX_test/tasks/manager_based/bpx_test/stair_geometry.py"
spec = importlib.util.spec_from_file_location("bpx_stair_geometry_test", path)
geometry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = geometry
spec.loader.exec_module(geometry)


class Terrain(SimpleNamespace):
    def copy(self):
        return copy.deepcopy(self)


def config(**kwargs):
    cfg = Terrain(flat=False, levels=8, seed=20261014, size=(20., 3.),
                  initial_height_range=(.03, .05), height_range=(.05, .15),
                  depth_range=(.25, .40), step_count_range=(4, 8),
                  width_range=(1.2, 2.), landing_length=3.2, approach_margin=1.55, proportion=.75)
    cfg.__dict__.update(kwargs)
    return cfg


@pytest.mark.parametrize("row", range(8))
def test_complete_routes_and_random_risers(row):
    cfg = config()
    low, high = np.asarray(cfg.initial_height_range) * (1 - row / 7) + np.asarray(cfg.height_range) * row / 7
    for jitter in np.linspace(.01, .99, 20):
        layout = geometry.stair_layout((row + jitter) / 8, cfg)
        count = layout.count
        delta = np.diff(np.r_[0., layout.heights])
        assert 4 <= count <= 8
        assert np.all((delta[:count] >= low) & (delta[:count] <= high))
        np.testing.assert_allclose(delta[count:], -delta[:count][::-1], atol=1.e-15)
        assert np.all(np.diff(layout.edges) > 0)
        assert layout.edges[0] > 1.55 and cfg.size[0] - layout.edges[-1] > 1.55
        assert np.all((layout.depths >= .25) & (layout.depths <= .40))
        x = np.r_[layout.edges[0] - .1, layout.edges + 1.e-6]
        heights = geometry.layout_height(layout, x, 1.5, cfg.size)
        np.testing.assert_allclose(heights, np.r_[0., layout.heights])
        # The last rise enters a level landing; the final drop returns to floor.
        assert geometry.layout_height(layout, layout.edges[count - 1] + .85, 1.5, cfg.size) == layout.heights[count - 1]
        assert geometry.layout_height(layout, layout.edges[-1] + .85, 1.5, cfg.size) == 0
        np.testing.assert_array_equal(geometry.layout_height(layout, x, 3., cfg.size), np.zeros_like(x))


def test_reproducibility_flat_seams_and_solid_vertical_mesh():
    cfg = config()
    a, b = geometry.stair_layout(.94, cfg), geometry.stair_layout(.94, cfg)
    np.testing.assert_array_equal(a.edges, b.edges)
    np.testing.assert_array_equal(a.heights, b.heights)
    meshes, origin = geometry.stair_terrain(.94, cfg)
    assert all(mesh.is_watertight for mesh in meshes)
    for mesh in meshes:
        # Only horizontal terraces / vertical walls, never discretization ramps.
        assert np.all(np.isclose(np.abs(mesh.face_normals[:, 2]), 0) | np.isclose(np.abs(mesh.face_normals[:, 2]), 1))
    assert origin[2] == geometry.layout_height(a, origin[0], origin[1], cfg.size)
    assert meshes[0].bounds[1, 2] == 0
    flat = geometry.stair_layout(.94, config(flat=True))
    assert flat.count == 0
    np.testing.assert_array_equal(geometry.layout_height(flat, [0., 10., 20.], 1.5, cfg.size), [0., 0., 0.])


def test_generator_table_matches_column_major_difficulty_draws():
    cfg = Terrain(curriculum=True, seed=20261014, size=(20., 3.), num_rows=8, num_cols=32,
                  difficulty_range=(0., 1.), sub_terrains={"flat": config(flat=True, proportion=.25), "stairs": config()})
    layouts, flat = geometry.generator_layouts(cfg)
    assert flat == list(range(8))
    rng = np.random.default_rng(cfg.seed)
    for col in range(32):
        for row in range(8):
            difficulty = (row + rng.uniform()) / 8
            expected = geometry.stair_layout(difficulty, config(flat=col < 8))
            np.testing.assert_array_equal(layouts[row][col].edges, expected.edges)
            np.testing.assert_array_equal(layouts[row][col].heights, expected.heights)


def test_rejects_short_landing_and_too_small_tile():
    with pytest.raises(ValueError, match="Landing"):
        geometry.stair_layout(.9, config(landing_length=1.))
    with pytest.raises(ValueError, match="Tile"):
        geometry.stair_layout(.9, config(size=(5., 3.)))
