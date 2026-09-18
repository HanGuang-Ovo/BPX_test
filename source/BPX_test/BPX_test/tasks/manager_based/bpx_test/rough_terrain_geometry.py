"""NumPy geometry for BPX blind rough locomotion (metres, no simulator dependency)."""
import numpy as np


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def terrain_amplitude(difficulty, levels, amplitude_range):
    """Quantize generator row jitter so the easiest/hardest rows reach exact bounds."""
    level = min(int(np.clip(difficulty, 0.0, 1.0) * levels), levels - 1)
    return amplitude_range[0] + level / (levels - 1) * (amplitude_range[1] - amplitude_range[0])


def wave_height_field(difficulty, cfg):
    """Flat central spawn square, smooth transition, rough annulus, flat tile seams."""
    amplitude = terrain_amplitude(difficulty, cfg.levels, cfg.amplitude_range)
    nx, ny = (round(side / cfg.horizontal_scale) + 1 for side in cfg.size)
    x, y = np.meshgrid(np.linspace(0, cfg.size[0], nx), np.linspace(0, cfg.size[1], ny), indexing="ij")
    # TerrainGenerator supplies seed/difficulty. Row jitter differentiates columns deterministically.
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed or 0), int(difficulty * 1e9)]))
    heights = np.zeros_like(x)
    for _ in range(24):
        angle = rng.uniform(0, 2 * np.pi)
        wavelength = rng.uniform(*cfg.wavelength_range)
        heights += np.sin(2 * np.pi / wavelength * (x * np.cos(angle) + y * np.sin(angle))
                          + rng.uniform(0, 2 * np.pi))
    heights /= max(np.max(np.abs(heights)), 1e-12)
    radius = np.maximum(np.abs(x - cfg.size[0] / 2), np.abs(y - cfg.size[1] / 2))
    spawn_mask = smoothstep((radius - cfg.platform_width / 2) / cfg.transition_width)
    edge_distance = np.minimum.reduce([x, y, cfg.size[0] - x, cfg.size[1] - y])
    edge_mask = smoothstep(edge_distance / cfg.transition_width)
    return amplitude * heights * spawn_mask * edge_mask


def wave_terrain(difficulty, cfg):
    """Isaac Lab SubTerrainBaseCfg callback: trimesh list and exactly flat origin."""
    import trimesh

    z = wave_height_field(difficulty, cfg)
    nx, ny = z.shape
    x, y = np.meshgrid(np.linspace(0, cfg.size[0], nx), np.linspace(0, cfg.size[1], ny), indexing="ij")
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
    a = (np.arange(nx - 1)[:, None] * ny + np.arange(ny - 1)[None, :]).ravel()
    # Counter-clockwise faces, positive Z normals. No int16 height quantization.
    faces = np.concatenate((np.column_stack((a, a + ny, a + ny + 1)),
                            np.column_stack((a, a + ny + 1, a + 1))))
    return [trimesh.Trimesh(vertices=vertices, faces=faces, process=False)], np.array([cfg.size[0]/2, cfg.size[1]/2, 0.0])
