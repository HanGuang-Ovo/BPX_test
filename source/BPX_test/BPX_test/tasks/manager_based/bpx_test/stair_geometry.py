# -*- coding: utf-8 -*-
"""实体楼梯几何生成模块(不依赖 Isaac Sim,可独立单元测试)。

本模块负责三件事:
1. ``stair_layout``   —— 按难度确定性地采样一段楼梯的几何参数(台阶边缘/高度/进深);
2. ``layout_height``  —— 给定 (x, y) 精确求解楼梯阶梯高度函数(含竖直立面与侧向落差);
3. ``stair_terrain``  —— Isaac Lab 地形生成器回调,用实体盒子(trimesh)拼出楼梯网格。

整体设计约定:
- 楼梯是"实体"几何:每级台阶都是一个从公共地面一直延伸到台阶顶面的长方体,
  竖直踢面(reser/riser)保持竖直,不会像 heightfield 地形那样出现斜坡近似。
- 一块地形 tile 内的行走序列为:平地接近段 -> 上楼梯 -> 顶部平台 -> 下楼梯 -> 平地退出段。
- 所有随机采样都由种子驱动,保证同一 seed 下几何完全可复现;
  训练侧 (stairs.py) 通过 ``generator_layouts`` 复现完全相同的采样序列,
  从而在奖励/重置逻辑中查询每块 tile 的真实边缘坐标。
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class StairLayout:
    """一段楼梯 tile 的几何描述(均使用地形生成回调的 [0, size] 局部坐标)。

    属性说明:
        edges:   (2*count,) 各台阶边缘的 x 坐标(上楼梯段在前,下楼梯段在后);
        heights: (2*count,) 各台阶对应的站立高度(最后一级固定为 0,即回到地面);
        depths:  (2*count,) 各台阶的踏面进深(踢面之间的水平距离);
        width:   楼梯有效宽度(tile y 方向居中,两侧留出落差);
        count:   单侧台阶数(上、下各 count 级,平地 tile 时为 0)。
    """

    edges: np.ndarray
    heights: np.ndarray
    depths: np.ndarray
    width: float
    count: int


def stair_layout(difficulty, cfg):
    """按课程难度采样一块可复现的 tile:平地 -> 上行 -> 平台 -> 下行 -> 平地。

    坐标系采用地形生成回调的 [0, size] 约定(左下角为原点)。
    每一级踢面高度与踏面进深都独立采样;下行段的踢面高度是上行段高度的
    反向排列,保证走完下楼梯后地面高度精确回到 0。

    参数:
        difficulty: [0, 1] 课程难度,决定台阶高度区间在 initial_height_range
                    与 height_range 之间的插值位置;
        cfg:        BpxStairTerrainCfg 配置对象。
    """
    # 平地 tile:没有台阶,直接返回空布局(tile 全宽可行走)。
    if cfg.flat:
        return StairLayout(np.empty(0), np.empty(0), np.empty(0), cfg.size[1], 0)
    # 课程至少两级、台阶至少两级,否则无法构成"上-平台-下"结构。
    if cfg.levels < 2 or cfg.step_count_range[0] < 2:
        raise ValueError("Stairs need at least two curriculum levels and two steps")
    # 由难度映射到课程等级 level,再归一化为插值系数 alpha ∈ [0, 1]。
    level = min(int(np.clip(difficulty, 0, 1) * cfg.levels), cfg.levels - 1)
    alpha = level / (cfg.levels - 1)
    # 台阶高度区间随等级线性增长:低等级接近 initial_height_range(几乎平地),
    # 高等级达到 height_range(完整楼梯高度)。
    low, high = np.asarray(cfg.initial_height_range) * (1 - alpha) + np.asarray(cfg.height_range) * alpha
    # 用 (seed, difficulty) 组合出独立 RNG:同一配置下每块 tile 结果确定,
    # 且不同 difficulty 的 tile 互不干扰。
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed or 0), int(difficulty * 1e9)]))
    count = int(rng.integers(cfg.step_count_range[0], cfg.step_count_range[1] + 1))  # 台阶数
    rises = rng.uniform(low, high, count)            # 每级踢面高度(上行)
    up_depths = rng.uniform(*cfg.depth_range, count)    # 上行段踏面进深
    down_depths = rng.uniform(*cfg.depth_range, count)  # 下行段踏面进深
    width = float(rng.uniform(*cfg.width_range))        # 楼梯有效宽度
    # 上行踏面 + 顶部平台 + 下行踏面的总水平跨度(首末踢面按 0 计)。
    span = up_depths[:-1].sum() + cfg.landing_length + down_depths[:-1].sum()
    # 楼梯整体在 tile 内水平居中,first 为上行第一级边缘的 x 坐标。
    first = (cfg.size[0] - span) / 2
    # 可行性检查:必须留出足够的平地接近段,且楼梯宽度不超过 tile 宽度。
    if first < cfg.approach_margin or width > cfg.size[1]:
        raise ValueError("Tile cannot fit the flights, safe approaches and stair width")
    # 顶部平台必须容纳"下行接近段 + 上行退出段"两段安全裕量。
    if cfg.landing_length < 2 * cfg.approach_margin:
        raise ValueError("Landing must fit the descending approach and ascending exit")
    # 上行段各边缘 x 坐标:从 first 开始累加踏面进深(不含最后一级)。
    up_edges = first + np.r_[0, np.cumsum(up_depths[:-1])]
    # 下行段边缘:从上行最后一条边缘跨过顶部平台后继续累加。
    down_edges = up_edges[-1] + cfg.landing_length + np.r_[0, np.cumsum(down_depths[:-1])]
    # 各边缘对应的站立高度:
    #   上行段 —— rises 的前缀和(逐级抬高);
    #   下行段 —— 总高度减去"反转后的前缀和"(逐级降低,最后精确回到 0)。
    heights = np.r_[np.cumsum(rises), rises.sum() - np.cumsum(rises[::-1])]
    heights[-1] = 0.0  # 强制最后一级之后回到地面高度
    return StairLayout(np.r_[up_edges, down_edges], heights,
                       np.r_[up_depths, down_depths], width, count)


def layout_height(layout, x, y, size):
    """精确的阶梯高度函数,含竖直踢面边缘与楼梯两侧的落差。

    参数:
        layout: StairLayout 实例;
        x, y:   查询坐标(数组或标量,可广播);
        size:   tile 尺寸 (size_x, size_y)。

    返回:
        每个 (x, y) 处的支撑面高度:
        - x 在某级台阶边缘之后 -> 该级台阶高度;
        - x 在第一条边缘之前  -> 0(平地接近段);
        - |y - size_y/2| 超出楼梯半宽 -> 0(楼梯两侧直接跌落到地面)。
    """
    x, y = np.broadcast_arrays(x, y)
    # searchsorted 定位 x 落在哪一级台阶:side="right" 保证恰好落在边缘上时
    # 取"边缘之后"的高度,与实体几何的阶梯定义一致。
    indices = np.searchsorted(layout.edges, x, side="right") - 1
    # 平地 tile 没有台阶,任何位置高度都为 0。
    if not layout.count:
        return np.zeros_like(x, dtype=float)
    # x 在第一条边缘之前(indices < 0)时高度取 0,否则取对应台阶高度。
    height = np.where(indices >= 0, layout.heights[indices.clip(min=0)], 0.0)
    # 楼梯之外的侧向区域(y 方向)直接落到地面高度 0。
    return np.where(np.abs(y - size[1] / 2) <= layout.width / 2, height, 0.0)


def stair_terrain(difficulty, cfg):
    """Isaac Lab 地形生成器回调;竖直踢面保持竖直(不使用 heightfield 斜坡)。

    用 trimesh 实体盒子拼接楼梯:
    - 一个覆盖整个 tile 的薄地板(floor);
    - 每级台阶一个从地板下方延伸到台阶顶面的实体盒子(terrace)。

    返回:
        (meshes, origin):meshes 为 trimesh 网格列表,origin 为 tile 中心
        的 (x, y, z),其中 z 取中心点处的楼梯高度(Isaac Lab 用它抬高 tile 原点)。
    """
    import trimesh

    layout = stair_layout(difficulty, cfg)
    # 公共地板:15 cm 厚的盒子,顶面恰好位于 z=0。
    floor = trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], 0.15))
    floor.apply_translation((cfg.size[0] / 2, cfg.size[1] / 2, -0.075))
    meshes = [floor]
    # 每级实体台阶向下延伸 0.10 m 插入地板之下:
    # 内部共享面被埋在实体内部(不影响碰撞),暴露出的顶面/踢面坐标
    # 与 layout_height 的查询结果严格一致。
    for i, height in enumerate(layout.heights[:-1]):
        end = layout.edges[i + 1]
        # 台阶盒子:x 方向为本级踏面长度,y 方向为楼梯宽度,z 方向高度
        # 加上插入地板的 0.10 m 余量。
        box = trimesh.creation.box(extents=(end - layout.edges[i], layout.width, height + 0.10))
        # 盒子中心定位:x 取本级踏面中点,y 居中于 tile,z 使顶面对准台阶高度。
        box.apply_translation(((layout.edges[i] + end) / 2, cfg.size[1] / 2, (height - 0.10) / 2))
        meshes.append(box)
    center = (cfg.size[0] / 2, cfg.size[1] / 2)
    # tile 中心高度作为 origin z:Isaac Lab 会把整块 tile 抬到该高度,
    # 但网格顶点本身使用世界 z=0 作为地面基准(见 stairs.ground_height 的注释)。
    return meshes, np.array([*center, float(layout_height(layout, *center, cfg.size))])


def generator_layouts(cfg):
    """逐位精确复现 Isaac Lab 2.2.1 按列优先的课程 RNG 采样序列。

    Isaac Lab 生成地形时按"列 -> 行"的顺序为每块 tile 抽取难度;
    此处以相同顺序、相同 seed 重放整条采样链,重建每块 tile 的
    StairLayout。训练侧重置/奖励逻辑(stairs.py)据此查询真实几何。

    注意:这些地形数值只用于仿真侧的重置与奖励计算,
    不会以任何形式进入策略的观测输入(保持盲走特性)。

    参数:
        cfg: TerrainGeneratorCfg(必须启用 curriculum 且提供 seed)。

    返回:
        (layouts, flat_columns):
        - layouts[row][col]:每块 tile 的 StairLayout;
        - flat_columns:平地 tile 所在的列编号列表。
    """
    if not cfg.curriculum or cfg.seed is None:
        raise ValueError("Stair queries require a seeded curriculum generator")
    sub_cfgs = list(cfg.sub_terrains.values())
    # 按各子地形 proportion 归一化得到累积分布,用于按列选择子地形类型。
    proportions = np.array([sub.proportion for sub in sub_cfgs])
    cumulative = np.cumsum(proportions / proportions.sum())
    rng = np.random.default_rng(cfg.seed)  # 与 Isaac Lab 内部同源的难度 RNG
    layouts = [[None] * cfg.num_cols for _ in range(cfg.num_rows)]
    flat_columns = []
    # 列优先遍历:与 Isaac Lab 生成器逐列抽取 difficulty 的顺序一致,
    # 确保 rng.uniform() 的消耗次序完全相同。
    for col in range(cfg.num_cols):
        # 该列落在累积分布的哪个子地形区间,即选用哪种子地形配置。
        sub = sub_cfgs[int(np.flatnonzero(col / cfg.num_cols + .001 < cumulative)[0])].copy()
        # 生成器会覆盖子地形的 seed 与 size,这里同样显式写入。
        sub.seed, sub.size = cfg.seed, cfg.size
        if sub.flat:
            flat_columns.append(col)
        for row in range(cfg.num_rows):
            # 行难度 = 行索引加 [0,1) 均匀抖动后归一化,再线性映射到难度区间:
            # 行号越大难度越高,同列内形成课程阶梯。
            fraction = (row + rng.uniform()) / cfg.num_rows
            difficulty = cfg.difficulty_range[0] + np.ptp(cfg.difficulty_range) * fraction
            layouts[row][col] = stair_layout(difficulty, sub)
    return layouts, flat_columns
