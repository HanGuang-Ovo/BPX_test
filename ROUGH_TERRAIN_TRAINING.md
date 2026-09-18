# BPX 崎岖地形第一阶段训练

任务：`BPX-Locomotion-Rough-v0`；日志：`logs/rsl_rl/bpx_rough/`。
该任务继承平地行走的全部奖励（含权重）、48 维单帧观测、12 维动作、PPO 网络及参数。
没有高度扫描、历史帧或新增策略传感器。原平地、Stand、Init 任务保持原样。

## 地形和复位

每块地形 8×8 m，中心为高度零的 2×2 m 平台。机器人根位置在平台中心 ±0.1 m 内随机，
朝向随机；初始根速度为零，根高度和关节姿态沿用直立行走复位。关节位置随机化、摩擦、PD 增益和周期推力沿用原任务。

平台外有 0.5 m 平滑过渡，之后是随机波叠加的起伏地形。波长 15～30 cm、网格间距 2.5 cm，
与当前 MuJoCo 的起伏生成方式一致，但不是相同随机路面。所有复位均回到当前难度的平坦平台。

8 级地形幅度上限依次为 ±0.5、±1、±1.5、±2、±2.5、±3、±3.5、±4 cm。
这些是高度绝对值上限，不保证每块路面达到正负峰值。波形归一化后再乘平台/边缘遮罩。
每一级有 4 个波形变体，由固定地形种子控制。所有机器人从第 0 级起步。

地形参数集中在 `bpx_rough_env_cfg.py` 的 `BpxWaveTerrainCfg`：

- `amplitude_range`：最易/最难幅度，单位 m。
- `wavelength_range`：波长范围，单位 m。
- `levels`：8，需与生成器 `num_rows` 一致。
- `platform_width` / `transition_width`：平坦平台宽度和过渡宽度。

MuJoCo 已同步为 ±2/±4 cm，并从统一幅度参数自动计算高度场缩放、偏移、归一化和标签。
训练端最高幅度也是 ±4 cm，但两端波形种子和场地布局不同，应分别评估通过率。

## 指令采样

使用原生 `UniformVelocityCommand`，每 10 秒独立均匀采样：

- `vx ∈ [-1, 1] m/s`
- `vy ∈ [-0.5, 0.5] m/s`
- `wz ∈ [-1, 1] rad/s`

没有专门的站立或原地旋转概率分支；连续均匀分布仍可能自然采到接近零的分量。

## 根据可通过性升降级

课程在回合结束时评估，而不是按训练迭代数自动变难。
机身离平台中心的最大轴向距离大于 1.95 m 才统计崎岖行走，预留平台、过渡带及机身长度的余量。
统计的是逐步累计路程，支持转弯；每步累计值有限幅，并排除复位跳变。

一个成功回合同时满足：

1. 崎岖区域累计路程至少 1 m，累计时间至少 1 秒。
2. 整个回合平均平面速度向量误差 ≤0.35 m/s，平均偏航速度误差 ≤0.5 rad/s。
3. 没有躯干触地或过度倾斜导致的失败终止。

连续两次成功升一级；物理失败降一级。回合持续至少 1 秒但平均跟踪误差超限也降一级。
跟踪正常但未充分进入崎岖地形的回合保持等级并清空成功连续计数。
最高等级成功后保持最高等级，不随机退回低等级。

任一轴距平台中心达到 3.4 m 时按 time-limit 截断并复位，防止进入相邻难度格。
该截断使用价值 bootstrap，不添加奖励或失败惩罚。它不等于成功，仍需满足上述通过条件。
原回合上限仍是 20 秒。课程难度/连续成功计数属于环境状态，不随 PPO checkpoint 保存；恢复训练从最低级开始。

TensorBoard 可查看：

- `Curriculum/terrain_levels/level`：全体平均等级，编号 0～7。
- `Curriculum/terrain_levels/success_rate`：本次复位批次成功率，不是全训练累计通过率。
- `Curriculum/terrain_levels/rough_distance`、`rough_time`：复位批次的平均崎岖路程和时间。
- 速度跟踪误差、各奖励分项及 `Episode_Termination/terrain_boundary`。

## 验证与训练

在原有 Isaac Lab Python 环境中，从仓库根目录执行。
本机如果 Conda 环境需要 Isaac Sim 扩展路径，可先执行：

```bash
conda activate isaaclab
source /home/hanguang/Isaac_Sim/setup_conda_env.sh
```

验证几何、配置契约和真实管理器升降级：

```bash
python scripts/validate_rough_task.py --headless
```

验证小规模训练流程：

```bash
python scripts/rsl_rl/train.py --task BPX-Locomotion-Rough-v0 \
  --num_envs 8 --max_iterations 2 --headless
```

从已有平地 checkpoint 微调（替换实际 checkpoint 路径）：

```bash
python scripts/rsl_rl/train.py --task BPX-Locomotion-Rough-v0 \
  --num_envs 512 --headless --resume \
  --checkpoint logs/rsl_rl/bpx_locomotion/<原运行目录>/model_<编号>.pt
```

新增支持：`--resume --checkpoint <文件路径>` 可跨实验加载；新日志仍写入 `bpx_rough`。
加载会保留现有 runner 的模型/优化器恢复行为。网络维度与原任务相同，无需部分加载。
原来 `--resume --load_run ... --checkpoint model_....pt` 的名称/正则查找方式仍可使用。

从头训练可省略 `--resume --checkpoint ...`。这里只验证短训练，没有替你启动长时间训练。

回放并导出：

```bash
python scripts/rsl_rl/play.py --task BPX-Locomotion-Rough-v0 \
  --num_envs 4 --checkpoint logs/rsl_rl/bpx_rough/<运行目录>/model_<编号>.pt
```

回放默认也从最低难度开始，适合观察课程，但不能作为固定难度评估。
若要固定最高难度并禁用升降级，可追加 Hydra 覆盖：

```bash
  env.curriculum.terrain_levels=null \
  env.scene.terrain.max_init_terrain_level=7 \
  env.scene.terrain.terrain_generator.difficulty_range='[0.875,1.0]'
```

这里令全部地形行都使用最高幅度，避免只设置 `max_init_terrain_level=7` 却随机落到低等级。
对比策略时使用相同地形种子、指令和等级，并另外回归平地行走效果。
