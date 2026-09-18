# 崎岖站立训练

任务 `BPX-Stand-Rough-v0` 从 `BPX-Stand-v0` 继承 48 维单帧观测、12 维动作、零速度指令和 PPO 网络。日志独立写入 `logs/rsl_rl/bpx_rough_stand/`。

## 微调已有 Stand checkpoint

在 Isaac Lab Python 环境、项目根目录运行：

```bash
python scripts/rsl_rl/train.py \
  --task BPX-Stand-Rough-v0 --num_envs 512 --headless \
  --resume \
  --checkpoint logs/rsl_rl/bpx_stand/2026-09-15_00-13-24_stable/model_499.pt \
  --max_iterations 1500
```

`--max_iterations` 是本次继续训练的迭代数。恢复模型与优化器；地形课程从最低级重新开始，课程状态不随 PPO checkpoint 保存。

## 地形、重置与奖励

配置：`source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_rough_stand_env_cfg.py`。

- 复用行走地形：8×8 m 地块，8 级振幅 ±0.5～±4 cm，波长 0.15～0.30 m。
- 90% 回合在地块局部 x=±2.5 m、y∈[-0.6,0.6] m 起伏区域出生，随机朝向；10% 在中央平地出生，保留平地能力。
- 重置高度为出生点附近 0.8×0.8 m 网格最高地面点 +0.42 m，初速度为零，关节偏移 ±0.03 rad。该余量用于温和落脚，不是 IK 四足贴地初始化。
- 高度奖励改为机身世界高度减附近地面平均高度，目标仍为 0.40 m。其余奖励和权重沿用 Stand。
- 地面查询直接对仿真地形网格投射射线，仅用于重置与奖励，不加入 actor/critic 观测，不要求实机增加传感器。
- 推扰保持 5～8 秒间隔，水平速度和偏航角速度扰动范围减为 ±0.1；后续站稳后可单独调大。

## 站稳课程

每回合 15 秒，前 0.5 秒为落脚稳定期。此后满足下列所有条件的时间占比达到 90%，且累计考察时间至少 14.4 秒、没有物理失败，视为成功：

- 水平速度 ≤0.15 m/s；偏航角速度 ≤0.25 rad/s。
- 机身倾角 ≤0.25 rad；距出生点水平漂移 ≤0.20 m。

崎岖样本连续成功 2 次升级；跌倒、漂移超过 0.65 m，或完整回合稳定率不达标则降级。未完成的非失败回合保持等级并清空成功计数。平地回合不影响等级和成功计数。最高级保持最高级，不随机跳回低等级。

观察课程指标 `level`、`success_rate`、`stable_fraction`、`max_drift`；后三项统计本次重置批次，包含平地样本，并非整场训练的累计指标。更改回合长度时同时调整 `min_duration`。

## 回放、导出与 MuJoCo

```bash
python scripts/rsl_rl/play.py \
  --task BPX-Stand-Rough-v0 --num_envs 4 \
  --checkpoint logs/rsl_rl/bpx_rough_stand/<run>/model_<n>.pt
```

导出的策略位于该 run 的 `exported/policy.onnx`。在 `sim2sim/config/bpx_terrain.toml` 修改 `stand_policy` 为该路径，保留已有的 `locomotion_policy` 和 `init_policy`：

```bash
python sim2sim/run_mujoco.py \
  --config sim2sim/config/bpx_terrain.toml \
  --backend onnx --viewer --gamepad --supervisor --duration 120
```

验证平地/轻微起伏/较大起伏上的持续站立，以及行走→停止→站立→再行走。标准站姿训练不能替代切换验证；若接管不稳定，再扩展初始速度和姿态分布。

固定最高难度回放可在 play 命令后追加：

```bash
env.curriculum.terrain_levels=null \
env.scene.terrain.terrain_generator.difficulty_range='[0.875,1.0]' \
env.events.reset_base.params.flat_probability=0.0
```

## 实现验证

```bash
python -u scripts/validate_rough_stand_task.py --headless
```

检查真实地面查询、平地/崎岖重置、高度奖励、观测动作兼容性、升降级、最高级钳制、漂移失败和有限值物理步进。短验证不代表策略已学会崎岖站立。
