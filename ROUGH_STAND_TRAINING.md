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
- Reset 采用分层采样：25% 中央平地、50% 当前课程等级、25% 历史低等级回放；只有当前等级崎岖样本影响课程升降级。
- 25% reset 使用较宽的合成 handoff 分布，近似 Init→Stand 和 Walk→Stand 时的姿态、速度、关节状态及上一动作；其余 reset 仍带温和扰动。
- 重置高度为出生点附近 0.8×0.8 m 网格最高地面点 +0.42 m。该余量用于温和落脚，不是 IK 四足贴地初始化。
- 高度奖励改为机身世界高度减附近地面平均高度，目标仍为 0.40 m。其余奖励和权重沿用 Stand。
- 增加左右关节镜像惩罚（权重 -1.0）：髋外展偏移左右反号，髋俯仰和膝关节偏移左右同号；误差在 0.10 rad 死区内不惩罚，以保留崎岖地形上的必要补偿。
- 地面查询复用与碰撞网格相同的确定性高度场并按三角面插值，仅用于重置与奖励；不创建额外 GPU 网格，不加入 actor/critic 观测，也不要求实机增加传感器。
- 推扰保持 5～8 秒间隔，水平速度和偏航角速度扰动范围减为 ±0.1；后续站稳后可单独调大。

## 站稳课程

每回合 15 秒，前 0.5 秒为落脚稳定期。此后满足下列稳定条件的时间占比达到 90%，四足同时接触占比达到 90%，左右对称占比达到 80%，累计考察时间至少 14.4 秒且没有物理失败，视为成功：

- 水平速度 ≤0.15 m/s；偏航角速度 ≤0.25 rad/s。
- 机身倾角 ≤0.10 rad；距出生点水平漂移 ≤0.20 m。
- 四个足端各自在三帧接触历史中的合力超过 5 N，才计为该时刻四足接触。
- 六组左右镜像关节对的 RMS 误差 ≤0.20 rad，才计为该时刻左右对称。

当前等级的自适应崎岖样本连续成功 2 次升级；跌倒、漂移超过 0.65 m，或完整回合稳定率、四足接触率、左右对称率不达标则降级。未完成的非失败回合保持等级并清空成功计数。平地和历史低等级回放不影响等级或成功计数。最高级保持最高级，不随机跳回低等级。

观察课程指标 `level`、`sampled_level`、`success_rate`、`stable_fraction`、`four_feet_fraction`、`symmetry_fraction`、`symmetry_error`、`max_drift`；除 `level` 外均统计本次重置批次，并非整场训练的累计指标。更改回合长度时同时调整 `min_duration`。

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

检查真实地面查询、分层/切换状态重置、四足接触课程、高度奖励、48 维观测动作兼容性、升降级、最高级钳制、漂移失败和有限值物理步进。短验证不代表策略已学会崎岖站立。
