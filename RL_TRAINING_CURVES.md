# BPX 训练曲线解读指南（PPO / TensorBoard）

本文覆盖行走、站立和趴卧起身三个任务。指标名称、奖励累计方式及检查点保存行为已按 2026-09-17 本机 Isaac Lab 2.2.1 和 RSL-RL 实现核对；升级依赖后应重新核对实际日志。任务参数见 [环境配置说明](BPX_ENV_SETUP.md)。

## 1. 打开正确的实验

在项目根目录执行：

```bash
# 同时查看所有任务
tensorboard --logdir logs/rsl_rl

# 只比较行走任务的运行
tensorboard --logdir logs/rsl_rl/bpx_locomotion
```

| 任务 | 默认实验目录 | 回合时长 | 最大策略步数 |
| --- | --- | ---: | ---: |
| Locomotion | `bpx_locomotion` | 20 秒 | 1000 |
| Stand | `bpx_stand` | 15 秒 | 750 |
| Init | `bpx_init` | 4 秒 | 200 |
| 兼容行走入口 BPX-Test | `bpx_flat` | 20 秒 | 1000 |

浏览器打开终端给出的地址，通常是 `http://localhost:6006`。每次 PPO 迭代每个环境采集 24 个策略步，再更新网络并写日志；不是全体环境总共只运行 24 步。回合指标需要有环境结束后才能产生。

平滑曲线有助于看趋势，但应同时查看原始曲线，避免高平滑度掩盖突然退化。不同任务、奖励权重或指令分布的总奖励不宜直接排名。

## 2. 常用指标与准确名称

当前实现常见 tag 如下；TensorBoard 中以实际 tag 为准。

| 指标 | 含义与用途 |
| --- | --- |
| `Train/mean_reward` | 最近完成回合的总回报均值；不是单步奖励 |
| `Train/mean_episode_length` | 最近完成回合的策略步数均值 |
| `Episode_Reward/<奖励项名>` | 加权、乘 dt 后的回合分项累计，再除以配置回合时长 |
| `Episode_Termination/<终止项名>` | 各终止条件的统计，结合回合长度分析 |
| `Metrics/base_velocity/error_vel_xy` | 指令生成器累计的平面速度误差指标 |
| `Metrics/base_velocity/error_vel_yaw` | 指令生成器累计的偏航速度误差指标 |
| `Policy/mean_noise_std` | 策略动作分布的平均标准差，不是观测噪声 |
| `Loss/value_function` | 价值函数损失 |
| `Loss/surrogate` | PPO 策略目标对应的损失 |
| `Loss/learning_rate` | 自适应学习率 |
| `Perf/total_fps` | 总采样吞吐量 |

仓库没有名为 `total` 的奖励项，不能把 `Episode Rewards/total` 当作必定存在的 tag。奖励分项前缀是 `Episode_Reward/`，回合长度 tag 中的 `mean_episode_length` 为小写。

`Episode_Termination/*` 是管理器在日志采样时统计的终止标记均值，不能未经核对就当作完整评估集的失败概率。

速度误差指标由 command 模块按其重采样周期归一化，提前终止时可能缩短累计时间，因此不宜直接当作固定时长评估的完整轨迹 RMSE。物理性能比较应另用固定指令、固定时长与实际状态日志计算。

## 3. 奖励数值怎样计算

Isaac Lab 的 RewardManager 每个策略步执行：

```text
step_reward_i = reward_function_i(state) × weight_i × step_dt
step_dt = 0.02 s
```

回合结束时，分项日志近似为：

```text
Episode_Reward/i = mean(本次结束回合的 Σ step_reward_i) / 配置的回合时长
```

分母是任务配置的最大回合秒数，不是该回合实际持续秒数。RSL-RL 还会对一次迭代收集到的这些日志求均值；`Train/mean_reward` 使用最近完成回合的回报缓冲区，两者的统计窗口和平均方式不完全相同。

例如行走策略连续 20 秒完美跟踪，只看权重为 1.0 和 0.5 的两项速度奖励：

```text
单步贡献 = (1.0 + 0.5) × 0.02 = 0.03
1000 步累计贡献 = 30
对应分项日志约为 1.0 和 0.5
```

这只是两项跟踪奖励的理想贡献，不含腾空奖励与其他惩罚。不能忽略 dt 后把它写成“回合奖励约 1500”。提前终止会影响累计值，但奖励曲线变化也可能来自权重、指令或状态分布变化。

## 4. 三个任务分别看什么

### 4.1 行走

重点关注 `Episode_Reward/track_lin_vel_xy_exp`、`track_ang_vel_z_exp`、两项速度误差以及躯干接触/倾角终止。回合长度接近 1000 表示很少提前结束，但不保证准确跟踪或步态质量。

当前指令含 10% 静止、25% 纯旋转和 65% 混合运动。比较新旧实验时先确认采样比例是否相同。纯旋转时，现用 `feet_air_time` 因平面速度指令为零而关闭，不能用该项单独评价转向步态。

### 4.2 站立

关注 `track_zero_lin_vel`、`track_zero_ang_vel`、`base_height_l2`、`flat_orientation_l2`、`feet_slide`、`joint_vel_l2` 和 `action_rate_l2`。同时回放检查推扰后的恢复及是否原地小踏步。回合长度上限为 750，不是 1000。

### 4.3 趴卧起身

Init 只因超时或关节位置越界结束，因此即使一直趴着也可能达到 200 步。主要看：

- `Episode_Reward/height_progress`：抬升进度。
- `Episode_Reward/recovered`：满足高度、姿态、四足接触及前足宽度条件的时间累计奖励。
- `Episode_Reward/recovered_stability`：完成起身后的稳定程度。
- `Episode_Reward/joint_torque_limit_l2`：力矩超限惩罚，权重会随课程变化。
- `Curriculum/joint_torque_limit/success_rate`：最近回合**最后一帧**满足 `recovered` 的比例。
- `Curriculum/joint_torque_limit/weight`：当前力矩惩罚权重。
- `Curriculum/joint_torque_limit/window_episodes`：已记录的窗口回合数。

`recovered` 奖励不是“成功率”。窗口最多 8192 个回合，至少收集 4096 个后，根据 70%～90% 的终帧成功率把力矩惩罚从 -0.1 调整到 -1.0；成功率回落时惩罚可减弱。强化约束可能导致总奖励下降，即使起身能力没有退化，也不能据此直接判断策略崩溃。

训练 `recovered` 使用倾角 ≤0.1 rad 和前足宽度 ≥0.24 m；部署 Supervisor 使用倾角 <0.25 rad、速度约束、四足接触及 0.50 秒连续确认。训练成功率不能直接替代部署起身成功率。

## 5. 如何理解震荡与中途回落

PPO 的样本分布、价值目标和自适应学习率都在变化，因此损失不必单调下降。两条 loss 仍有诊断价值：持续异常放大、非有限值或与回报同时退化时，应检查奖励尺度、数值稳定性和更新幅度。没有通用的“必须超过初值几十倍才异常”阈值。

奖励“先升、再降、再升”可能来自行为探索，也可能来自课程权重变化、指令难度变化、策略退化或统计窗口。仅凭曲线形状无法断定机器人正在从站立过渡到迈步，也不能把回落一律视为好信号。

建议按以下顺序核对：

1. 比较 `params/`、训练命令与代码版本，确认奖励权重和采样分布。
2. 查看终止原因、回合长度、物理误差及 Init 课程权重是否同步变化。
3. 在相同指令、初始状态和噪声设置下回放回落前后检查点。
4. 用独立评估记录成功率、滑移、力矩和速度误差，再判断是否恢复训练或调整参数。

动作标准差下降只表示探索分布变窄，不证明学会目标行为；保持较高标准差也不必然表示失败。

## 6. 检查点保存与正确回放

本机 `OnPolicyRunner.learn()` 除了每 `save_interval=50` 次迭代保存，还会在训练循环正常结束后保存最终模型。因此从零开始正常完成 1500 次迭代时，通常有 `model_1499.pt`；Init 默认 3000 次时通常有 `model_2999.pt`。异常退出则以磁盘中实际存在的文件为准。

`play.py --checkpoint` 接收文件路径，不会把裸文件名自动拼到 `--load_run` 目录。请替换下列占位路径：

```bash
python scripts/rsl_rl/play.py --task BPX-Locomotion-v0 --num_envs 1 --real-time \
    --checkpoint "logs/rsl_rl/bpx_locomotion/<运行目录>/model_400.pt"

python scripts/rsl_rl/play.py --task BPX-Locomotion-v0 --num_envs 1 --real-time \
    --checkpoint "logs/rsl_rl/bpx_locomotion/<运行目录>/model_1499.pt"
```

`--load_run` 在未直接提供 `--checkpoint` 时参与检查点搜索。回放会重新导出 `policy.pt` 和 `policy.onnx`；若比较多个检查点，注意同一运行目录内的 `exported/` 会被后一次导出覆盖。

`play.py` 默认复用训练环境配置；正式对比需固定初始条件、指令与随机化。速度箭头只显示跟踪方向，不能代替对完整指令范围的量化测试。

## 7. 调参前的判断

| 现象 | 先核对 |
| --- | --- |
| 行走回合长期很短 | 躯干接触/倾角终止、初始扰动、映射与 PD |
| 站立满回合但小踏步 | 足滑、关节速度、动作变化与推扰恢复 |
| Init 满 200 步但不起身 | `recovered`、终帧成功率及回放姿态 |
| Init 成功率提高而总奖励回落 | 力矩课程权重是否正在加强 |
| loss 和回报同时突变 | 学习率、奖励量纲、非有限状态和最近修改 |
| `feet_air_time` 略为负 | 腾空时间、指令类别和落足事件，而非直接增大权重 |

需要降低难度时，可减少初始扰动或调整指令范围，但行走任务的偏航范围必须仍覆盖 `pure_rotation_speed_range`。专门练静止站立应使用 Stand 任务；小幅非零速度范围并不等于站立任务。

每次改变一个主要变量，保留原始日志和模型，再用相同评估条件比较结果。课程、随机化和步态改进方向见 [改进路线图](BPX_IMPROVEMENT_ROADMAP.md)。
