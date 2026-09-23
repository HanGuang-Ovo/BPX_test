# 崎岖行走 H=10 历史观测

任务：`BPX-Locomotion-Rough-History-v0`；日志：`logs/rsl_rl/bpx_rough_history_asymmetric/`。
复用 `BPX-Locomotion-Rough-v0` 的地形、指令、复位、动作与 PPO 参数。
观测和奖励在 `bpx_rough_history_env_cfg.py` 中显式定义，不继承单帧配置；奖励初始参数保持一致，可独立调整。
采用非对称 Actor-Critic：`policy` 为 450 维，`critic` 为 480 维，隐藏层均为 `[128, 128, 128]`。
只有 Critic 接收机身坐标系基座线速度真值（`root_lin_vel_b`）；其他观测项沿用 Actor 的噪声配置，两组独立采样噪声。
旧单帧任务保留；历史版本从头训练，不能直接 `--resume` 旧的 48 维或对称 480 维 Actor checkpoint。

## 观测契约

Actor 每帧 45 维：角速度、重力投影、速度指令、相对关节位置、相对关节速度、上一拍动作。
Critic 每帧在这些观测前增加 3 维无噪声基座线速度，共 48 维；两者均保留 10 帧历史。
H=10 包含当前帧，以 50 Hz 更新，首末帧跨度 0.18 秒。当前状态 `s_t` 配对先前执行的 `a_(t-1)`，策略输出 `a_t`。

Isaac Lab 为每个观测项维护历史，再逐项展开和拼接。它不是十个完整 48 维向量的直接拼接：

| Actor 输入切片 | Critic 输入切片 | 内容，每项内部从最旧到最新 |
| --- | --- | --- |
| 不提供 | `[0:30]` | 10×3 基座线速度真值 |
| `[0:30]` | `[30:60]` | 10×3 基座角速度 |
| `[30:60]` | `[60:90]` | 10×3 重力投影 |
| `[60:90]` | `[90:120]` | 10×3 速度指令 |
| `[90:210]` | `[120:240]` | 10×12 相对关节位置 |
| `[210:330]` | `[240:360]` | 10×12 相对关节速度 |
| `[330:450]` | `[360:480]` | 10×12 上一拍动作 |

复位后首次观测填满该环境的历史窗口；其他并行环境不受影响。
训练噪声在每次新观测入缓存前施加，旧历史不重复加噪。读取已计算的历史不推进缓存。
关节顺序、动作缩放和默认角沿用单帧基线。

## 验证与训练

在原有 Isaac Lab 环境中执行；本机如需扩展路径，先 `source /home/hanguang/Isaac_Sim/setup_conda_env.sh`。

```bash
python scripts/validate_rough_task.py --history --headless
python scripts/rsl_rl/train.py --task BPX-Locomotion-Rough-History-v0 \
  --num_envs 8 --max_iterations 2 --headless
```

验证覆盖维度、首次填充、历史移位、动作时间关系、单环境复位隔离、训练/部署逐元素对齐，以及原有地形课程。
正式训练示例：

```bash
python scripts/rsl_rl/train.py --task BPX-Locomotion-Rough-History-v0 \
  --num_envs 512 --headless
```

与单帧基线使用相同训练预算和评估地形，比较通过率、速度跟踪误差、跌倒率与动作变化。
短训练只验证流程，不代表已学会崎岖行走或取得性能提升。

## 导出与 MuJoCo

使用历史任务回放 checkpoint，现有 play 脚本会导出 450 输入、12 输出的 TorchScript 和 ONNX：

```bash
python scripts/rsl_rl/play.py --task BPX-Locomotion-Rough-History-v0 \
  --num_envs 1 --checkpoint logs/rsl_rl/bpx_rough_history_asymmetric/YOUR_RUN/model_1499.pt
```

将 `sim2sim/config/bpx_terrain_history.toml` 中两处 `YOUR_RUN` 替换为实际目录。
该配置使用趴姿复位，`dimension=450`、`history_length=10`、`include_base_lin_vel=false`。
配合 Supervisor 与自动 Init 完成起身；旧配置默认包含线速度且 H=1，保持兼容。
导出模型本身不包含历史缓存，调用方必须按上述契约构造 450 维输入。

```bash
python sim2sim/validate_policy_export.py --config sim2sim/config/bpx_terrain_history.toml
python sim2sim/run_mujoco.py --config sim2sim/config/bpx_terrain_history.toml \
  --backend onnx --supervisor --auto-init --duration 20
python -m unittest discover -s sim2sim -p 'test_*.py' -v
```

MuJoCo 每个策略周期更新一次历史，每次运行复位后用首帧填满缓存。
Supervisor 启用时，在 Stand/Init 期间持续维护历史，切回行走时沿用这些真实记录；
Stand/Init 仍接收 48 维单帧观测，只有行走策略（包括站立回退到行走策略的情况）接收历史。
历史中的 action 是混合后实际施加的动作。历史指令记录当时各模式实际使用的指令，Init 期间为零。
历史策略在切换场景下的表现仍需训练完成后单独评估。

新旧模型不可混用：TOML 中的 `YOUR_RUN` 是新模型占位路径，需在训练并导出后更新。
本次仅改变 rough history 行走策略；Stand/Init 的 Actor 和 Supervisor 停车判断仍使用线速度。
奖励与终止条件可以继续读取仿真真值，它们不属于 Actor 的输入。
