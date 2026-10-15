# BPX 上下楼训练：独立 Move / Hold 策略

新训练使用两个独立任务、Actor/Critic、优化器和日志目录，共用 50 Hz、450 维 Actor 历史观测、528 维 Critic 和 12 维关节位置动作。

| 任务 | 目标 | 默认初始化 | 日志目录 |
| --- | --- | --- | --- |
| `BPX-Stairs-Move-v0` | 起步、速度变化、连续上/下楼，不安排停驻 | 原行走 Actor；Critic/优化器全新 | `logs/rsl_rl/bpx_stairs_move/` |
| `BPX-Stairs-Hold-v0` | 真实行走状态接管、制动、收脚和保持 | rough-stand/model_1998.pt 的站立 Actor 迁移；冻结 Move 仅负责预运行 | `logs/rsl_rl/bpx_stairs_hold/` |

本文是当前上下楼训练的主文档。Move 负责起步、速度变化与连续通过，Hold 负责接管后的制动、收脚与保持；两者分别训练、保存模型和验收。走→停→再走由上层状态机衔接，无需第三项独立 PPO 训练。

## 共享接口与代码位置

| 项目 | 当前设置 |
| --- | --- |
| 控制与物理频率 | 策略 50 Hz；物理 200 Hz，`dt=0.005`、`decimation=4` |
| Actor | 角速度、重力投影、速度指令、相对关节位置、关节速度、上一实际动作；每项 10 帧，共 450 维 |
| 历史排列 | 观测项依次拼接，每项内部从最旧帧到最新帧 |
| Critic | 无噪声本体历史 480 维，加当前地形/承重/任务状态 48 维，共 528 维 |
| 动作 | 固定关节顺序的 12 维位置动作，缩放 0.5，加默认关节角 |
| 默认关节角 | hip roll 为 0，hip pitch 为 0.7 rad，knee 为 −1.4 rad |
| 执行器 | 显式延迟 PD，Kp=40、Kd=1，力矩上限 30 N·m，速度上限 20 rad/s |

Actor 保持盲走接口。地形高度、停车点、世界位置误差及任务内部阶段只用于训练调度、奖励、验收或 Critic。

以下配置路径相对于 `source/BPX_test/BPX_test/tasks/manager_based/bpx_test/`：

| 文件 | 内容 |
| --- | --- |
| `bpx_stairs_common_cfg.py` | 地形、机器人、传感器、动作和观测契约 |
| `bpx_stairs_move_env_cfg.py` | Move 指令、奖励与通过课程 |
| `bpx_stairs_hold_env_cfg.py` | Hold 指令、奖励、保持验收与停驻课程 |
| `agents/bpx_stairs_move_ppo_cfg.py` | Move 初始化、PPO 配置和日志目录 |
| `agents/bpx_stairs_hold_ppo_cfg.py` | Hold 站立来源、输入映射、探索强度和日志目录 |
| `mdp/stairs_skills.py` | 独立课程、Hold 重置与保持监视器 |
| `stairs_hold_env.py` | 冻结 Move 预运行、Hold 接管和实际动作掩码 |
| `stair_geometry.py`、`mdp/stairs.py` | 共享几何、地面查询、速度调度及奖励工具 |
| `bpx_stairs_base_env_cfg.py` | 两项任务共用的场景、奖励和观测基线 |

`scripts/rsl_rl/finetune.py` 负责权重迁移，`scripts/rsl_rl/hold_ppo.py` 负责 Hold 的 PPO 样本过滤。

## 地形与随机化

地形包含平地接近段、上楼、3.2 m 顶部平台和下楼。上/下行回合分别从首阶前的平地或下降边缘前的平台开始。默认地形库为 8 行难度 × 32 列，包含 8 列平地和 24 列楼梯。

| 参数 | 默认范围 |
| --- | --- |
| 台阶高度 | 第 0 行 3–5 cm；逐级扩展，第 7 行 5–15 cm |
| 踏面深度 | 每阶独立采样 25–40 cm |
| 连续阶数 | 每段 4–8 阶 |
| 楼梯有效宽度 | 1.2–2.0 m |
| 出生接近距离 | 首阶/下降边缘前 0.60–1.50 m |
| 楼梯前进速度 | 0.10–0.40 m/s，每 2–5 秒重采样 |
| 指令变化率 | vx 0.5 m/s²，vy 0.4 m/s²，wz 0.8 rad/s² |

两项任务共享摩擦 0.5–1.2、PD 增益 0.85–1.15 倍、0–4 个物理步的执行延迟，以及每回合固定的 ±0.02 rad 目标角误差。重置时机身和关节有小幅扰动。站立来源模型使用 ImplicitPD，微调后需要适应这里的延迟显式 PD。

## 启动训练

在项目根目录、已激活的 Isaac Lab 环境中分别启动。512 环境是起始示例，按可用显存调整。

```bash
conda activate isaaclab

# Move：默认从原 rough-history/model_4999.pt 迁移 Actor。
python scripts/rsl_rl/train.py \
  --task BPX-Stairs-Move-v0 --num_envs 512 --headless

# Hold：默认迁移 rough-stand/model_1998.pt，默认行走老师负责预运行。
python scripts/rsl_rl/train.py \
  --task BPX-Stairs-Hold-v0 --num_envs 512 --headless
```

Hold 默认也可用原 rough-history/model_4999.pt 预运行，因此能先练平地停车，不必等待 Move 完成全部课程。高阶 Hold 应换成具备相应通过能力的 Move checkpoint。冻结的 Move 必须符合当前 450→[128,128,128] ELU→12、无经验归一化接口，它不属于 Hold 优化器，也不写入 Hold 模型权重。

`--move_checkpoint` 配置行走老师；`--finetune_checkpoint` 初始化当前训练的策略，两者用途不同。Hold 默认初始化路径为 `logs/rsl_rl/bpx_rough_stand/2026-09-20_13-01-51/model_1998.pt`（该目录最后保存的模型，不代表已评估为最佳模型）。两个新任务微调时均迁移 Actor 并重建 Critic/优化器。`--resume --checkpoint <同任务模型>` 用于恢复已经训练的新任务；旧 rough-stand 不应直接 `--resume` 到 Hold。`--from_scratch` 可显式关闭默认迁移。

## 从旧站立策略初始化 Hold

源模型实际为 48 维单帧 Actor/Critic，Actor 包含机身线速度；隐藏层均为 `[128,128,128]`、ELU、12 维输出、无经验归一化。关节顺序、动作缩放 0.5、默认站姿、50 Hz 以及 PD 增益 40/1 与楼梯任务一致；原执行器为 ImplicitPD，楼梯执行器是带延迟的显式 PD，因此迁移后仍需适应动力学差异。

Hold 的 450 维盲走接口保留，通过显式 `stand48_to_blind450` 输入层映射迁移：

- 将角速度、重力投影、速度指令、关节位置/速度、上一动作的权重接到对应历史项的最新一帧。
- 其余九帧权重初始化为零，第一层偏置和后续 Actor 层完整复制；历史权重后续仍可学习。
- 缺失的三维机身线速度输入按零处理，不向 Actor 加入仿真真值。

因此，初始化的 Hold 对任意历史输出等价于“旧 Actor 读取最新一帧，并令机身线速度为零”。这保留静止附近的策略基础，但不保留旧策略基于非零线速度的反馈，制动与抗扰需要继续训练。48 维 Critic 不迁移到新 528 维价值网络。

初始探索标准差使用 0.03；源模型实际各关节约为 0.016–0.046，避免直接提高到通用楼梯配置的 0.08。学习率、优化器和课程按新 Hold 任务重新开始，运行名为 `stairs_hold_v2`。启动信息及 `params/initialization.yaml` 记录来源、输入映射、Critic 重建和探索标准差。映射仅在 Hold 配置显式启用，其他任务仍拒绝不兼容 Actor；已经是 450 维的 Hold 模型可直接迁移。

已对真实 `model_1998.pt` 做 CPU 迁移核验：128 组历史输入的上述输出等价性最大误差为 `1.19e-7`，后续 Actor 参数完整保留、Critic 保持新初始化、优化器状态为空。结果见 [迁移核验记录](outputs/stairs_hold_initialization/rough_stand_1998_transfer.json)。这验证权重迁移，不代表已通过楼梯停驻物理验收。

直接使用默认站立初始化和默认行走老师：

```bash
python scripts/rsl_rl/train.py \
  --task BPX-Stairs-Hold-v0 --num_envs 512 --headless
```

也可显式指定站立来源，或用 `--move_checkpoint` 单独指定行走老师：

```bash
python scripts/rsl_rl/train.py \
  --task BPX-Stairs-Hold-v0 --num_envs 512 --headless \
  --finetune_checkpoint logs/rsl_rl/bpx_rough_stand/2026-09-20_13-01-51/model_1998.pt
```

## Move 课程与验收

Move 的 `stop_probability=0`、`flat_standing_probability=0`，仅保留 0–0.5 秒的起步等待和速度指令斜坡。默认 25% 平地回放，剩余楼梯回合随机选择上行/下行；楼梯回合另有 25% 概率回放更低难度。

通过等级为 0–7，只由当前难度的楼梯样本更新：连续成功 3 次升级，连续失败 2 次降级。成功要求安全驶出、平均水平速度跟踪误差 <0.20 m/s、平均偏航角速度误差 <0.30 rad/s。成绩不依赖 Hold 的保持结果。

驶出时机身通过末阶之后的目标线，四足越过出口足端阈值，并连续 0.20 秒满足支撑、直立和水平速度 ≤0.65 m/s。出口仍给前进指令；楼梯回合最长 80 秒，平地回合 20 秒。

Move 保留水平/偏航速度跟踪、前进进度、物理驶出奖励，以及净空、踢面碰撞、滑动、落脚冲击和安全约束。主要任务奖励为速度跟踪 +2.0/+0.4、进度 +0.5、物理驶出 +2/事件；移除所有 `hold_*` 项及联合停驻完成奖励。

## Hold 课程

| 等级 | 分布与接管位置 |
| --- | --- |
| 0 | 平地默认站姿附近；无 Move 预运行，直接练保持 |
| 1 | 平地接近区，从冻结 Move 的真实运动状态停车 |
| 2 | 第 0 行低台阶，踏面中心停车 |
| 3 | 第 0 行低台阶，边缘停车 |
| 4–10 | 第 1–7 行楼梯，边缘停车 |

除 0 级外，机器人先出生在平坦接近区，由冻结 Move 走到停车点并按原斜坡减速，指令归零后切到 Hold。楼梯支撑姿态由真实行走形成，覆盖上/下行和首/中/末阶。观测历史、上一实际动作、接触及执行器延迟状态连续保留。20% 回放样本抽取更低 Hold 等级；只有当前等级的成绩参与升级。

Hold 使用固定 2–4 秒保持窗口，排除最初 0.30 秒；之后至少观测 0.5 秒，且至少 80% 时间同时满足：三维线速度 <0.08 m/s、三轴角速度模长 <0.20 rad/s、至少三足承重、相对本段零指令锚点漂移 <0.10 m、重力投影 z <−0.85。

计时只按实际 Hold 物理步推进，不等待站稳才开始。窗口结束即结束回合，未达标也结束；不要求重启或到达出口。同帧物理失败不能通过课程或领取成功奖励。奖励提供各分项的连续反馈，0.30 秒后才加强关节运动惩罚，允许收脚调整步；使用相对支撑高度，不强制默认关节姿态。

`stairs_hold_env.py` 选择实际执行的老师/学生动作。`scripts/rsl_rl/hold_ppo.py` 将老师预运行步从 Actor、Critic、熵、KL 和优势归一化中排除。学生采样动作及 log probability 保持一致，不被老师动作覆盖；纯预运行 rollout 跳过优化器更新。当前支持单设备、前馈 PPO，不支持 RND/对称增强/循环网络。

预运行失败单独统计，不降低 Hold 等级。高阶频繁预运行失败，说明老师尚不能产生所需状态，应先更新 Move checkpoint。

## 指标、恢复与回放

- Move：`Curriculum/terrain_mastery/level`、`success_rate`、`ascent_success`、`descent_success`。
- Hold：`Curriculum/hold_mastery/level`、`success_rate`、`sample_count`、`success_count`。
- 老师失败：`Curriculum/hold_mastery/warmup_failure_count`。
- 保持质量：`Metrics/base_velocity/hold_joint_pass_fraction` 及各分项比例。
- 有效训练样本：`Loss/hold_samples`、`Loss/hold_sample_fraction`。

成功率日志是 reset 批次均值，评估时应累计实际事件数；无样本批次的零值不表示能力下降。两种课程均未写入模型 checkpoint，恢复时需明确设置所需 `initial_level`。Hold checkpoint 也不包含冻结老师的权重，恢复高阶训练时需重新指定同一老师。

固定等级训练，例如只练平地停车：

```bash
python scripts/rsl_rl/train.py \
  --task BPX-Stairs-Hold-v0 --num_envs 512 --headless \
  --move_checkpoint "/绝对路径/到/Move/model_N.pt" \
  env.curriculum.hold_mastery.params.initial_level=1 \
  env.curriculum.hold_mastery.params.enabled=false \
  env.events.reset_base.params.replay_probability=0
```

回放使用通用脚本，Hold 老师通过环境参数指定：

```bash
python scripts/rsl_rl/play.py \
  --task BPX-Stairs-Hold-v0 --num_envs 4 --checkpoint "/绝对路径/到/Hold/model_N.pt" \
  env.move_checkpoint="/绝对路径/到/Move/model_N.pt" \
  env.curriculum.hold_mastery.params.initial_level=3 \
  env.curriculum.hold_mastery.params.enabled=false \
  env.events.reset_base.params.replay_probability=0
```

后续走→停→再走使用两个已训练策略做集成评估，无需第三项独立 PPO 训练。当前改动提供两个策略的训练和独立回放；部署 Supervisor 的路由需在选定模型后配置。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest scripts/tests -q

# 两个场景分别在新进程运行。
python scripts/validate_stairs_skills.py --task BPX-Stairs-Move-v0 --headless
python scripts/validate_stairs_skills.py --task BPX-Stairs-Hold-v0 --headless \
  --move_checkpoint "/绝对路径/到/Move/model_N.pt"

# 短轮启动检查，不代表已经收敛。
python scripts/rsl_rl/train.py \
  --task BPX-Stairs-Move-v0 --num_envs 32 --max_iterations 2 --headless --run_name smoke
python scripts/rsl_rl/train.py \
  --task BPX-Stairs-Hold-v0 --num_envs 32 --max_iterations 2 --headless --run_name smoke
```

CPU 回归覆盖课程独立性、固定验收窗口、重置计时及诊断顺序、冻结演员载入、动作路由、终止帧掩码、PPO 样本过滤、空 rollout、Critic 重建及 48→450 映射在零线速度下的输出等价性。物理稳定性和收敛仍需在可用的 Isaac Sim/CUDA 环境中验证。

## 历史实验

旧的单 Actor 混合楼梯任务已删除。当前训练只使用本文开头的 Move 与 Hold 两个独立任务。
