# BPX 四足机器人训练环境配置说明

本文按 2026-09-17 的源码说明公共接口及行走、站立、趴卧起身三个任务。项目以 Isaac Lab 2.2.1 为环境基线；具体检查点应以其运行目录中的 `params/env.yaml` 和 `params/agent.yaml` 为准。安装步骤见 [项目 README](README.md)，跨仿真操作见 [sim2sim 工作流](sim2sim/BPX_SIM2SIM_WORKFLOW.md)。

## 1. 配置分工与任务入口

| 文件 | 职责 |
| --- | --- |
| [bpx_base_env_cfg.py](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_base_env_cfg.py) | 机器人、平地场景、关节顺序、观测/动作、公共随机化和时序 |
| [bpx_locomotion_env_cfg.py](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_locomotion_env_cfg.py) | 行走指令、重置、奖励和终止 |
| [bpx_stand_env_cfg.py](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_stand_env_cfg.py) | 零指令站立与抗小扰动 |
| [bpx_init_env_cfg.py](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_init_env_cfg.py) | 趴卧起身、稳定保持和力矩惩罚课程 |
| [mdp/](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/mdp/) | 自定义奖励、分层指令与课程函数 |
| [agents/](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/agents/) | 公共 PPO 参数与各任务实验名、迭代数 |
| [任务注册](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/__init__.py) | 三个任务及旧名称兼容入口 |

| Gym ID | 回合时长 / 策略步数上限 | 默认迭代数 | 实验名 |
| --- | --- | --- | --- |
| `BPX-Locomotion-v0` | 20 秒 / 1000 | 1500 | `bpx_locomotion` |
| `BPX-Stand-v0` | 15 秒 / 750 | 1500 | `bpx_stand` |
| `BPX-Init-v0` | 4 秒 / 200 | 3000 | `bpx_init` |
| `BPX-Test-v0` | 20 秒 / 1000 | 1500 | `bpx_flat` |

`BPX-Test-v0` 使用当前行走环境，并保留原日志目录；它不会冻结或恢复旧版本奖励。Recovery 配置文件仅保留导入兼容名称，没有另注册 `BPX-Recovery-v0`。

## 2. 机器人与公共控制接口

### 2.1 资产与场景

USD 路径从包内位置计算，指向 `BPX_structure/usd/bpx.usd`。资产来自 [URDF](source/BPX_test/BPX_test/BPX_structure/bpx/urdf/bpx.urdf)，转换参数见 [usd/config.yaml](source/BPX_test/BPX_test/BPX_structure/usd/config.yaml)。

- `merge_fixed_joints=false`：保留足端独立刚体供接触检测使用。
- 转换时关节驱动增益为零；训练时由 Isaac Lab 执行器配置 PD。
- 平地场景启用全身接触传感器和足端腾空时间记录，无高度扫描观测。
- USD 文件被 Git 忽略，首次克隆需要准备完整的 `bpx.usd` 及引用资产。

### 2.2 默认站姿与执行器

| 参数 | 当前值 |
| --- | --- |
| 公共初始根位置 | `[0, 0, 0.40] m`；各任务重置事件可进一步修改 |
| 默认髋横滚 / 髋俯仰 / 膝角 | `0.0 / 0.7 / -1.4 rad` |
| 执行器 | 12 关节 `ImplicitActuatorCfg`，由 PhysX 处理隐式 PD |
| 名义刚度 / 阻尼 | `40 / 1` |
| 仿真力矩 / 速度限制 | `30 N·m / 20 rad/s` |
| 软关节位置限位系数 | `0.95` |
| 并行环境数 / 间距 | `2048 / 2.5 m` |
| 物理步长 / 动作降采样 | `0.005 s / 4`，即物理 200 Hz、策略 50 Hz |

PD 增益是当前工程参数，不应仅凭机器人质量推断其合理性；需结合关节响应、接触和力矩饱和验证。MuJoCo 默认显式 PD 与 PhysX 的隐式驱动语义不同。

### 2.3 唯一关节顺序与动作

策略按关节类型优先排列，每组腿序为 `fl → fr → hl → hr`：

```text
0..3   fl/fr/hl/hr_hip_roll_joint
4..7   fl/fr/hl/hr_hip_pitch_joint
8..11  fl/fr/hl/hr_knee_joint
```

动作和关节观测均使用显式名称列表及 `preserve_order=True`。动作映射为：

```text
q_target = q_default + 0.5 × action
```

当前 PPO 配置未显式开启动作裁剪，因此 **不能把动作范围理解为固定 [-1, 1]，也不能把目标角范围理解为默认角 ±0.5 rad**。默认角始终为站姿；Init 的趴姿只改变物理重置状态，不改变动作零点。

### 2.4 观测与 Init 非对称输入

平地行走和站立使用以下公共 48 维 `policy` 观测。Init 单独定义 `policy`（Actor）与 `critic` 组，均为单帧：Actor 删除最前面的三轴基座线速度，输入 45 维；Critic 保留 48 维，线速度使用无噪声真值，其余观测沿用表中的噪声。Actor 保留角速度和恒零的速度指令槽位。RSL-RL 包装器自动将 `critic` 组用于价值网络。

Init 观测变更后需要重新训练；原 48 维 Actor 检查点不能直接 resume 到新网络。导出的 Actor 为 45 维，sim2sim 可选 ONNX 专家加载器按模型输入维度适配，同时兼容旧 48 维模型。

| 切片 | 观测 | 训练噪声 |
| --- | --- | --- |
| `[0:3]` | 机身系基座线速度 | U(-0.1, 0.1) |
| `[3:6]` | 机身系基座角速度 | U(-0.2, 0.2) |
| `[6:9]` | 机身系重力投影 | U(-0.05, 0.05) |
| `[9:12]` | `[vx, vy, wz]` 指令 | 无 |
| `[12:24]` | 相对默认站姿的关节角 | U(-0.01, 0.01) |
| `[24:36]` | 相对默认关节速度；默认速度为零 | U(-1.5, 1.5) |
| `[36:48]` | 上一步动作 | 无 |

训练默认启用噪声。`play.py` 仍加载任务训练配置，不会自动关闭所有噪声和随机化；固定评估应显式配置。MuJoCo 运行器默认不添加训练噪声。

### 2.5 公共启动随机化

| 项目 | 范围 |
| --- | --- |
| 静摩擦 | `[0.5, 1.4]` |
| 动摩擦 | `[0.4, 1.0]` |
| 恢复系数 | `0` |
| 材质分桶数 | `64` |
| 刚度与阻尼 | 各自乘以 `[0.9, 1.1]` 的随机因子 |

这些事件在仿真启动时执行，不是每个回合重新采样材质和增益。

## 3. 行走任务

### 3.1 指令与重置

[StratifiedVelocityCommand](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/mdp/velocity_commands.py) 每 10 秒重采样，三种互斥模式为：

- 10% 静止：三个分量为零。
- 25% 纯旋转：`vx=vy=0`，`|wz| ∈ [0.2, 1.0] rad/s`，正负方向随机。
- 65% 混合运动：`vx ∈ [-1,1] m/s`、`vy ∈ [-0.5,0.5] m/s`、`wz ∈ [-1,1] rad/s`。

基座重置时，x/y 位置偏移为 ±0.5 m、偏航约 ±π，各线速度/角速度分量为 ±0.5。关节重置采用**加性偏移**：默认角 ±0.2 rad，速度 ±0.1 rad/s。每 10～15 秒通过设置水平速度施加 ±0.5 m/s 推扰。

### 3.2 奖励

| 奖励项 | 权重 | 含义 |
| --- | ---: | --- |
| `track_lin_vel_xy_exp` | 1.0 | 平面速度跟踪，核分母为 0.25 |
| `track_ang_vel_z_exp` | 0.5 | 偏航角速度跟踪，核分母为 0.25 |
| `feet_air_time` | 0.5 | 首次落足时累计腾空时间与 0.4 秒阈值之差 |
| `lin_vel_z_l2` | -2.0 | 垂向运动 |
| `ang_vel_xy_l2` | -0.05 | 横滚/俯仰角速度 |
| `flat_orientation_l2` | -0.5 | 躯干倾斜 |
| `dof_torques_l2` | -1e-5 | 关节力矩平方 |
| `dof_acc_l2` | -2.5e-7 | 关节加速度平方 |
| `action_rate_l2` | -0.01 | 相邻动作差 |
| `joint_deviation_l1` | -0.05 | 偏离默认姿态 |
| `undesired_contacts` | -1.0 | 躯干、髋、大腿或小腿接触力超过 1 N |

当前引用的 `feet_air_time` 仅在**平面指令模长 > 0.1 m/s** 时启用；纯旋转也不会获得该项。腾空时间短于阈值时该项可为负，不能仅凭负值认定训练失败。

奖励管理器会乘以策略步长 `dt=0.02 s`；TensorBoard 分项再除以配置回合时长。具体计算与解读见 [训练曲线说明](RL_TRAINING_CURVES.md)。

### 3.3 终止

20 秒超时、躯干接触力超过 1 N、倾角超过 1 rad。超时作为截断处理，没有另外配置一个“超时惩罚”奖励项。

## 4. 站立任务

站立任务所有 command 为零，目标是在约 0.40 m 高度保持水平、减少足端滑动和关节抖动。

- 重置：基座 x/y ±0.05 m、roll/pitch ±0.10 rad；关节角偏移 ±0.10 rad、关节速度 ±0.05 rad/s。
- 推扰：每 5～8 秒设置 x/y 速度 ±0.30 m/s、yaw 角速度 ±0.20 rad/s。
- 主要奖励：存活、零线速度/偏航跟踪；惩罚机身运动、倾斜、高度误差、足滑、关节速度与动作变化。
- 终止：15 秒超时、躯干接触力超过 1 N、倾角超过 0.8 rad。

这是独立训练目标；行走策略收到零指令时的行为不等同于已训练的 Stand 策略。

## 5. 趴卧起身任务与课程

### 5.1 初始状态与目标

Init 从腹部朝地、小腿贴地且足端朝前的趴姿开始。重置根高度约 0.132～0.134 m，髋俯仰约 0.98 rad、膝约 -2.62 rad，并有小幅位置、姿态和速度扰动。目标是起身到约 0.40 m 后继续保持稳定。

三个 command 槽位始终为零，不编码起身时间或参考轨迹。回合固定 4 秒；终止项只有超时和关节位置越界。允许起身过程身体接触地面，**满 200 步不等于成功起身**。

### 5.2 成功与稳定奖励

奖励包含抬升进度、接近默认站姿、足端接触、前足宽度、起身完成与完成后的低速度保持。`recovered` 当前同时要求：

- 根高度 ≥0.36 m；
- 倾角 ≤0.1 rad；
- 四足各自在接触传感器最近 3 帧窗口内出现过 >1 N 的接触力；
- 前足机身系横向间距至少 0.24 m。

训练成功条件与 MuJoCo Supervisor 切换条件不同：后者使用倾角 < 0.25 rad、速度阈值、四足接触及连续 0.50 秒确认，当前不检查前足宽度。两者应分别评估。

### 5.3 力矩惩罚课程

[modify_reward_weight_by_success_rate](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/mdp/curriculums.py) 调整 `joint_torque_limit_l2` 的权重；该奖励惩罚超过 28 N·m 的力矩部分，执行器硬限制仍为 30 N·m。

| 条件 | 权重 |
| --- | --- |
| 已记录回合不足 4096 个 | -0.1 |
| 最近窗口成功率 ≤70% | -0.1 |
| 成功率 70%～90% | 线性调整至 -1.0 |
| 成功率 ≥90% | -1.0 |

窗口最多保留最近 8192 个回合，成功按回合结束前最后一帧的 `recovered` 判定。成功率下降时，惩罚也会减弱；不是按训练迭代单向增强的课程。TensorBoard 记录 `Curriculum/joint_torque_limit/success_rate`、`weight` 和 `window_episodes`。

## 6. PPO 与运行

公共 PPO 配置：actor/critic 均为 `[128,128,128]` ELU，初始动作标准差 1.0，每环境采集 24 步，5 轮学习、4 个 minibatch，学习率 `1e-3` 自适应、目标 KL `0.01`、熵系数 `0.01`，未启用经验观测归一化。每 50 次迭代周期保存检查点；本机 RSL-RL 在正常完成训练后还会保存最终模型。

以下命令在项目根目录、Isaac Lab 环境中执行：

```bash
python scripts/list_envs.py
python scripts/zero_agent.py --task BPX-Stand-v0 --num_envs 4
python scripts/rsl_rl/train.py --task BPX-Locomotion-v0 --num_envs 64 --max_iterations 2 --headless

python scripts/rsl_rl/train.py --task BPX-Locomotion-v0 --headless
python scripts/rsl_rl/train.py --task BPX-Stand-v0 --headless
python scripts/rsl_rl/train.py --task BPX-Init-v0 --headless

tensorboard --logdir logs/rsl_rl
python scripts/rsl_rl/play.py --task BPX-Locomotion-v0 --num_envs 1 --real-time
```

短训练仅检查训练循环、日志和检查点保存，不能证明策略质量。原模板迁移时曾记录过“4 环境、50 步零动作”和“64 环境、2 次 PPO 迭代”的验证；这些属于早期版本记录，不代表当前三个任务均已重新验证。

## 7. 调整时的注意事项

- 显存不足时先减少 `--num_envs`，例如 512。
- 调整行走指令时同时检查分层采样比例及 `pure_rotation_speed_range`；偏航范围必须覆盖纯旋转范围。
- 调整动作裁剪、默认角或观测顺序会影响全部策略与部署端，不能只修改其中一端。
- 调整 Init 时同时观察终帧成功率、起身时间、前足宽度和力矩；不要用满回合长度代替成功率。
- 调整 `feet_air_time` 前先观察步频、滑移、触地冲击与速度误差，避免仅靠增大奖励权重调步态。
- 增加历史观测、地形扫描或真实执行器模型属于后续设计，需同步更新导出和推理接口。
