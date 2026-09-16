# BPX 从 Isaac Lab / Isaac Sim 到 MuJoCo 的 Sim2Sim 完整流程

本文档说明本工程的 sim2sim 是如何建立的、每一层为什么需要对齐、当前代码如何运行，以及出现异常时怎样定位问题。这里的 sim2sim 指：**在 Isaac Lab / PhysX 中训练策略，把同一个策略放入 MuJoCo，使用 MuJoCo 的机器人状态重新构造训练时的观测，并形成完整闭环控制。**

它不是简单地把 ONNX 文件“放进 MuJoCo 播放”，也不是复制一段 Isaac Sim 中记录好的动作。

## 1. 最终闭环是什么

整个运行链路可以概括为：

```text
MuJoCo 当前状态
    │
    ├─ 基座线速度、角速度、姿态
    ├─ 12 个关节位置和速度
    └─ 上一时刻策略动作
    │
    ▼
按 Isaac Lab 的定义构造 48 维观测 o_t
    │
    ▼
ONNX / TorchScript Actor：a_t = policy(o_t)
    │
    ▼
12 维动作转换为目标关节角
q_target = q_default + 0.5 × a_t
    │
    ▼
PD 控制器计算 12 个关节力矩
τ = Kp(q_target - q) - Kd·q̇
    │
    ▼
力矩限制到 ±30 Nm，并写入 MuJoCo 执行器
    │
    ▼
MuJoCo 前进一步，再从新状态重新构造观测
```

这是一个反馈闭环：机器人每走一步都会改变下一次观测，策略再根据新观测产生新动作。
其中速度指令既可以由 `--vx/--vy/--wz` 固定给出，也可以由 Linux 游戏手柄实时生成；
两种来源最终都写入同一观测切片 `observation[9:12]`，不会绕过策略直接控制关节。

## 2. 工程中的文件分工

| 文件 | 作用 |
| --- | --- |
| [`bpx_base_env_cfg.py`](../source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_base_env_cfg.py) | 三个策略共享的机器人、场景、关节顺序、观测、动作和仿真时序 |
| [`bpx_locomotion_env_cfg.py`](../source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_locomotion_env_cfg.py) | 行走任务的速度指令、重置、奖励和终止条件 |
| [`bpx_stand_env_cfg.py`](../source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_stand_env_cfg.py) | 零指令静止站立任务，重点抑制机身运动、足端滑动和关节抖动 |
| [`bpx_init_env_cfg.py`](../source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_init_env_cfg.py) | 腹部朝地趴卧、恢复四足稳定站姿的 Init 任务 |
| [`agents/rsl_rl_ppo_cfg.py`](../source/BPX_test/BPX_test/tasks/manager_based/bpx_test/agents/rsl_rl_ppo_cfg.py) | 三个任务共享的 PPO 和 Actor-Critic 默认配置 |
| [`bpx_test_env_cfg.py`](../source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_test_env_cfg.py) | 旧类名兼容层；新代码不应继续向这里增加任务配置 |
| [`bpx.xml`](../source/BPX_test/BPX_test/BPX_structure/mjcf/bpx.xml) | MuJoCo 使用的机器人和地面模型 |
| [`config/bpx_flat.toml`](config/bpx_flat.toml) | sim2sim 的集中配置和接口契约 |
| [`bpx_sim2sim/config.py`](bpx_sim2sim/config.py) | 读取 TOML、解析路径并检查维度和参数 |
| [`bpx_sim2sim/robot.py`](bpx_sim2sim/robot.py) | 按名称映射关节，读取状态，构造观测并执行 PD 控制 |
| [`bpx_sim2sim/policy.py`](bpx_sim2sim/policy.py) | ONNX、TorchScript 和零动作三种策略后端 |
| [`bpx_sim2sim/runner.py`](bpx_sim2sim/runner.py) | 以正确频率运行策略和物理仿真，显示 viewer 并记录 CSV |
| [`bpx_sim2sim/gamepad.py`](bpx_sim2sim/gamepad.py) | 发现 Linux 游戏手柄、读取 input-event、标定摇杆并生成速度指令 |
| [`bpx_sim2sim/supervisor.py`](bpx_sim2sim/supervisor.py) | INIT、STAND、WALK、STOPPING、DISABLED 行为状态机 |
| [`bpx_sim2sim/behaviors.py`](bpx_sim2sim/behaviors.py) | 将行为状态映射到 locomotion、stand 或 Init 策略 |
| [`bpx_sim2sim/transition.py`](bpx_sim2sim/transition.py) | command 变化率限制和策略 action 平滑混合 |
| [`run_mujoco.py`](run_mujoco.py) | 用户运行 MuJoCo 策略的命令行入口 |
| [`validate_setup.py`](validate_setup.py) | 检查 MJCF、关节映射、观测维度和零动作 PD |
| [`validate_policy_export.py`](validate_policy_export.py) | 比较 TorchScript 与 ONNX 输出 |
| [`probe_isaaclab.py`](probe_isaaclab.py) | 从 Isaac Lab 导出确定性接口和轨迹基准 |
| [`compare_trajectories.py`](compare_trajectories.py) | 按时间对齐 Isaac JSON 与 MuJoCo CSV 并计算 RMSE |
| [`JOINT_ORDER_MAPPING_GUIDE.md`](JOINT_ORDER_MAPPING_GUIDE.md) | 关节顺序的专项检查和防错说明 |

## 3. 第一步：冻结并理解训练侧接口

sim2sim 的第一原则是：**目标模拟器必须适配已训练策略的接口，而不是要求策略猜测目标模拟器的接口。**

### 3.0 多策略训练任务

训练环境已拆成三个独立 Gym 任务。三者共享 48 维观测、12 维动作、关节顺序、
默认关节角、动作缩放和 50 Hz 策略频率，但分别拥有自己的指令、重置分布、奖励和终止条件：

| Gym ID | 训练目标 | 默认日志目录 |
| --- | --- | --- |
| `BPX-Locomotion-v0` | 平地速度跟踪 | `logs/rsl_rl/bpx_locomotion/` |
| `BPX-Stand-v0` | 零指令静止站立和抗小扰动 | `logs/rsl_rl/bpx_stand/` |
| `BPX-Init-v0` | 从腹部朝地趴卧状态恢复四足稳定站姿 | `logs/rsl_rl/bpx_init/` |
| `BPX-Test-v0` | 旧行走任务兼容别名 | `logs/rsl_rl/bpx_flat/` |

```bash
python scripts/rsl_rl/train.py --task BPX-Locomotion-v0 --headless
python scripts/rsl_rl/train.py --task BPX-Stand-v0 --headless
python scripts/rsl_rl/train.py --task BPX-Init-v0 --headless
```

每个任务的 checkpoint 和 TensorBoard 数据自动进入不同目录，不会互相覆盖。也可以通过
`--experiment_name` 临时覆盖默认目录；该参数现在会真正写入 runner 配置。

训练完成后，分别回放目标 checkpoint；`play.py` 会在该 run 的 `exported/` 目录生成
`policy.pt` 和 `policy.onnx`：

```bash
python scripts/rsl_rl/play.py --task BPX-Locomotion-v0 --checkpoint model_1499.pt
python scripts/rsl_rl/play.py --task BPX-Stand-v0 --checkpoint model_1499.pt
python scripts/rsl_rl/play.py --task BPX-Init-v0 --checkpoint /path/to/model_2999.pt
```

随后把三种 ONNX 分别写入 `config/bpx_flat.toml` 的 `locomotion_policy`、
`stand_policy`、`init_policy`。启用 Supervisor 后会自动加载，不再需要每次在
命令行重复输入路径；命令行参数仍可用于临时覆盖。

当前 Init 已配置为：

```text
logs/rsl_rl/bpx_init/2026-09-14_23-03-13/exported/policy.onnx
```

手柄运行时，MuJoCo 先以 `0.11 m` 高度和训练使用的趴卧关节角复位，并进入
`WAITING_INIT`。RB 的按下沿触发 Init；达到高度、倾角、速度和四足接触条件并持续
`0.5 s` 后自动切换 Stand。此后摇杆 command 按原有迟滞规则进入 Walk。等待和起身期间
command 始终为零；使用 `--gamepad-deadman` 时，LB 仍仅控制行走 command。

```bash
python sim2sim/run_mujoco.py \
    --backend onnx --viewer --gamepad --gamepad-deadman \
    --supervisor --duration 120
```

站立和 Init 任务中的三维 command 都恒为零，三个导出模型仍保持相同的 48 维输入接口。
Init 不再跟踪固定时长的高度与竖直速度轨迹，而是用实际抬升进度、关节接近默认站姿程度
和足端接触比例提供连续奖励。前足机体系横向宽度必须至少达到 `0.24 m`，否则不能获得
`recovered` 与 `recovered_stability`；抬升后还会逐步启用髋横滚站姿奖励和足端滑动
惩罚。稳定项只在完成起身后抑制机身运动。动作变化率、关节速度和力矩惩罚保持较轻，
避免在尚未学会起身时压制探索。
MuJoCo Supervisor 使用相同的四足接触要求，并连续确认后才从 Init 切换到 stand。

旧的时序奖励 checkpoint 不能用于验证新的状态驱动起身逻辑；上面记录的当前导出模型
来自状态驱动版训练。

因此首先读取训练运行保存的配置：

```text
logs/rsl_rl/bpx_flat/2026-09-08_11-28-30/
├── model_1499.pt
├── exported/policy.pt
├── exported/policy.onnx
└── params/env.yaml
```

这里要确认六类信息：

1. 策略输入有哪些观测、拼接顺序和维度；
2. 策略输出控制什么、缩放是多少；
3. 关节排列和正方向；
4. 默认姿态；
5. 物理频率和策略频率；
6. 执行器刚度、阻尼和力矩限制。

不能只查看当前源码来推断一个旧模型的接口，因为源码可能在模型训练后被修改。旧模型应优先参考其运行目录中的 `params/env.yaml`、训练时日志以及配套元数据。

### 3.1 当前策略网络

当前 RSL-RL 配置为：

```text
输入：48 维观测
Actor 隐藏层：128 → 128 → 128
激活函数：ELU
输出：12 维动作
empirical_normalization：False
```

由于没有经验归一化器，MuJoCo 端直接按训练语义构造 `float32` 观测，不需要额外加载均值和方差。若以后开启 observation normalization，必须把归一化器参数一同导出并在推理前应用。

### 3.2 当前模型文件的关系

- `model_1499.pt` 是 RSL-RL 完整训练检查点，包含训练状态以及 Actor-Critic 参数；
- `policy.pt` 是便于 PyTorch 运行的 TorchScript Actor；
- `policy.onnx` 是便于跨运行时部署的 ONNX Actor。

已经检查过当前导出策略与 `model_1499.pt` 中 Actor 参数一致，并且 TorchScript 与 ONNX 对相同输入的最大动作误差约为 `2e-7`。因此当前问题定位重点不在模型导出，而在仿真接口和动力学。

## 4. 第二步：建立唯一的接口契约

本工程把目标侧参数集中写在 [`config/bpx_flat.toml`](config/bpx_flat.toml)。这样可以避免时序、动作缩放、默认角和关节顺序散落在多个脚本里。

当前契约如下：

| 参数 | 当前值 |
| --- | ---: |
| 物理步长 | `0.005 s` |
| 物理频率 | `200 Hz` |
| decimation | `4` |
| 策略周期 | `0.020 s` |
| 策略频率 | `50 Hz` |
| 观测维度 | `48` |
| 动作维度 | `12` |
| 动作缩放 | `0.5 rad` |
| 默认基座高度 | `0.4 m` |
| PD 刚度 `Kp` | `40` |
| PD 阻尼 `Kd` | `1` |
| 力矩限制 | `±30 Nm` |

`config.py` 会检查：

```text
观测维度 = 12 + 3 × 关节数 = 48
动作维度 = 关节数 = 12
```

路径也会相对于仓库根目录转换成绝对路径，从而避免因为当前终端目录不同而加载错误模型。

## 5. 第三步：准备和检查 MuJoCo 机器人模型

Isaac Lab 当前使用由 URDF 转换得到的 USD，MuJoCo 使用已有的 MJCF：

```text
Isaac：URDF → USD → PhysX
MuJoCo：bpx.xml → MuJoCo
```

两个模型至少需要检查：

- link 层级和关节类型；
- 关节名称、轴、限位和正方向；
- link 质量、质心和惯量；
- 碰撞几何及其局部位姿；
- 足端尺寸；
- 执行器与关节的一一对应关系；
- 基座初始高度和关节默认角；
- 地面摩擦和接触参数。

### 5.1 为什么按名称映射

MuJoCo 的 `qpos`、`qvel` 和 `ctrl` 不是同一种数组布局：自由关节在 `qpos` 中占 7 维，在 `qvel` 中占 6 维，执行器也有独立索引。因此不能假设：

```text
策略索引 i = qpos 索引 i = qvel 索引 i = actuator 索引 i
```

`BpxMujocoRobot` 会对 TOML 中每个关节名称分别查询：

```text
joint ID
joint qpos address
joint dof/qvel address
驱动该关节的唯一 actuator ID
```

只要名称正确，MJCF 内部采用单腿优先排列也不会影响策略排列。

### 5.2 当前策略的标准关节顺序

```text
 0  fl_hip_roll_joint
 1  fr_hip_roll_joint
 2  hl_hip_roll_joint
 3  hr_hip_roll_joint
 4  fl_hip_pitch_joint
 5  fr_hip_pitch_joint
 6  hl_hip_pitch_joint
 7  hr_hip_pitch_joint
 8  fl_knee_joint
 9  fr_knee_joint
10  hl_knee_joint
11  hr_knee_joint
```

这是“关节类型优先”顺序。最初的 MuJoCo 配置错误地采用了“一条腿的 roll、pitch、knee 排在一起”的顺序。两者都是 12 维，程序不会报维度错误，却会把动作和观测传给错误关节，导致机器人启动后立即乱动和倾倒。

现在训练配置已经使用显式 `BPX_POLICY_JOINT_NAMES`，动作、关节位置观测和关节速度观测都设置 `preserve_order=True`。当前已导出的旧模型虽然是在正则匹配配置下训练，但训练时实际解析顺序正好与上表一致。

## 6. 第四步：在 MuJoCo 中重建 48 维观测

当前 Actor 接收：

```text
o_t = [
    base_lin_vel_body(3),
    base_ang_vel_body(3),
    projected_gravity_body(3),
    command_vx_vy_wz(3),
    joint_pos_relative(12),
    joint_velocity(12),
    last_action(12),
]
```

对应索引为：

| 切片 | 维度 | 含义 |
| --- | ---: | --- |
| `[0:3]` | 3 | 机体坐标系基座线速度 |
| `[3:6]` | 3 | 机体坐标系基座角速度 |
| `[6:9]` | 3 | 机体坐标系投影重力 |
| `[9:12]` | 3 | `[vx, vy, wz]` 速度指令 |
| `[12:24]` | 12 | `q - q_default` |
| `[24:36]` | 12 | 关节速度 |
| `[36:48]` | 12 | 上一次策略动作 |

观测项的数值、单位、坐标系和顺序必须同时一致。仅仅凑成 48 维并不代表接口正确。

### 6.1 基座线速度

MuJoCo 自由关节给出的平移速度需要从世界坐标系旋转到机体坐标系。除此之外，本工程还补偿了自由关节原点和 torso 质心之间的偏移：

```text
v_origin_body = R_world_to_body × v_origin_world
v_com_body = v_origin_body + ω_body × r_com_body
```

这个补偿用于匹配 Isaac Lab 的根 link 质心线速度语义。如果直接读取自由关节原点速度，机器人转动时会产生系统性误差。

### 6.2 基座角速度

读取自由基座的角速度，并保持与 Isaac Lab 相同的机体坐标系表达和 `rad/s` 单位。

### 6.3 投影重力

策略不直接观察四元数，而是把世界坐标系的单位重力方向转换到机体坐标系：

```text
g_body = R_world_to_body × [0, 0, -1]
```

机器人水平时约为：

```text
[0, 0, -1]
```

它能够表达 roll 和 pitch 倾斜，但不能单独表达绝对 yaw。这正适合速度控制：策略需要知道身体是否倾斜，通常不需要知道机器人在世界中的绝对朝向。

### 6.4 速度指令

指令严格是：

```text
[vx, vy, wz]
```

- `vx`：机体 x 方向目标线速度，单位 `m/s`；
- `vy`：机体 y 方向目标线速度，单位 `m/s`；
- `wz`：绕 z 轴的目标偏航角速度，单位 `rad/s`。

第三项不是 z 方向线速度。当前策略也没有接收目标 `vz`。

运行器支持两种指令来源：

```text
固定指令：--vx / --vy / --wz
动态指令：command_source() → [vx, vy, wz]
```

动态来源会在每个 50 Hz 策略周期开始时采样，经过形状和有限值检查后写入观测，
同时写入 CSV 的 `command_vx`、`command_vy`、`command_wz` 列。手柄控制使用的就是这条
动态指令接口，所以手柄改变的是策略的目标速度，而不是直接向 MuJoCo 执行器写力矩。

### 6.5 关节状态和上一动作

关节位置不是绝对角，而是：

```text
joint_pos_relative = q - q_default
```

`last_action` 只有上一帧动作，不是多帧动作历史。首次运行时上一动作初始化为全零；每次策略推理后，新动作会成为下一次观测的 `last_action`。

### 6.6 为什么部署时不加训练噪声

训练配置对速度、投影重力和关节状态加入了随机噪声，以增强策略鲁棒性。MuJoCo 部署默认使用干净观测，因为噪声是训练手段，不是策略接口本身。

如果以后专门评估抗噪性能，可以再人为加入可控噪声；基础 sim2sim 对齐阶段不应把随机噪声混入误差来源。

## 7. 第五步：加载并验证策略

### 7.1 ONNX 后端

`OnnxPolicy` 使用 ONNX Runtime 的 CPU provider：

```text
float32 observation (48,)
→ reshape (1, 48)
→ ONNX Runtime
→ action (12,)
```

运行器检查输出必须是 12 维有限数值，否则立即报错。

### 7.2 TorchScript 后端

TorchScript 使用 PyTorch 加载 `policy.pt`，功能与 ONNX 后端相同。它主要用于：

- 在熟悉的 PyTorch 环境中查看输入输出；
- 与 ONNX 做相同观测下的逐元素对比；
- 判断问题来自模型导出还是仿真接口。

sim2sim 并不强制要求 ONNX；选择何种格式取决于目标运行时。本工程默认使用 ONNX，是因为独立 MuJoCo 环境只需安装 ONNX Runtime，无需安装完整 PyTorch。

### 7.3 零动作后端

`ZeroPolicy` 始终输出 12 个零，用于隔离策略：

```text
action = 0
q_target = q_default
```

如果零动作下模型和 PD 都不稳定，就不应先怀疑神经网络，而应检查初始姿态、重力、碰撞、关节方向或控制器。

## 8. 第六步：把策略动作转换为执行器输入

Isaac Lab 中使用：

```python
JointPositionActionCfg(
    scale=0.5,
    use_default_offset=True,
)
```

所以策略输出不是力矩，也不是绝对关节角，而是默认角附近的归一化位置偏移：

```text
q_target = q_default + 0.5 × action
```

例如某关节默认角是 `0.7 rad`，策略输出 `0.2`：

```text
q_target = 0.7 + 0.5 × 0.2 = 0.8 rad
```

MuJoCo 默认使用显式 PD：

```text
τ = 40(q_target - q) - 1·q̇
τ = clip(τ, -30, 30)
```

策略动作本身当前不裁剪到 `[-1, 1]`，最终执行力矩会限制到 `±30 Nm`。

配置还记录了 `velocity_limit=20 rad/s`，对应 Isaac 的 `velocity_limit_sim`。MuJoCo 没有与其完全等价的单一设置，当前运行器没有直接裁剪物理关节速度，这是尚未完全对齐的一项。

### 8.1 为什么选择显式 PD

Isaac Lab 使用 PhysX 隐式执行器语义。MuJoCo 可以配置内置位置伺服，但两种引擎的隐式积分和执行器实现并不完全相同。当前工程对两种方式做过比较，逐物理步显式重算 PD 在现有模型上效果更好，因此默认：

```toml
control.mode = "explicit"
simulation.integrator = "euler"
```

这并不意味着显式 PD 在所有机器人上都更好，只是当前 BPX 的已验证选择。

## 9. 第七步：对齐控制频率

训练环境为：

```text
physics dt = 0.005 s → 200 Hz
decimation = 4
policy dt = 0.005 × 4 = 0.020 s → 50 Hz
```

MuJoCo 运行器采用同样的双频率结构：

```text
每 20 ms：
    构造新观测
    运行一次 Actor
    更新动作和目标关节角

每 5 ms：
    使用当前目标角和最新关节状态重新计算 PD 力矩
    执行一次 mj_step
```

因此一个动作保持 4 个物理步，但 PD 力矩不会简单保持 4 步不变，而是随 `q` 和 `q̇` 每步更新。

如果错误地以 200 Hz 调用策略、以 50 Hz 更新物理，或者只在策略帧计算一次 PD 力矩，闭环动态都会与训练环境明显不同。

## 10. 第八步：日志、终止和 viewer

### 10.1 CSV 日志

运行器以策略频率记录：

- 时间和基座位置；
- 机体线速度、角速度和投影重力；
- 倾角和速度指令；
- 每个关节的位置、速度、目标角、动作和力矩。

日志列名带有关节名称，便于检查左右腿或关节类型是否发生交换。

### 10.2 倾倒是否终止

默认情况下倾倒不会结束 sim2sim，仿真会持续到指定时长，方便观察完整故障过程。需要提前结束时添加：

```bash
--terminate-on-fall
```

此时可能的结束原因包括：

```text
base_height       基座高度过低
bad_orientation   倾角超过阈值
torso_contact     躯干接触地面
```

### 10.3 viewer 段错误的处理

之前机器人倾倒退出后出现过“段错误（核心已转储）”。这发生在 MuJoCo 被动 viewer 的后台渲染线程仍在清理 OpenGL 资源时 Python 解释器已经退出，并不等于策略或动力学计算发生段错误。

当前 `_SafePassiveViewerContext` 会先关闭 viewer，再等待其后台线程结束，从而解决 MuJoCo 3.13 下的退出竞争。

### 10.4 游戏手柄实时指令

手柄输入参考 `/home/hanguang/DRV/usb_hid_receiver.py`，使用 Linux 内核标准
`/dev/input/event*` 接口。读取是非阻塞的，不会让 200 Hz 物理循环等待输入，也不依赖
`hidapi` 或 `python-evdev`。启动时会通过 `EVIOCGABS` 查询每根轴的当前值、最小值、
最大值和硬件死区，因此同时支持常见的有符号范围（如 `-32768..32767`）和无符号范围
（如 `0..255`）。软件死区外的数值会重新缩放到完整的 `[-1, 1]`。

默认映射与训练指令范围一致：

| 手柄输入 | Linux 轴 | command | 满量程 |
| --- | ---: | ---: | ---: |
| 左摇杆上下 | `ABS_Y=1` | `vx`，上推为正 | `±1.0 m/s` |
| 左摇杆左右 | `ABS_X=0` | `vy`，左推为正 | `±0.5 m/s` |
| 右摇杆左右 | `ABS_RX=3` | `wz`，左推为正 | `±1.0 rad/s` |

列出手柄：

```bash
python sim2sim/run_mujoco.py --list-gamepads
```

启动实时控制：

```bash
python sim2sim/run_mujoco.py \
    --backend onnx \
    --viewer \
    --gamepad \
    --gamepad-deadman \
    --duration 120
```

`--gamepad-deadman` 表示只有按住 LB（Linux 按钮码 `310`）时才接受非零指令；松开后
下一策略周期立即发送零速度指令。有多个设备时先通过 `--list-gamepads` 查看序号，再添加
`--gamepad-index 0`；也可以用 `--gamepad-device /dev/input/eventN` 显式指定节点。

如果手柄不是标准轴布局，可先观察原始事件：

```bash
python /home/hanguang/DRV/usb_hid_receiver.py \
    --backend input --device /dev/input/eventN
```

然后通过以下参数调整：

```text
--gamepad-vx-axis / --gamepad-vy-axis / --gamepad-wz-axis
--gamepad-vx-sign / --gamepad-vy-sign / --gamepad-wz-sign
--gamepad-max-vx / --gamepad-max-vy / --gamepad-max-wz
--gamepad-deadzone
```

交互控制必须保持实时限速，不要同时传入 `--no-realtime`。如果节点存在但无法打开，检查
`ls -l /dev/input/eventN`，并为当前用户配置该节点所属 `input` 组或对应 udev 规则的读取权限。
拔出手柄或收到非法轴配置时，运行器会关闭 viewer 并报告明确错误，而不是继续使用陈旧指令。

### 10.5 上层行为状态机

传入 `--supervisor` 后，运行链路变为：

```text
原始 command
    ↓
BehaviorSupervisor：迟滞、驻留时间、跌倒判断
    ↓
INIT / STAND / WALK / STOPPING / DISABLED
    ↓
BehaviorPolicies：选择对应底层策略
    ↓
ActionBlender：从上一帧实际 action 平滑切换
    ↓
PD 与 MuJoCo
```

默认转换为：

```text
STAND   -- command > 0.12，持续 0.15 s --> WALK
WALK    -- command < 0.05，持续 0.30 s --> STOPPING
STOPPING -- 速度稳定，持续 0.20 s ------> STAND
STOPPING -- 1.50 s 仍未稳定 ------------> STAND（超时保护）
任意正常状态 -- 高度 < 0.20 m 或倾角 > 0.70 rad --> INIT / DISABLED
INIT -- 高度和姿态恢复，持续 0.50 s --------------> STAND
```

command 阈值先分别除以训练满量程 `[1.0, 0.5, 1.0]`，再取三个分量绝对值的最大值，
所以阈值对 `vx`、`vy`、`wz` 使用一致的归一化语义。进入和退出阈值不同，可以避免
手柄在边界附近使状态反复跳变；`command_slew_rate` 限制速度指令变化，`ActionBlender`
使用 smoothstep 在默认 `0.30 s` 内混合旧 action 和新策略 action。

三种行为策略路径统一写在 `config/bpx_flat.toml` 的 `[paths]`。Supervisor 启用时自动
检查并加载；可选文件不存在只打印 warning。没有 stand 时 STAND 复用零 command 的
locomotion，没有 Init 时未站立状态进入 DISABLED。专用策略必须遵守当前 48 维观测、
12 维动作、关节顺序、默认角和 `action_scale=0.5` 的共同接口契约。
由于当前 Init 只学习腹部朝地的水平趴卧姿态，Supervisor 还要求起始倾角小于
`init_start_max_tilt=0.35 rad`；侧翻和仰翻不会送入这个超出训练分布的策略。

```bash
python sim2sim/run_mujoco.py \
    --backend onnx --viewer --gamepad --gamepad-deadman \
    --supervisor \
    --duration 120 \
    --log sim2sim/logs/supervisor.csv
```

不加 `--supervisor` 且不传 `--stand-policy`/`--init-policy` 时，不会加载可选专家，
仍运行原来的单 locomotion 策略。两个命令行路径参数保留为 TOML 的临时覆盖入口。

状态机参数集中在 `config/bpx_flat.toml` 的 `[supervisor]`。CSV 新增
`behavior_mode`、`behavior_policy`、`transition_alpha`、`raw_command_*`；原来的
`command_*` 表示经过 Supervisor 过滤后真正进入策略观测的指令。

## 11. 分层验证流程

不要一开始只看“机器人能不能走”。正确做法是逐层缩小问题范围。

### 11.1 检查独立环境

```bash
cd /home/hanguang/BPX_test
conda activate bpx-sim2sim
which python
python -c "import mujoco, onnxruntime; print(mujoco.__version__, onnxruntime.__version__)"
```

当前独立环境使用 Python 3.11、MuJoCo 3.13 和 ONNX Runtime 1.30。

### 11.2 检查 MJCF、映射和零动作 PD

```bash
python sim2sim/validate_setup.py
```

这一步验证模型能加载、12 个关节和执行器都能按名称找到、观测是 48 维、初始关节角正确且零动作不会出现 NaN/Inf。

注意：脚本通过只代表 MuJoCo 内部自洽；打印出的 `joint order` 还必须与 Isaac Lab 的 `Resolved joint names` 逐项比较。

### 11.3 检查两种导出格式

如果当前环境同时安装了 PyTorch：

```bash
python sim2sim/validate_policy_export.py --samples 100
```

如果 TorchScript 与 ONNX 对相同输入不一致，应先处理导出或运行时问题，不进入动力学对齐。

### 11.4 运行零策略

```bash
python sim2sim/run_mujoco.py \
    --backend zero \
    --duration 2 \
    --no-realtime \
    --log sim2sim/logs/mujoco_zero_action.csv
```

它回答的是：“不依赖神经网络，MJCF 的初始姿态和 PD 控制是否合理？”

### 11.5 查看实际策略

无窗口快速运行：

```bash
python sim2sim/run_mujoco.py \
    --backend onnx \
    --vx 0.2 --vy 0 --wz 0 \
    --duration 20 \
    --no-realtime \
    --log sim2sim/logs/forward_02.csv
```

打开画面：

```bash
python sim2sim/run_mujoco.py --backend onnx --viewer --vx 0.2
```

### 11.6 从 Isaac Lab 生成确定性基准

先进入 Isaac Lab 环境：

```bash
conda activate isaaclab
python sim2sim/probe_isaaclab.py \
    --headless --device cpu --skip-policy --steps 100 \
    --vx 0 --vy 0 --wz 0 \
    --json-output sim2sim/logs/isaac_zero_action.json
```

探测脚本会：

- 关闭观测噪声；
- 关闭材质、执行器增益和随机推力事件；
- 保留 reset 事件，但把重置随机范围收缩为零；
- 直接从环境返回的 policy 观测读取状态；
- 使用零动作生成确定性轨迹。

必须保留 reset 事件。之前曾直接删除全部 reset 事件，导致关节没有被写回默认姿态，使轨迹比较从错误初始状态开始。

当前机器存在 Warp/CUDA 驱动接口警告，所以探测脚本使用 CPU，并避免额外访问容易创建 PhysX tensor view 的懒加载属性。

### 11.7 对拍 Isaac 与 MuJoCo

生成 2 秒 MuJoCo 零动作日志后运行：

```bash
python sim2sim/compare_trajectories.py \
    --isaac sim2sim/logs/isaac_zero_action.json \
    --mujoco sim2sim/logs/mujoco_zero_action.csv
```

脚本按仿真时间寻找最近样本，并比较：

- 基座机体线速度；
- 基座机体角速度；
- 投影重力；
- 关节位置；
- 关节速度；
- 动作。

MuJoCo CSV 的首行是 `t=0`，Isaac 轨迹首行是第一个策略步之后的 `t=0.02 s`，因此不能简单按文件行号直接相减。

### 11.8 one-hot 关节测试

每次只令 `action[i]` 非零，检查训练端和部署端是否都控制标准列表中的第 `i` 个关节。当前缩放为 0.5，因此：

```text
action[i] = 0.2
→ q_target[i] - q_default[i] = 0.1 rad
```

优先比较控制器内部目标角，不只看画面，因为动力学耦合会让其他关节也产生被动运动。

## 12. 如何根据现象判断问题层级

| 现象 | 优先检查 |
| --- | --- |
| 启动后立即剧烈乱动 | 关节顺序、动作语义、关节正方向、观测切片 |
| 缓慢下沉或逐渐倾倒 | 默认姿态、Kp/Kd、重力、质量惯量、接触参数 |
| 能站立但不跟踪速度 | 指令是否写入 `[9:12]`、坐标系、是否加载正确模型 |
| 步态相位类似但幅度明显不对 | 动作缩放、PD 参数、力矩饱和、控制频率 |
| 只在落足时快速发散 | 足端碰撞、摩擦、恢复系数、求解器参数 |
| TorchScript 正常而 ONNX 异常 | 导出模型、输入名称、dtype、运行时版本 |
| 仿真正常结束后才段错误 | viewer / GLFW / OpenGL 线程清理，不是策略动作 |
| 12 维都存在但左右腿动作互换 | 关节顺序映射 |

调试顺序建议始终是：

```text
模型文件正确性
→ 维度
→ 关节顺序和方向
→ 观测语义与坐标系
→ 动作缩放
→ 控制频率与 PD
→ 质量惯量
→ 碰撞、摩擦和求解器
```

先检查离散接口，再调整连续动力学。否则可能通过修改摩擦或 Kp 暂时掩盖一个关节映射错误。

## 13. 本次实际定位和修复过程

最初的现象是：

```text
Isaac Sim 中可以行走
MuJoCo 中启动后迅速倾倒、动作混乱
```

排查过程是：

1. 确认实际加载了 `policy.onnx`；
2. 确认 ONNX 是最终训练检查点 Actor 的导出；
3. 确认 TorchScript 和 ONNX 输出一致；
4. 核对 48 维观测、默认姿态、动作缩放和控制频率；
5. 用 Isaac 启动日志检查 `JointPositionAction` 的真实关节顺序；
6. 发现 Isaac 是“4 个 roll → 4 个 pitch → 4 个 knee”，MuJoCo 却是单腿优先；
7. 修正 TOML 关节列表及默认角数组；
8. 按名称映射 MuJoCo 的 qpos、qvel 和 actuator；
9. 重新运行后策略稳定行走；
10. 将训练配置改为显式关节列表和 `preserve_order=True`，防止以后再次依赖隐含顺序。

这个案例说明：神经网络输入输出维度正确、模型也能成功推理，仍然可能因为每一维的“语义标签”不同而完全失效。

## 14. 当前验证结果

### 14.1 前进策略

修复关节顺序后，名义参数下执行：

```text
command = [0.2, 0.0, 0.0]
duration = 20 s
```

结果为：

```text
20 秒稳定运行
2～20 秒平均机体前向速度约 0.187 m/s
累计前进约 3.72 m
```

这说明当前策略在 MuJoCo 中不只是保持站立，而是确实响应了前进速度指令。

### 14.2 确定性零动作对拍

当前保存的 100 个对齐样本结果为：

| 状态组 | 总体 RMSE | 最大单帧 RMSE |
| --- | ---: | ---: |
| 基座机体线速度 | `0.0494` | `0.0914` |
| 基座机体角速度 | `0.0844` | `0.1873` |
| 投影重力 | `0.0217` | `0.0476` |
| 关节位置 | `0.0459 rad` | `0.0608 rad` |
| 关节速度 | `0.2241 rad/s` | `0.5886 rad/s` |
| 动作 | `0` | `0` |

这些误差不为零是正常的：PhysX 和 MuJoCo 使用不同的接触、约束、积分器和执行器实现。sim2sim 的目标首先是接口一致、行为稳定和统计特性接近，而不是要求长时间轨迹逐浮点数相同。

### 14.3 手柄 command 接入验证

手柄接入完成后的自动测试覆盖：

- 有符号和无符号摇杆范围归一化；
- 软件死区及驱动报告的硬件 `flat` 死区；
- 默认轴方向和三项速度限幅；
- LB deadman 松开时输出全零；
- 动态 command 在每个策略周期进入 48 维观测的 `[9:12]`；
- 原有固定 command、零策略和 ONNX 策略运行不受影响。

当前自动测试共 25 项并全部通过，ONNX 策略也完成了短时闭环回归。验证环境中当时没有
映射可用的 `/dev/input` 手柄节点，因此 USB 手柄的最终硬件轴确认需要在连接设备后执行；
这不会影响已完成的事件解码、归一化和策略接口测试。

### 14.4 上层状态机验证

状态机测试覆盖完整的 `STAND → WALK → STOPPING → STAND` 循环、进入/退出迟滞、
command 变化率限制、action 连续混合、无 Init 策略时安全停用，以及存在 Init 策略时进入
INIT。真实 ONNX 在 Supervisor 模式下完成 2 秒回归：先以 STAND 和零 command
启动，满足进入条件后切换到 WALK，并稳定接收 `vx=0.2 m/s`。

## 15. 当前仍未完全对齐的部分

### 15.1 碰撞几何

目前发现 URDF 与 MJCF 的部分碰撞盒存在尺寸或数量差异，例如：

- 大腿 box 在 URDF 中完整尺寸为 `0.045 × 0.13 × 0.029 m`，MJCF 对应完整尺寸约为 `0.045 × 0.16 × 0.029 m`；
- 小腿部分 box 的长度和局部位置不同；
- MJCF 小腿包含一个额外碰撞几何。

这些差异会改变落足、腿部擦碰和自碰撞，应作为后续精细动力学对齐项目，而不是在接口错误尚未排除时先调整。

### 15.2 摩擦和接触

Isaac 训练中机器人材质会随机化：

```text
静摩擦：0.6～1.2
动摩擦：0.4～1.0
恢复系数：0
```

MuJoCo 当前没有复现每个 episode 的这组随机化，地面和机器人实际采用 MJCF / MuJoCo 中的接触参数。因此当前只验证名义条件，不能据此认为完整域随机范围已经对齐。

### 15.3 执行器语义

当前显式 PD 匹配了名义 `Kp`、`Kd` 和力矩限制，但没有精确复现：

- PhysX 隐式驱动器的离散积分语义；
- 电机延迟和通信延迟；
- 实际电机转矩-速度曲线；
- 摩擦、齿隙和死区；
- `velocity_limit_sim` 的完全等价行为。

### 15.4 状态估计

sim2sim 可以直接从 MuJoCo 读取真实状态。真正 sim2real 时，基座线速度、姿态和角速度需要来自 IMU、编码器和状态估计器。尤其基座线速度通常不是单个传感器直接给出的，需要估计器或策略结构调整。

因此“策略能在 MuJoCo 中运行”是 sim2real 的重要中间验证，但不是实机部署完成。

## 16. 后续改进顺序

建议按以下顺序继续：

1. 随策略导出 `policy_metadata.json`，记录完整关节顺序、观测布局、默认角、动作缩放、dt 和 decimation；
2. 让 MuJoCo 启动时自动比较元数据，发现不一致立即退出；
3. 增加自动 one-hot 动作和观测映射测试；
4. 对零动作和相同首帧策略动作做更细的短时响应比较；
5. 对齐 URDF、USD、MJCF 的碰撞几何；
6. 对 Kp、Kd、armature、joint damping、摩擦和接触参数做单变量扫描；
7. 逐步测试 `vx`、`vy`、`wz` 的正负指令，而不只测试 `vx=0.2`；
8. 加入观测延迟、动作延迟、噪声和执行器偏差；
9. 建立固定回归测试，防止机器人模型或 Isaac Lab 升级后接口悄悄变化；
10. 再进入实机状态估计、通信频率、安全限制和急停设计。

推荐的速度指令测试矩阵为：

```text
站立：[0.0,  0.0,  0.0]
前进：[0.2,  0.0,  0.0]
后退：[-0.2, 0.0,  0.0]
左移：[0.0,  0.2,  0.0]
右移：[0.0, -0.2,  0.0]
左转：[0.0,  0.0,  0.5]
右转：[0.0,  0.0, -0.5]
组合：[0.2,  0.1,  0.3]
```

每项都应记录稳定时间、平均跟踪误差、最大倾角、最大关节速度、最大力矩和是否发生非足端接触。

## 17. 最简运行步骤

如果只想重新运行当前已验证的策略：

```bash
cd /home/hanguang/BPX_test
conda activate bpx-sim2sim

python sim2sim/validate_setup.py

python sim2sim/run_mujoco.py \
    --backend onnx \
    --viewer \
    --vx 0.2 --vy 0 --wz 0
```

如果出现问题，先无窗口保存日志：

```bash
python sim2sim/run_mujoco.py \
    --backend onnx \
    --vx 0.2 --vy 0 --wz 0 \
    --duration 10 \
    --no-realtime \
    --log sim2sim/logs/debug.csv
```

如果使用手柄实时控制：

```bash
python sim2sim/run_mujoco.py --list-gamepads

python sim2sim/run_mujoco.py \
    --backend onnx \
    --viewer \
    --gamepad \
    --gamepad-deadman \
    --supervisor \
    --duration 120
```

运行时终端会每秒显示当前 `command=[vx, vy, wz]`，可以先小幅推动摇杆确认方向，再逐步
增大幅度。为了降低首次测试风险，也可以添加
`--gamepad-max-vx 0.4 --gamepad-max-vy 0.2 --gamepad-max-wz 0.5`。

## 18. 理解 sim2sim 的核心结论

完成 sim2sim 需要同时处理三层一致性：

1. **模型一致性**：加载的是正确检查点导出的同一个 Actor；
2. **接口一致性**：观测、动作、关节顺序、坐标系、单位和频率完全相同；
3. **动力学一致性**：质量、惯量、碰撞、摩擦、执行器和求解器足够接近。

本次最严重的问题属于第二层：关节顺序错误。它修复后机器人立即从“启动倾倒”变为“稳定前进”。目前剩余问题主要属于第三层，即 PhysX 与 MuJoCo 的精细动力学差异。

所以判断 sim2sim 是否成功，不应只问“ONNX 是否加载”，而应问：

```text
同一个物理状态是否产生了同一种观测？
同一个动作索引是否控制了同一个物理关节？
相同控制频率下，两个模拟器是否产生了相近且稳定的闭环行为？
```

这三项成立，才真正完成了从 Isaac Lab 到 MuJoCo 的策略迁移。
