# BPX Isaac Lab → MuJoCo sim2sim

本目录将 Isaac Lab/RSL-RL 训练得到的 BPX 速度策略部署到 MuJoCo。现有训练环境和机器人资产不会被修改。

## 已对齐的接口

- 物理步长：`0.005 s`（200 Hz）
- 策略降采样：4（50 Hz）
- 动作：12维关节位置偏移，`q_target = q_default + 0.5 * action`
- 执行器：逐物理步 PD，`Kp=40`、`Kd=1`、总力矩限制 `±30 Nm`
- 关节顺序：四个髋横滚、四个髋俯仰、四个膝关节，与 Isaac Articulation 的实际顺序一致
- 观测：48维，顺序严格为：

```text
base_lin_vel_body(3)
base_ang_vel_body(3)
projected_gravity_body(3)
command_vx_vy_wz(3)
joint_pos_relative(12)
joint_velocity(12)
last_action(12)
```

其中 `base_lin_vel_body` 与 Isaac Lab 一样取根 link 的质心速度；运行器会补偿 MuJoCo 自由关节原点与躯干质心之间的偏移。

训练时观测包含随机噪声。部署推理使用干净观测，不在 MuJoCo 运行器中继续添加训练噪声或域随机化。

## 环境准备

建议使用独立 Conda 环境，避免修改已有的 Isaac Lab 环境：

```bash
cd /home/hanguang/BPX_test
conda create -n bpx-sim2sim python=3.11 -y
conda activate bpx-sim2sim
python -m pip install --upgrade pip
python -m pip install -r sim2sim/requirements.txt
```

检查实际使用的解释器和依赖：

```bash
which python
python -c "import mujoco, onnxruntime; print(mujoco.__version__, onnxruntime.__version__)"
```

当前创建的 `bpx-sim2sim` 环境使用 Python 3.11、MuJoCo 3.13 和 ONNX Runtime 1.30，可以直接运行 ONNX 后端。TorchScript后端还要求同一个环境安装 `torch`；只运行 ONNX 时无需安装 PyTorch。

## 1. 验证模型与控制器

```bash
python sim2sim/validate_setup.py
```

该命令检查：

- MJCF可以加载；
- 自由基座、12个关节和12个执行器能够按名称匹配；
- 观测维度为48；
- 初始关节角正确；
- 零动作 PD 仿真不会产生 NaN/Inf。

## 2. 无窗口运行策略

ONNX后端：

```bash
python sim2sim/run_mujoco.py \
    --backend onnx \
    --vx 0.2 --vy 0.0 --wz 0.0 \
    --duration 10 \
    --no-realtime \
    --log sim2sim/logs/forward_02.csv
```

TorchScript后端：

```bash
python sim2sim/run_mujoco.py --backend torchscript --device cpu --duration 10
```

若同一 Python 环境同时安装了 PyTorch 与 ONNX Runtime，可以验证两个导出文件是否一致：

```bash
python sim2sim/validate_policy_export.py --samples 100
```

只验证模型和 PD，不加载训练策略：

```bash
python sim2sim/run_mujoco.py --backend zero --duration 2 --no-realtime
```

## 3. 打开可视化

```bash
python sim2sim/run_mujoco.py --backend onnx --viewer --vx 0.2
```

速度指令始终采用 `[vx, vy, wz]`，其中前两项为机体平面线速度，第三项为偏航角速度。

### 使用手柄实时发送 command

手柄读取复用了 `/home/hanguang/DRV/usb_hid_receiver.py` 所采用的 Linux
`input-event` 接口，不需要额外安装 `hidapi` 或 `python-evdev`。先连接手柄并列出设备：

```bash
python sim2sim/run_mujoco.py --list-gamepads
```

若只检测到一个手柄，直接启动：

```bash
python sim2sim/run_mujoco.py \
    --backend onnx --viewer --gamepad --duration 120
```

默认映射与训练速度范围如下：

| 手柄输入 | 机器人 command | 满量程 |
|---|---:|---:|
| 左摇杆上下 | `vx`，上推为前进 | `±1.0 m/s` |
| 左摇杆左右 | `vy`，左推为正 | `±0.5 m/s` |
| 右摇杆左右 | `wz`，左推为逆时针 | `±1.0 rad/s` |

启动后终端每秒打印当前 `command=[vx, vy, wz]`。摇杆默认死区为 `0.1`；需要按住
LB 才允许非零指令时，加 `--gamepad-deadman`。有多个手柄时，按列表序号选择：

```bash
python sim2sim/run_mujoco.py --backend onnx --viewer --gamepad \
    --gamepad-index 0 --gamepad-deadman --duration 120
```

也可以直接指定设备节点，例如 `--gamepad-device /dev/input/event25`。如果手柄的
轴布局不同，先用接收脚本观察轴码：

```bash
python /home/hanguang/DRV/usb_hid_receiver.py \
    --backend input --device /dev/input/event25
```

然后用 `--gamepad-vx-axis`、`--gamepad-vy-axis`、`--gamepad-wz-axis` 指定轴码；
若某个方向相反，将对应的 `--gamepad-*-sign` 从默认 `-1` 改为 `1`。可用
`--gamepad-max-vx`、`--gamepad-max-vy`、`--gamepad-max-wz` 限制最大速度，例如：

```bash
python sim2sim/run_mujoco.py --backend onnx --viewer --gamepad \
    --gamepad-max-vx 0.4 --gamepad-max-vy 0.2 --gamepad-max-wz 0.5 \
    --duration 120
```

交互控制需要实时仿真，因此不要加 `--no-realtime`。若打开设备时报权限错误，请检查
`ls -l /dev/input/eventN`，并让当前用户获得该节点所属 `input` 组或相应 udev 规则的读取权限。

### 启用上层行为状态机

`--supervisor` 会在策略上方启用以下状态机：

```text
趴姿复位 → WAITING_INIT ──RB 按下──→ INIT ──稳定确认──→ STAND
                                                       │
                                                       └→ WALK → STOPPING → STAND
```

基础用法：

```bash
python sim2sim/run_mujoco.py \
    --backend onnx --viewer --gamepad --gamepad-deadman \
    --supervisor --duration 120 \
    --log sim2sim/logs/supervisor.csv
```

状态机使用不同的进入/退出阈值和持续时间防止零点抖动；进入 `STOPPING` 后平滑降低
command，切换底层策略时在 `0.3 s` 内混合 action。所有阈值位于
[`config/bpx_flat.toml`](config/bpx_flat.toml) 的 `[supervisor]`。

默认复位高度为 `0.11 m`，关节为腹部朝地趴姿。程序启动后用固定关节 action 保持趴姿，
此时以及 `INIT` 期间所有摇杆 command 都会清零。按下手柄 RB（Linux `BTN_TR=311`）的
按下沿后只触发一次 Init；无需持续按住。若同时使用 `--gamepad-deadman`，LB 仍只负责
行走 command 的使能，与 RB 的起身触发互不冲突。配置了 Init policy 时，使用
`--gamepad` 会自动启用 Supervisor，也可以继续显式写出 `--supervisor`。

三种行为策略的 ONNX 路径统一记录在 `config/bpx_flat.toml` 的 `[paths]`：
`locomotion_policy`、`stand_policy`、`init_policy`。启用 Supervisor 时会自动加载这些路径；可选文件不存在只打印
`[WARNING]`，不会阻止现有策略运行。未加载到站立策略时，`STAND` 回退为零 command
的 locomotion policy；未加载到 Init 策略时，未站立状态进入 `DISABLED`。
Init 当前只训练腹部朝地且机身接近水平的趴卧姿态；进入 Init 后，策略观测的原
command 三维槽位保持为零。起身过程完全由本体状态反馈驱动，不再接收完成时刻或
参考速度信号。Supervisor 仅在根高度低于 `0.20 m` 且倾角小于
`0.35 rad` 时选择它，侧翻或仰翻会进入 `DISABLED`。
Init 退出还要求根高度超过 `0.36 m`、倾角与机身速度合格且四个足端持续接触地面，
避免两腿支撑或仍在晃动时过早切换到 stand policy。

当前 TOML 已指向状态驱动版 Init 导出模型
`logs/rsl_rl/bpx_init/2026-09-14_23-03-13/exported/policy.onnx`。旧的时序奖励 checkpoint
不能直接作为新策略使用；需要重新训练时执行：

```bash
python scripts/rsl_rl/train.py --task BPX-Init-v0 --headless
```

```bash
python sim2sim/run_mujoco.py \
    --backend onnx --viewer --gamepad --gamepad-deadman \
    --supervisor --duration 120
```

`--stand-policy` 和 `--init-policy` 仍可用于临时覆盖 TOML，但日常运行不再需要写
长路径。不使用手柄、不加 `--supervisor` 且不提供专家策略参数时，程序仍走单 locomotion
运行路径；但默认物理复位姿态现在是趴姿，该路径主要保留给接口诊断。若没有 Init policy，
检测到高度过低或倾角过大后会进入 `DISABLED` 并结束仿真。
CSV 会额外记录 `behavior_mode`、`behavior_policy`、`transition_alpha`、原始 command 和
经过状态机过滤的 command。

默认情况下，机器人倾倒后仍继续仿真，直到达到 `--duration` 指定的时长或手动关闭窗口。
`WAITING_INIT` 和 `INIT` 本来就处于低高度，因此这两个阶段不会触发物理跌倒终止。若调试
Stand/Walk 时希望在高度过低、倾角过大或躯干触地后立即结束，可显式添加：

```bash
python sim2sim/run_mujoco.py --backend onnx --viewer --vx 0.2 --terminate-on-fall
```

## 4. 与 Isaac Lab 接口对拍

先激活安装了 Isaac Lab 的环境，再执行：

```bash
python sim2sim/probe_isaaclab.py \
    --headless --device cpu --vx 0.2 --vy 0 --wz 0 \
    --json-output sim2sim/logs/isaac_interface.json
```

脚本关闭训练随机化和观测噪声，输出实际关节顺序、默认角、观测项、48维观测及首帧策略动作。它不会修改训练环境配置。

若 Isaac Sim 内的 TorchScript运行时有问题，可先跳过策略，导出零动作物理基准：

```bash
python sim2sim/probe_isaaclab.py --headless --device cpu --skip-policy --steps 100 \
    --json-output sim2sim/logs/isaac_zero_action.json
```

生成相同命令或零动作的 MuJoCo CSV后，可以直接计算各状态组误差：

```bash
python sim2sim/compare_trajectories.py \
    --isaac sim2sim/logs/isaac_zero_action.json \
    --mujoco sim2sim/logs/mujoco_zero_action.csv
```

## 配置与诊断

配置集中在 [`config/bpx_flat.toml`](config/bpx_flat.toml)。运行器按关节名称建立映射，不依赖 `qpos` 或电机数组碰巧同序。

运行日志记录机体速度、姿态、关节状态、目标角、动作和力矩。若策略在 MuJoCo 中表现异常，建议按顺序检查：

1. 关节正方向与默认角；
2. 48维观测的顺序、坐标系和单位；
3. 动作缩放与 PD 参数；
4. 力矩饱和；
5. 接触、摩擦、惯量等动力学差异。

使用 `--viewer` 时，窗口左上角会显示运行时诊断信息，包括 `base_link height`、base position、倾角、机体坐标系速度、足端接触数、当前行为模式/策略、command，以及 action/torque 最大值。这里的 `base_link height` 与运行器内部的 `base_height()` 一致，表示配置中 `robot.base_body_name`（当前为 `torso`）的 body 原点世界坐标 Z 值，不是机身最低点到地面的距离。

当前 MJCF 文件内部声明的步长为 `0.002 s`，运行器会显式覆盖为训练使用的 `0.005 s`。默认采用逐物理步显式 PD，因为它在当前模型上的A/B测试优于 MuJoCo内置位置伺服。把配置中的 `control.mode` 改为 `implicit`、`simulation.integrator` 改为 `implicitfast`，可以继续比较两种积分语义。

## 当前验证状态

- MJCF、12关节名称映射、48维观测、坐标变换、质心速度修正和零动作 PD 均已通过自动测试；
- 已导出的 TorchScript 与 ONNX 对相同首帧观测的动作最大误差约为 `2e-7`；
- 已处理 MuJoCo被动 viewer 自动关闭时的线程清理竞争，MuJoCo 3.13下零策略和 ONNX策略均可无段错误退出；
- Isaac Lab训练日志最终 `Train/mean_episode_length` 为约 `991/1000`，策略在源环境中已基本收敛；
- 已修正策略关节顺序：Isaac实际顺序为四个髋横滚、四个髋俯仰、四个膝关节，而不是按单腿排列；
- 修正后，名义参数下 `vx=0.2 m/s` 已稳定运行20秒，平均机体前向速度约 `0.187 m/s`，累计前进约 `3.72 m`；
- 确定性零动作轨迹对拍中，关节位置 RMSE约 `0.046 rad`、关节速度 RMSE约 `0.224 rad/s`，仍存在 PhysX与MuJoCo动力学差异。

当前剩余工作属于精细动力学对齐：继续比较相同初始条件下的策略轨迹，再逐项拟合执行器响应、接触和摩擦。当前机器仍会报告 Warp/CUDA驱动接口警告；探测脚本通过 CPU仿真并直接读取环境返回的 policy观测绕过了额外 PhysX tensor view，可正常导出 JSON。

## 当前边界

- `velocity_limit=20 rad/s` 作为训练接口参数被记录，但 MuJoCo没有与 PhysX `velocity_limit_sim` 完全等价的单一设置；当前运行器不直接裁剪物理状态。
- PhysX与MuJoCo的接触求解器不同，因此即使机器人参数相同，也不应期待轨迹逐帧完全一致。
- 第一阶段目标是稳定站立和低速前进；完整速度范围应在这两项通过后逐级验证。
