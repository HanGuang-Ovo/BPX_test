# BPX Isaac Lab → MuJoCo 跨仿真运行

本目录将 Isaac Lab / RSL-RL 导出的 BPX 行走、站立和趴卧起身策略放入 MuJoCo 闭环执行。本文按 2026-09-17 的代码与 TOML 配置更新；设计原理、排错过程和历史实验见 [完整工作流](BPX_SIM2SIM_WORKFLOW.md)。

## 1. 准备环境与模型

所有命令在项目根目录执行。MuJoCo 环境可与 Isaac Lab 分开；配置读取使用 `tomllib`，需要 Python 3.11+：

```bash
conda create -n bpx-sim2sim python=3.11 -y
conda activate bpx-sim2sim
python -m pip install -r sim2sim/requirements.txt
python -c "import mujoco, onnxruntime; print(mujoco.__version__, onnxruntime.__version__)"
```

依赖文件声明 NumPy、MuJoCo 和 ONNX Runtime 的最低版本，没有锁定精确版本。TorchScript 后端及导出一致性检查还需要同一环境中的 PyTorch。

在 Isaac Lab 环境中用 `scripts/rsl_rl/play.py` 分别加载三个任务的目标检查点，生成 `exported/policy.onnx` 和 `exported/policy.pt`。训练与导出命令见 [项目 README](../README.md)。

更新 [config/bpx_flat.toml](config/bpx_flat.toml) 中的 `[paths]`：

| 字段 | 用途 |
| --- | --- |
| `mjcf` | 带网格资源的 MuJoCo 模型 |
| `locomotion_policy` | ONNX 主策略 |
| `stand_policy` | Supervisor 可选站立 ONNX 策略 |
| `init_policy` | Supervisor 可选起身 ONNX 策略 |
| `torchscript_policy` | TorchScript 主策略及导出对比模型 |

相对路径以项目根目录解析。默认路径指向具体实验，不会自动选择最新模型；`logs/` 被 Git 忽略，模型不保证随仓库提供。默认 ONNX 与 TorchScript 主策略路径可能来自不同实验，做一致性检查前必须指向**同一个检查点**的两种导出。

## 2. 先检查模型和 PD

```bash
python sim2sim/validate_setup.py
python sim2sim/run_mujoco.py --backend zero --duration 2 --no-realtime
```

`validate_setup.py` 加载 MJCF，按名称构建 12 关节/执行器映射，打印观测与动作形状，并执行零动作 PD 数值检查。通过只表示模型和控制接口具备运行前提，不证明策略性能，也不能替代训练侧关节顺序核对。

动作零点对应默认站姿；从默认趴姿执行 zero 后端会尝试趋向站姿，不是“趴姿保持”，也不等于神经网络 Init 策略。

## 3. 默认流程：手柄起身与多策略控制

默认复位为根高度 `0.133 m` 的腹部朝地趴姿，髋俯仰 `0.98 rad`、膝 `-2.62 rad`。在 Linux 上连接手柄、配置好三个 ONNX 模型后执行：

```bash
python sim2sim/run_mujoco.py --list-gamepads
python sim2sim/run_mujoco.py \
    --backend onnx --viewer --gamepad --supervisor --gamepad-deadman \
    --duration 120 --log sim2sim/logs/supervisor.csv
```

1. `WAITING_INIT`：固定 action 保持趴姿，所有速度 command 为零。
2. 按一下 **RB**（`BTN_TR=311`），产生一次起身请求并进入 `INIT`，不必持续按住。
3. 起身完成并连续稳定后转入 `STAND`。
4. 按住 **LB** 并推摇杆后进入 `WALK`；松开后经 `STOPPING` 减速，再回到 `STAND`。

LB 仅在指定 `--gamepad-deadman` 时作为速度指令使能，与 RB 起身互不冲突。松开 LB 会让原始速度指令归零，实际机器人停止仍经过状态机与平滑控制。

### 手柄映射与选项

| 输入 | 默认 Linux 轴 | 输出 | 满量程 |
| --- | --- | --- | --- |
| 左摇杆上下 | `ABS_Y=1` | `vx`，上推为正 | ±1.0 m/s |
| 左摇杆左右 | `ABS_X=0` | `vy`，左推为正 | ±0.5 m/s |
| 右摇杆左右 | `ABS_RX=3` | `wz`，左推为正 | ±1.0 rad/s |

默认软件死区为 0.1。可用 `--gamepad-index 0` 或 `--gamepad-device /dev/input/eventN` 选择设备；使用 `--gamepad-vx-axis`、`--gamepad-vy-axis`、`--gamepad-wz-axis` 调整轴码，对应 `--gamepad-*-sign` 取 -1 或 1。

控制器使用 Linux input-event，不需要 `hidapi`、`python-evdev` 或仓库之外的个人接收脚本。轴范围通过设备查询获得；布局不符时需按实际设备调整轴映射。设备权限问题可先检查 `ls -l /dev/input/eventN`。

限制速度的示例：

```bash
python sim2sim/run_mujoco.py --backend onnx --viewer --gamepad --supervisor \
    --gamepad-max-vx 0.4 --gamepad-max-vy 0.2 --gamepad-max-wz 0.5 \
    --duration 120
```

交互控制保持实时限速，不添加 `--no-realtime`。

### 六状态与策略路由

```text
支持的趴姿 + 可用 Init → WAITING_INIT ── RB ──→ INIT ── 稳定确认 ──→ STAND
持续运动指令：STAND → WALK
停车：WALK → STOPPING → STAND
停止过程中指令恢复：STOPPING → WALK
异常状态 / 不支持的起身姿态 / Init 倾角超限 → DISABLED → 结束本次仿真
```

| 状态 | 动作来源 / 速度指令 |
| --- | --- |
| `WAITING_INIT` | `prone_hold` 固定动作；零指令 |
| `INIT` | 起身策略；零指令 |
| `STAND` | 站立策略；零指令 |
| `WALK` | 行走策略；平滑后的运动指令 |
| `STOPPING` | 行走策略；指令逐渐降到零 |
| `DISABLED` | 不执行策略，运行器结束本次仿真 |

普通行走切换使用指令活跃度 `max(abs(command) / [1.0, 0.5, 1.0])`：进入阈值 0.12、持续 0.15 秒；退出阈值 0.05、持续 0.30 秒，且 WALK 至少驻留 0.30 秒。STOPPING 在低速持续 0.20 秒或停止等待达到 1.50 秒后转 Stand；指令恢复也可回到 Walk。不同策略对象间默认用 0.30 秒混合 action。

未站立判定为高度 <0.20 m 或倾角 >0.70 rad。仅当高度低且倾角 <0.35 rad、Init 模型可用时，进入起身等待；不会把任意跌倒自动送入 Init。Init 倾角达到 0.55 rad 时中止。

起身完成需高度 >0.36 m、倾角 <0.25 rad、平面速度 <0.05 m/s、角速度模长 <0.10 rad/s、四足接触，连续保持 0.50 秒。部署端当前不检查训练奖励中的前足宽度，完成条件与训练 `recovered` 并不完全相同。

### 启用规则、缺失模型与终止

- 显式 `--supervisor`、指定 `--stand-policy`/`--init-policy`，或使用手柄且 TOML 配置了 Init 路径，均会启用 Supervisor。
- `--stand-policy`、`--init-policy` 临时覆盖相应 ONNX 路径；主策略格式由 `--backend` 决定。
- 缺少主策略时无法运行对应神经网络后端；可选 Stand/Init 文件缺失会警告。Stand 缺失时回退到零指令行走策略；默认趴姿缺少 Init 时进入 DISABLED 并结束。
- 不使用手柄、Supervisor 和专家路径参数时，运行单主策略，不存在 DISABLED 状态判断。
- 单策略默认不因跌倒提前结束，可加 `--terminate-on-fall`。Supervisor 的 DISABLED 终止独立于这个选项；启用物理跌倒终止时，WAITING_INIT 和 INIT 阶段免于该项检查。

## 4. 无手柄：站姿单策略测试

仅加 `--supervisor` 不会自动发送起身请求。直接测试行走策略或做零动作轨迹对拍时，需要从与 Isaac 行走任务一致的站姿开始。

先复制配置：

```bash
cp sim2sim/config/bpx_flat.toml sim2sim/config/bpx_standing.toml
```

在副本的 `[initial_state]` 中替换对应字段（不要重复添加同名字段）：

```toml
base_position = [0.0, 0.0, 0.40]
base_quaternion_wxyz = [1.0, 0.0, 0.0, 0.0]
joint_position = [0.0, 0.0, 0.0, 0.0, 0.7, 0.7, 0.7, 0.7, -1.4, -1.4, -1.4, -1.4]
```

保持 `default_joint_position` 原值，它是动作与关节相对观测的零点。然后执行：

```bash
python sim2sim/run_mujoco.py --config sim2sim/config/bpx_standing.toml \
    --backend onnx --vx 0.2 --vy 0 --wz 0 --duration 10 --no-realtime \
    --log sim2sim/logs/forward.csv

python sim2sim/run_mujoco.py --config sim2sim/config/bpx_standing.toml \
    --backend onnx --viewer --vx 0.2

python sim2sim/run_mujoco.py --config sim2sim/config/bpx_standing.toml \
    --backend torchscript --device cpu --duration 10
```

TorchScript 测试需先确认 `torchscript_policy` 是目标模型。需要跌倒即停时添加 `--terminate-on-fall`。

## 5. 导出一致性与轨迹对拍

### 同一模型的两种导出

将 TOML 的 `locomotion_policy` 与 `torchscript_policy` 指向同一检查点的 ONNX / TorchScript 导出，然后运行：

```bash
python sim2sim/validate_policy_export.py --samples 100
```

默认比较最大绝对动作误差，容差为 `1e-5`。这验证格式一致性，不验证策略能否完成任务。

### 零动作物理基准

先在 Isaac Lab 环境运行：

```bash
python sim2sim/probe_isaaclab.py --headless --device cpu --skip-policy --steps 100 \
    --vx 0 --vy 0 --wz 0 --json-output sim2sim/logs/isaac_zero_action.json
```

再切换到 MuJoCo 环境，使用前面准备的**站姿配置**：

```bash
python sim2sim/run_mujoco.py --config sim2sim/config/bpx_standing.toml \
    --backend zero --vx 0 --vy 0 --wz 0 --duration 2.02 --no-realtime \
    --log sim2sim/logs/mujoco_zero_action.csv

python sim2sim/compare_trajectories.py \
    --isaac sim2sim/logs/isaac_zero_action.json \
    --mujoco sim2sim/logs/mujoco_zero_action.csv
```

使用 2.02 秒是为覆盖 Isaac 100 步的末帧 `t=2.00 s`；MuJoCo CSV 在策略更新前记样本，首行为 `t=0`。比较器使用最近时间样本，不做插值，也不拒绝时间覆盖不足，需先检查两份日志。

探测脚本关闭观测噪声、启动随机化与推扰，并固定种子、收窄重置范围。**它的关节名称和默认角元数据来自 TOML，观测项名称/维度也在脚本中声明；不是独立从 Isaac 运行时解析这些元数据。** 关节顺序还需核对训练启动日志和公共环境定义。

探测策略动作时须显式指定 `--policy /实际路径/policy.pt`，否则脚本使用早期实验的固定路径；它不自动跟随 TOML 的模型路径。非零指令长轨迹还需留意 command 重采样和站立掩码是否改变实际指令。

## 6. 公共接口、日志与边界

- 物理 200 Hz、策略 50 Hz；观测 48 维、动作 12 维。
- 观测顺序：机身线速度(3)、角速度(3)、重力投影(3)、指令(3)、相对关节角(12)、关节速度(12)、上一实际动作(12)。
- 关节顺序：四个 hip_roll、四个 hip_pitch、四个 knee；每组内部 `fl/fr/hl/hr`。
- 目标角：`q_default + 0.5 × action`；默认角为 roll=0、pitch=0.7、knee=-1.4 rad。
- 显式 PD 名义参数：Kp=40、Kd=1、力矩限制 ±30 N·m。配置记录速度限制 20 rad/s，但运行器不直接裁剪物理关节速度。
- Supervisor 模式的初始 `last_action` 对应复位姿态；后续使用混合后实际施加的动作。单策略模式初始动作历史为零。

运行器补偿 MuJoCo 自由关节原点与 torso 质心的速度差，推理不添加训练噪声。PD 隐式/显式语义及接触、惯量等差异仍需验证。

CSV 记录原始与过滤后的 command、行为状态/策略、动作混合系数、基座状态、关节位置/速度/目标角/动作/力矩。Viewer 的 `base_link height` 是配置中 torso body 原点的世界 Z 坐标，不是机身最低点离地距离。

旧文档中的 `2e-7` 导出误差、20 秒前进与零动作 RMSE 属于早期实验记录，见 [工作流历史结果](BPX_SIM2SIM_WORKFLOW.md#14-历史验证结果)。它们不能替代当前三份策略和状态切换的复验。本目录提供验证工具；本次文档同步未重新训练或运行物理仿真。

## 三级起伏地形测试场

新场景在加载时从原 `bpx.xml` 生成，机器人资产、关节顺序、PD 参数和策略输入不变。
原 `config/bpx_flat.toml` 继续用于平地测试。新增两个配置中的策略路径初始复制自平地配置；
更换检查点时应同步修改对应配置的 `[paths]`，不会自动跟随平地配置变化。

| 区域 | 世界坐标 X | 宽度 | 高度范围上限 |
| --- | --- | --- | --- |
| LEVEL 0 / FLAT | -2～2 m | 4 m | 0 |
| LEVEL 1 / MILD | 2～6 m | 4 m | ±2 cm |
| LEVEL 2 / MODERATE | 6～10 m | 4 m | ±4 cm |

起伏为固定种子的平滑随机波叠加，主要波长 15～30 cm；幅值范围是上限，不保证每块地形达到上下限。
起伏入口、难度切换、末端和两侧有 0.5 m 渐变带。两侧平坦通道供返回使用。
这是固定路面几何，不包含可滚动碎石。各区域摩擦和接触材料保持一致。
高度场替换原平面，避免平面填平负高度凹处；测试场外围的平整地面延伸至 X/Y=±50 m。

**手柄起身和行走**（原点趴姿出生，RB 起身）：

```bash
python sim2sim/run_mujoco.py \
  --config sim2sim/config/bpx_terrain.toml \
  --backend onnx --viewer --gamepad --supervisor \
  --duration 120 --log sim2sim/logs/terrain_gamepad.csv
```

**直接测试行走策略**（原点站姿出生，无需起身触发）：

```bash
python sim2sim/run_mujoco.py \
  --config sim2sim/config/bpx_terrain_standing.toml \
  --backend onnx --viewer --vx 0.2 --vy 0 --wz 0 \
  --duration 60 --terminate-on-fall --log sim2sim/logs/terrain_walk.csv
```

窗口默认展示整个测试场，可缩放查看机器人及起伏。两侧边线为灰/绿/橙色，英文标签位于各区旁边，
均不参与碰撞。状态栏显示当前区域、世界高度、局部地面高度和机身离地高度。

CSV 保留原 `base_z` 世界高度，新增 `ground_z`、`base_clearance`、`terrain_region`。
Supervisor 和跌倒终止使用机身正下方地面对应的垂直离地高度；这不是四足支撑面拟合高度。
地面查询只用于仿真监督和记录，不进入 48 维策略观测。脚和躯干接触检测覆盖高度场和外围地面。
地面使用可视化组 5，运行器自动开启显示；手动隐藏该组只影响显示，不影响碰撞。

配置中 `[terrain]` 的 `seed` 可用于改变路面；比较不同策略时请保持相同 seed。
训练策略没有更新，停车切换后的 Stand 以及崎岖区域重新起身仍需单独评估。

验证命令：

```bash
python sim2sim/validate_setup.py --config sim2sim/config/bpx_terrain_standing.toml
python -m unittest discover -s sim2sim -p 'test_terrain.py' -v
```

早期 ±1/±2 cm 版本使用配置中现有 locomotion ONNX 做过一次 50 秒、vx=0.2 m/s 的无界面测试，
进入全部三级区域并到达约 (10.43, -0.90) m，未触发跌倒终止。
这是单一种子、单次直行结果，横向偏移仍然明显，不代表转向、停车或多地形通过率。

地形幅度在 `bpx_sim2sim/terrain.py` 的 `MILD_AMPLITUDE` 和 `MODERATE_AMPLITUDE` 中设置（单位 m）。
高度场缩放、Z 偏移、归一化和标签从这两个值自动计算。当前 ±4 cm 对应竖直缩放 0.08 m、
Z 偏移 -0.04 m；不会通过截断高度数据削平峰谷。上述早期行走结果不能代表当前难度表现。
