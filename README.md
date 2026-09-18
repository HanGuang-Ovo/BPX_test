# BPX 四足机器人多策略训练与跨仿真验证

本项目基于 Isaac Lab 的管理器式强化学习环境，使用 RSL-RL 的 PPO 算法分别训练 BPX 四足机器人的行走、静止站立和趴卧起身策略，并将导出的策略部署到 MuJoCo，验证不同物理引擎中的控制表现（sim2sim）。项目现有环境说明以 Isaac Lab 2.2.1 为基线。

行走、站立、起身三类策略共享机器人模型、48 维单帧观测和 12 维动作接口；行走和站立分别提供平地与崎岖地形训练任务。MuJoCo 端通过上层状态机（Supervisor）选择策略，支持手柄速度指令、起身触发、停止过渡、动作平滑和 CSV 日志记录。

## 任务与功能

| Gym 任务名称 | 训练目标 | 回合时长 | 默认训练迭代数 | 日志目录 |
| --- | --- | --- | --- | --- |
| `BPX-Locomotion-v0` | 平地前后、横向移动及偏航速度跟踪 | 20 秒 | 1500 | `logs/rsl_rl/bpx_locomotion/` |
| `BPX-Locomotion-Rough-v0` | 平台复位、课程起伏地形上的盲走 | 20 秒（含边界截断） | 1500 | `logs/rsl_rl/bpx_rough/` |
| `BPX-Stand-v0` | 零速度指令下保持稳定站姿 | 15 秒 | 1500 | `logs/rsl_rl/bpx_stand/` |
| `BPX-Stand-Rough-v0` | 崎岖区域复位、站稳课程训练 | 15 秒 | 1500 | `logs/rsl_rl/bpx_rough_stand/` |
| `BPX-Init-v0` | 从腹部朝地的趴卧状态起身并保持站立 | 4 秒 | 3000 | `logs/rsl_rl/bpx_init/` |
| `BPX-Test-v0` | 行走任务的兼容入口 | 20 秒 | 1500 | `logs/rsl_rl/bpx_flat/` |

行走任务的指令范围为 `vx ∈ [-1.0, 1.0] m/s`、`vy ∈ [-0.5, 0.5] m/s`、`wz ∈ [-1.0, 1.0] rad/s`。平地行走按静止、纯旋转和混合运动分层采样，崎岖行走的三个速度分量独立均匀采样。站立和起身任务的三个指令分量始终为零。

Init 针对腹部朝地、机身接近水平的趴姿训练，不覆盖侧翻或仰翻后的翻身恢复。原有任务使用平地场景；新增 Rough 任务使用课程起伏地形。

## 项目结构

```text
BPX_test/
├── source/BPX_test/                 # 可编辑安装的 Isaac Lab 扩展包
│   ├── config/extension.toml        # 扩展元数据
│   └── BPX_test/
│       ├── BPX_structure/           # URDF、USD、MJCF 与网格资产
│       └── tasks/manager_based/bpx_test/
│           ├── bpx_base_env_cfg.py  # 公共机器人、场景、观测与动作
│           ├── bpx_locomotion_env_cfg.py
│           ├── bpx_rough_env_cfg.py # 崎岖行走与地形配置
│           ├── bpx_rough_stand_env_cfg.py # 崎岖站立与站稳课程
│           ├── rough_terrain_geometry.py # 起伏地形网格生成
│           ├── bpx_stand_env_cfg.py
│           ├── bpx_init_env_cfg.py
│           ├── agents/             # 各任务的 PPO 配置
│           └── mdp/                # 自定义奖励、指令采样与课程
├── scripts/
│   ├── list_envs.py                 # 列出已注册的 BPX 任务
│   ├── zero_agent.py               # 零动作环境检查
│   ├── random_agent.py             # 随机动作环境检查
│   ├── validate_rough_task.py       # 崎岖行走配置与课程验证
│   ├── validate_rough_stand_task.py # 崎岖站立重置、奖励与课程验证
│   └── rsl_rl/                     # 训练、回放与策略导出
├── sim2sim/
│   ├── config/bpx_flat.toml         # 平地、趴姿起身流程
│   ├── config/bpx_terrain.toml      # 三级地形、趴姿起身流程
│   ├── config/bpx_terrain_standing.toml # 三级地形、站姿直接测试
│   ├── bpx_sim2sim/                # MuJoCo 运行器、策略与手柄接口
│   ├── run_mujoco.py               # 跨仿真运行入口
│   ├── validate_setup.py           # 模型、关节映射与 PD 检查
│   └── validate_policy_export.py   # ONNX / TorchScript 输出对比
├── logs/rsl_rl/                    # 训练生成的检查点、参数与导出模型
└── outputs/                        # 运行生成的输出
```

## 环境准备

以下命令均在项目根目录执行。训练和回放使用已安装 Isaac Lab、Isaac Sim 及 RSL-RL 的 Python 环境；MuJoCo 推理可以使用独立环境。

### 安装训练扩展

将本项目放在 Isaac Lab 安装目录之外，并在 Isaac Lab 对应环境中执行：

```bash
python -m pip install -e source/BPX_test
python scripts/list_envs.py
```

`list_envs.py` 应列出上表中的六个任务。项目安装脚本不会自动安装完整的 Isaac Lab / Isaac Sim 运行环境；若使用 Isaac Lab 自带的启动脚本，可将命令中的 `python` 替换为 `/path/to/IsaacLab/isaaclab.sh -p`。

**首次克隆后请检查机器人资产。** 训练需要 `source/BPX_test/BPX_test/BPX_structure/usd/bpx.usd` 及其引用的 USD 文件。仓库的 `.gitignore` 忽略了 USD 文件，因此仅克隆代码可能缺少这些资产。需要复制完整 USD 资产，或使用 Isaac Lab 的 URDF 转换工具，参照 [转换配置](source/BPX_test/BPX_test/BPX_structure/usd/config.yaml) 从 `BPX_structure/bpx/urdf/bpx.urdf` 重新生成；保留足端固定关节对应的独立刚体，并保持转换参数与训练配置一致。

### 安装 MuJoCo 推理环境

sim2sim 配置读取使用 Python 标准库 `tomllib`，因此该环境需要 Python 3.11 或更高版本。使用 Conda 时可执行：

```bash
conda create -n bpx-sim2sim python=3.11 -y
conda activate bpx-sim2sim
python -m pip install -r sim2sim/requirements.txt
python sim2sim/validate_setup.py
```

依赖包括 NumPy、MuJoCo 和 ONNX Runtime。仅使用 ONNX 推理无需安装 PyTorch；使用 TorchScript 或导出一致性检查时，还需要在该环境中安装 `torch`。

## 训练、回放与导出

本节命令在 Isaac Lab 环境中执行。

### 检查环境并开始训练

可先用少量环境检查资产加载和训练流程：

```bash
# 零动作检查：动作零点对应默认站姿
python scripts/zero_agent.py --task BPX-Stand-v0 --num_envs 4

# 随机动作检查
python scripts/random_agent.py --task BPX-Locomotion-v0 --num_envs 4

# 短训练检查：验证训练循环、日志和检查点保存
python scripts/rsl_rl/train.py \
    --task BPX-Locomotion-v0 --num_envs 64 --max_iterations 2 --headless
```

分别训练三个策略：

```bash
python scripts/rsl_rl/train.py --task BPX-Locomotion-v0 --headless
python scripts/rsl_rl/train.py --task BPX-Stand-v0 --headless
python scripts/rsl_rl/train.py --task BPX-Init-v0 --headless
```

默认并行环境数为 2048，可通过 `--num_envs 512` 等参数降低显存占用。训练默认使用三层 `[128, 128, 128]` 的 ELU 网络，每个环境每轮采集 24 步，每 50 次迭代保存一次检查点；`--max_iterations` 可覆盖任务的默认迭代数。

常用参数还包括 `--seed`、`--run_name`、`--experiment_name` 和 `--logger`。训练结果保存在 `logs/rsl_rl/<实验名>/<时间戳>[_运行名]/`，其中 `params/` 记录本次运行的环境和算法配置。

恢复训练时，将下列运行目录和检查点文件名替换为实际值：

```bash
python scripts/rsl_rl/train.py \
    --task BPX-Locomotion-v0 --headless --resume \
    --load_run "<运行目录名>" --checkpoint "model_<迭代编号>.pt"
```

### 从平地策略微调崎岖地形策略

两个崎岖任务均采用 ±0.5～±4 cm 的 8 级起伏地形，保持原有单帧观测和动作接口，不向策略输入地形高度。

| 任务 | 重置位置 | 难度课程 | 奖励 |
| --- | --- | --- | --- |
| `BPX-Locomotion-Rough-v0` | 中央 2×2 m 平地 | 根据崎岖区域通过距离、停留时间和速度跟踪表现升降级 | 沿用平地行走奖励 |
| `BPX-Stand-Rough-v0` | 90% 崎岖区域、10% 中央平地 | 崎岖样本连续站稳两回合升级，失败降级 | 高度改为相对附近地面，其余沿用 Stand |

站立每回合 15 秒，排除前 0.5 秒落脚期后，稳定时间占比须达到 90%，同时满足速度、倾斜和漂移条件。平地样本不参与升级。地面高度查询仅用于仿真重置与奖励，不增加策略观测或实机传感器需求。

以下为本机已有平地检查点的微调示例；其他机器需替换为实际文件路径：

```bash
# 崎岖行走
python scripts/rsl_rl/train.py \
    --task BPX-Locomotion-Rough-v0 --num_envs 512 --headless --resume \
    --checkpoint logs/rsl_rl/bpx_locomotion/2026-09-15_13-15-43_stable/model_4999.pt \
    --max_iterations 1500

# 崎岖站立
python scripts/rsl_rl/train.py \
    --task BPX-Stand-Rough-v0 --num_envs 512 --headless --resume \
    --checkpoint logs/rsl_rl/bpx_stand/2026-09-15_00-13-24_stable/model_499.pt \
    --max_iterations 1500
```

`--resume --checkpoint` 加载训练状态，不是导出模型。`--max_iterations` 表示本次继续训练的迭代数；新日志分别写入 `bpx_rough` 和 `bpx_rough_stand`。PPO 检查点不保存地形课程状态，恢复训练后课程从最低级开始。

详细参数与评估方法见 [崎岖行走训练](ROUGH_TERRAIN_TRAINING.md) 和 [崎岖站立训练](ROUGH_STAND_TRAINING.md)。需要检查实现时可运行：

```bash
python -u scripts/validate_rough_task.py --headless
python -u scripts/validate_rough_stand_task.py --headless
```

### 查看训练曲线

```bash
tensorboard --logdir logs/rsl_rl
```

在浏览器打开终端显示的地址，通常为 `http://localhost:6006`。结合任务奖励、回合长度、动作平滑程度和仿真回放判断策略效果；短训练检查仅用于验证流程。曲线含义见 [训练曲线说明](RL_TRAINING_CURVES.md)。

### 回放并导出策略

```bash
# 按默认加载规则选择检查点，使用一个机器人实时回放
python scripts/rsl_rl/play.py \
    --task BPX-Locomotion-v0 --num_envs 1 --real-time

# 指定检查点；请替换为实际文件路径
python scripts/rsl_rl/play.py \
    --task BPX-Locomotion-v0 --num_envs 1 \
    --checkpoint "logs/rsl_rl/bpx_locomotion/<运行目录>/model_<迭代编号>.pt"
```

`play.py` 加载检查点后会自动在该检查点所在目录下生成：

```text
exported/
├── policy.pt      # TorchScript 推理模型
└── policy.onnx    # ONNX 推理模型
```

站立和起身策略采用相同流程，分别使用 `BPX-Stand-v0`、`BPX-Init-v0` 及其对应检查点。崎岖策略应选择对应的 Rough 任务，例如：

```bash
python scripts/rsl_rl/play.py \
    --task BPX-Stand-Rough-v0 --num_envs 4 \
    --checkpoint "logs/rsl_rl/bpx_rough_stand/<运行目录>/model_<迭代编号>.pt"
```

崎岖行走使用 `BPX-Locomotion-Rough-v0` 和 `bpx_rough` 目录下的检查点。

训练检查点 `model_*.pt` 与导出的 `policy.pt` 用途不同，MuJoCo 端加载导出模型。

## MuJoCo 跨仿真运行

本节命令在 MuJoCo 推理环境中执行。

### 配置模型路径

根据运行流程选择配置；三个 TOML 相互独立，不会继承或同步策略路径：

| 配置 | 地形 / 初始姿态 | 用途 |
| --- | --- | --- |
| [bpx_flat.toml](sim2sim/config/bpx_flat.toml) | 平地 / 趴姿 | 平地手柄起身与多策略测试 |
| [bpx_terrain.toml](sim2sim/config/bpx_terrain.toml) | 三级起伏地形 / 趴姿 | 手柄 RB 起身、崎岖行走与站立切换 |
| [bpx_terrain_standing.toml](sim2sim/config/bpx_terrain_standing.toml) | 三级起伏地形 / 站姿 | 直接测试行走策略；文件名中的 standing 指初始姿态 |

编辑实际传给 `--config` 的文件的 `[paths]`，将 `locomotion_policy`、`stand_policy`、`init_policy` 指向各自导出的 `policy.onnx`；使用 TorchScript 时同时更新 `torchscript_policy`。

配置中的相对路径均相对于项目根目录解析。默认文件记录了具体实验目录，且 `logs/` 被 Git 忽略，因此这些检查点不保证随代码仓库提供，也不会自动切换到新训练的模型。

无需策略文件即可检查模型和 PD 控制：

```bash
python sim2sim/validate_setup.py
python sim2sim/run_mujoco.py --backend zero --duration 2 --no-realtime
```

这些命令用于检查 MJCF 加载、关节映射、观测构造与数值稳定性，不代表训练策略已通过运动性能验证。

### 手柄控制与多策略切换

配置好三个 ONNX 策略后，在 Linux 上连接手柄并运行：

```bash
python sim2sim/run_mujoco.py --list-gamepads
python sim2sim/run_mujoco.py \
    --backend onnx --viewer --gamepad --supervisor \
    --duration 120 --log sim2sim/logs/gamepad.csv
```

默认配置从趴姿开始，运行流程如下：

1. `WAITING_INIT`：保持趴姿，等待按下手柄 **RB**。
2. `INIT`：调用起身策略，速度指令保持为零；满足高度、姿态和稳定性条件后进入 `STAND`。
3. `STAND`：调用站立策略；摇杆指令持续超过阈值后进入 `WALK`。
4. `WALK`：调用行走策略跟踪速度；松开摇杆后经 `STOPPING` 减速，再回到 `STAND`。

默认左摇杆上下控制 `vx`、左右控制 `vy`，右摇杆左右控制 `wz`。可加 `--gamepad-deadman`，要求按住 **LB** 才接受非零速度指令；多个手柄可用 `--gamepad-index 0` 或 `--gamepad-device /dev/input/eventN` 选择。

Supervisor 使用指令变化率限制、切换迟滞、稳定时间确认和动作混合来平滑过渡。遇到不支持的起身姿态、起身倾角过大等条件时进入 `DISABLED`，运行器结束本次仿真。缺少站立模型时会使用零指令行走策略作为回退；缺少起身模型时无法完成默认趴姿起身流程。

### 三级地形与崎岖策略测试

测试场依次为平地、轻微起伏（±2 cm）、较大起伏（±4 cm），带区域标签和平坦返回通道。地形振幅在 [terrain.py](sim2sim/bpx_sim2sim/terrain.py) 的 `MILD_AMPLITUDE`、`MODERATE_AMPLITUDE` 修改，单位为米；高度场归一化和标签随参数调整。

在 `bpx_terrain.toml` 中，将 `locomotion_policy` 指向崎岖行走导出模型，将 `stand_policy` 指向崎岖站立导出模型，`init_policy` 保留已有起身模型，然后运行：

```bash
python sim2sim/run_mujoco.py \
    --config sim2sim/config/bpx_terrain.toml \
    --backend onnx --viewer --gamepad --supervisor \
    --duration 120 --log sim2sim/logs/terrain_gamepad.csv
```

按 RB 在出生平地起身后，分别检查三个区域上的行走与站立，并测试“行走 → 停止 → 站立 → 再行走”。现有 Init 仍是平地起身策略，崎岖行走和站立训练不包含崎岖地形起身。

CSV 中 `base_z` 是世界高度，`ground_z` 为当地地面高度，`base_clearance` 为局部离地高度，`terrain_region` 为当前区域。具体说明见 [三级地形使用说明](sim2sim/README.md#三级起伏地形测试场)。

从站姿直接测试行走可使用现成配置，先单独更新该配置的行走模型路径：

```bash
python sim2sim/run_mujoco.py \
    --config sim2sim/config/bpx_terrain_standing.toml \
    --backend onnx --viewer --vx 0.2 --vy 0.0 --wz 0.0 --duration 50
```

### 无手柄的单策略测试

默认 TOML 的初始姿态是趴姿。若要直接测试行走策略，请先复制一份配置，并将 `[initial_state]` 的 `base_position` 改为 `[0.0, 0.0, 0.40]`，将 `joint_position` 设为与 `default_joint_position` 相同的站姿：

```bash
cp sim2sim/config/bpx_flat.toml sim2sim/config/bpx_standing.toml
# 按上述说明修改 bpx_standing.toml，再运行
python sim2sim/run_mujoco.py \
    --config sim2sim/config/bpx_standing.toml \
    --backend onnx --vx 0.2 --vy 0.0 --wz 0.0 \
    --duration 10 --no-realtime --log sim2sim/logs/forward.csv
```

需要窗口时加 `--viewer`；需要按墙钟时间播放时去掉 `--no-realtime`。仅添加 `--supervisor` 不会自动触发趴姿起身，当前命令行流程通过手柄 RB 发送起身请求。

若要比较 ONNX 与 TorchScript 导出结果，先确认配置中的两个模型来自**同一个检查点**，再执行：

```bash
python sim2sim/validate_policy_export.py --samples 100
```

## 训练与部署的公共接口

接口定义集中在 [bpx_base_env_cfg.py](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_base_env_cfg.py)，MuJoCo 端参数位于实际使用的 `sim2sim/config/*.toml`。修改接口后，需要同步训练、导出和部署配置。

| 项目 | 当前配置 |
| --- | --- |
| 物理步长 / 频率 | `0.005 s` / 200 Hz |
| 策略执行频率 | 每 4 个物理步更新一次，即 50 Hz |
| 观测 / 动作维度 | 48 / 12 |
| 动作含义 | `目标关节角 = 默认关节角 + 0.5 × action` |
| 默认关节角 | 髋横滚 `0.0`、髋俯仰 `0.7`、膝关节 `-1.4` rad |
| 名义 PD 增益 | `Kp = 40`、`Kd = 1` |
| 力矩 / 速度限制配置 | `30 N·m` / `20 rad/s` |

48 维观测按以下顺序拼接：

```text
机身坐标系线速度(3) + 角速度(3) + 重力投影(3)
+ 速度指令[vx, vy, wz](3)
+ 相对默认姿态的关节角(12) + 相对关节速度(12) + 上一步动作(12)
```

关节按“类型优先”排列：先四个髋横滚，再四个髋俯仰，最后四个膝关节；每组内部均为 `fl → fr → hl → hr`（左前、右前、左后、右后）。不要直接用 MJCF 中的关节存储顺序替代策略顺序，详见 [关节顺序映射说明](sim2sim/JOINT_ORDER_MAPPING_GUIDE.md)。

训练使用观测噪声、摩擦和执行器增益随机化；MuJoCo 推理使用无附加噪声的观测。Isaac Lab 使用隐式 PD 执行器，MuJoCo 默认逐物理步执行显式 PD，因此参数一致仍需进行跨仿真验证。MuJoCo 的 `joint_position` 决定复位姿态，`default_joint_position` 决定策略动作零点，两者含义不同。

## 常见问题

| 现象 | 检查方法 |
| --- | --- |
| 无法导入 `isaaclab` 或 `BPX_test` | 确认当前解释器属于 Isaac Lab 环境，并重新执行可编辑安装 |
| 找不到 `bpx.usd` 或引用资产 | 检查完整 USD 资产是否已复制或转换生成；USD 文件被 Git 忽略 |
| 训练显存不足 | 降低 `--num_envs`，并使用 `--headless` |
| MuJoCo 找不到策略文件 | 更新 TOML 的 `[paths]`，检查是否已通过 `play.py` 导出模型 |
| 趴姿一直等待、不起身 | 检查 Init 模型是否加载，并在 Supervisor 手柄流程中按下 RB |
| ONNX / TorchScript 对比失败 | 先确认两份导出来自同一检查点；默认配置可能指向不同实验 |
| 编辑器无法解析 Isaac Sim 模块 | 在 VS Code 中运行 `setup_python_env` 任务，按提示填写 Isaac Sim 安装路径 |

## 开发与参考文档

修改任务目标时，优先调整对应的 `bpx_*_env_cfg.py`、`agents/` 和 `mdp/`；修改公共观测、动作或关节顺序时，需要同时检查三个策略和 sim2sim 的兼容性。

仓库提供 pre-commit 配置，可手动运行格式检查：

```bash
python -m pip install pre-commit
pre-commit run --all-files
```

- [崎岖行走课程与微调](ROUGH_TERRAIN_TRAINING.md)
- [崎岖站立课程与微调](ROUGH_STAND_TRAINING.md)
- [训练—导出—MuJoCo 工作流](sim2sim/BPX_SIM2SIM_WORKFLOW.md)
- [MuJoCo 运行器与手柄使用说明](sim2sim/README.md)
- [训练环境配置说明](BPX_ENV_SETUP.md)
- [强化学习训练曲线说明](RL_TRAINING_CURVES.md)
- [行走策略改进路线](BPX_IMPROVEMENT_ROADMAP.md)
- [行走控制参考资料](BPX_LOCOMOTION_LITERATURE.md)
- [项目交互式思维导图](BPX_PROJECT_MINDMAP.html)

部分专题文档保留了早期实验记录；具体奖励权重、采样范围和运行参数以当前源码及本次训练保存的 `params/` 为准。
