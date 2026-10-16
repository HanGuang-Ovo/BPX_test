# 上下楼梯策略的 MuJoCo 回放

训练入口和课程见根目录 [README](../README.md#当前上下楼梯工程怎么读) 与 [STAIRS_TRAINING.md](../STAIRS_TRAINING.md)。本页只说明部署回放；修改 TOML 不会改变 Isaac Lab 的训练场景。

## 当前运行方式

使用 `config/bpx_stairs_history.toml`。流程为趴卧 → PD 起身 → STAND → WALK，WALK 始终使用 `paths.locomotion_policy` 指定的统一行走 Actor。Supervisor 管理起身、站立、行走和停车；当前运行器不再根据地形切换平地/楼梯专家。旧路由配置解析和相关兼容文件仍保留，但不参与当前 WALK 策略选择。

Actor 输入为 450 维、10 帧历史，输出为 12 维动作；不输入基座线速度真值或楼梯高度。历史按观测项分块、从最旧到最新排列，在 50 Hz 更新。Stand 使用单独的站立模型，Init 默认使用足端轨迹、IK 和 PD。

上下楼训练只开放前进、后退。平地指令包括侧走和旋转，不能据此认为楼梯上的侧走、转弯已经训练完成。上楼和下楼表现应分别验证。

## 先选择并导出模型

在项目根目录、训练环境中执行；替换下面的实际 checkpoint 路径：

```bash
python sim2sim/export_checkpoint.py \
  --checkpoint /path/to/RUN/model_XXXX.pt \
  --output /path/to/RUN/exported
```

导出器读取对应运行的 `params/agent.yaml`，检查 TorchScript 与 Actor 输出一致性，并保存检查点 SHA256 和输入输出维度；仅支持未启用经验归一化的前馈 ActorCritic。需要 torch、rsl_rl、PyYAML 和 onnx，不需要启动 Isaac Sim。

然后更新 TOML 的两个路径，使它们来自**同一个 checkpoint**：

```toml
[paths]
locomotion_policy = "/path/to/RUN/exported/policy.onnx"
torchscript_policy = "/path/to/RUN/exported/policy.pt"
```

当前仓库 TOML 的 ONNX 与 TorchScript 默认路径来自不同运行目录，不能直接用来做两种格式的一致性对比；应先按上述步骤替换。这些路径指向本机历史产物，并不自动选择最新训练模型。模型和 Stand 策略位于被 Git 忽略的 `logs/`，迁移机器时需要自行提供。

## 运行

以下命令从项目根目录执行，使用已安装 `sim2sim/requirements.txt` 的 Python 环境。

```bash
# 固定前进指令：自动 PD 起身后开始行走
python sim2sim/run_mujoco.py \
  --config sim2sim/config/bpx_stairs_history.toml \
  --backend onnx --auto-init --viewer --vx 0.3 --vy 0 --wz 0 \
  --duration 30 --log outputs/sim2sim/stairs_forward.csv

# 手柄：楼梯验证先限制为前进/后退
python sim2sim/run_mujoco.py --list-gamepads
python sim2sim/run_mujoco.py \
  --config sim2sim/config/bpx_stairs_history.toml \
  --backend onnx --viewer --supervisor --gamepad --gamepad-deadman \
  --gamepad-max-vx 0.5 --gamepad-max-vy 0 --gamepad-max-wz 0 \
  --duration 300 --log outputs/sim2sim/stairs_gamepad.csv
```

按 RB 发起身，等待进入 STAND；按住 LB 并推左摇杆前后行走；松开 LB 后经过 STOPPING 回到站立。多手柄时使用 `--gamepad-index`。需要曲线可添加 `--velocity-plot`、`--torque-plot`，具体说明见 [sim2sim README](README.md)。场景出口不会自动停车。

## 回放场景与训练场景的区别

| 项目 | Isaac Lab 训练 | 当前 MuJoCo 回放 |
| --- | --- | --- |
| 布局 | 上楼、下楼、平地独立地块，中心出生并向外运动 | 沿世界 +X 连续上楼、顶平台、下楼 |
| 阶数与高度 | 11 个踢面、8 级高度课程 | 上下各 6 次 15 cm 高度变化，总高 0.9 m |
| 普通踏面 | 20 cm | 20 cm |
| 控制入口 | PPO 自动采样速度指令 | 固定指令或手柄 |
| 起步 | 中央平台站姿重置 | 平地趴姿，PD 起身后交接 |

MuJoCo 楼梯宽 2 m，从 X=2.0 m 开始上升，顶部平台 X=3.0–5.0 m，下楼至 X=6.0 m 后恢复平地。`[terrain]` 可以修改尺寸。实体箱体有竖直踢面，不是插值斜坡；局部离地高度、接触和射线查询包含台阶。

## 退出条件与验证范围

当前 TOML 设置 `terminate_on_tilt=false`，单独倾角超过阈值不触发对应的终止路径；低高度、不支持的起身姿态、非有限状态等条件仍生效。具体阈值以 TOML 和 Supervisor 为准。

```bash
python -m unittest discover -s sim2sim -p 'test_*.py' -v
```

几何、接口、导出一致性和短时间起身检查只能证明工程流程，不证明策略已经能通过当前台阶。历史模型的 CSV、预览和诊断仍保存在 `outputs/`；旧的低台阶通过记录不能作为当前 15 cm 场景或新模型的通过结论。
