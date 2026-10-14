# BPX sim2real：实机部署与独立调试工具

部署代码和调试工具分别放在两个包中。实机起身、Stand/Walk/STOPPING、策略推理与 SDK 收发在 `bpx_sim2real/`；只读窗口、诊断、状态采集和离线回放在 `tools/bpx_sim2real_tools/`。

```text
sim2real/
├── run.py                         # 部署入口：check / run
├── bpx_sim2real/                   # 起身、策略控制、SDK、坐标和配置
├── visualize.py                   # 只读窗口快捷入口
├── tools/
│   ├── run.py                     # 调试入口：monitor / inspect / analyze / replay
│   └── bpx_sim2real_tools/         # 窗口、诊断、采集与回放实现
├── config/                        # 共用的运行与坐标配置
└── tests/                         # 模拟 SDK、控制流程和工具边界回归
```

工具复用部署的坐标、状态和 Actor；部署运行时不加载工具包或 Tkinter。部署日志与调试录制共用 `bpx_runtime/jsonl.py` 写入器，录制读取、重采样、统计和回放只属于工具包。

建议使用 `python sim2real/tools/run.py ...` 或安装后的 `bpx-real-tools ...` 运行工具。原有 `python sim2real/run.py monitor/inspect/analyze/replay ...` 命令仍按需转发，`python sim2real/visualize.py ...` 保持可用。重构后重新执行 `python -m pip install -e '.[inference]'`，更新 `bpx-monitor` 和新增的工具入口。

通过官方 SDK 读取 BPX 状态，在可视化窗口中对照 12 个关节的反馈、Stand/Walk 策略输出和四元数解算姿态。先确认数据和坐标对应关系，再准备实机接管。

窗口不发送运动或标零命令。默认使用官方关节索引、`sign=1`、`offset=0` 和 IMU 默认坐标；只有发现实际不一致时才添加覆盖，不需要先填写 12 份重复标定参数。

## 文档入口

| 文档 | 适合什么时候阅读 |
| --- | --- |
| [文件结构与职责说明](FILE_STRUCTURE.md) | 查看每个文件的用途、部署与工具边界，以及起身、策略推理和 SDK 收发的调用关系 |
| [窗口图解使用与问题分析](WINDOW_GUIDE.md) | 看懂每一列、三个 q、姿态、曲线和参数页，按现象排错 |
| [操作手册](OPERATION_GUIDE.md) | 逐步安装、联网、初始化、录制、回放和准备实机接管 |
| [硬件接入说明](HARDWARE_SETUP.md) | 查看连接和回读记录 |

![窗口阅读区域标注](docs/window_guide/figures/01-overview.png)

图中反馈来自实际窗口的 **DEMO 模拟数据**，策略预览使用配置中的 ONNX 模型，不是实机测量。图解手册包含完整截图、12 张标注/示意图和八类问题的排查步骤。图片可点击放大；截图使用 24 号字展示全部列，启动默认字号为 32。

## 快速开始

以下命令在项目根目录执行。已有 `bpx-real` 环境和本机配置时，跳过创建与安装步骤。

### 1. 准备运行环境

```bash
conda create -n bpx-real python=3.11 -y
conda activate bpx-real
python -m pip install -e '.[inference]'
python -m pip install --no-index --find-links bpx_sdk_open-master/wheelhouse bpx-sdk-open==1.0.9
```

运行环境需要 Python 3.11+、NumPy、官方 SDK 和 Tkinter；策略预览还需要 ONNX Runtime。SDK wheel 必须匹配 Python、系统和架构，并兼容机器人固件。若提示没有 Tkinter，先运行 `python -m tkinter` 检查该环境。

### 2. 准备本机配置

```bash
cp -n sim2real/config/runtime.toml sim2real/config/local.runtime.toml
cp -n sim2real/config/joint_calibration.template.toml sim2real/config/local.calibration.toml
mkdir -p outputs/sim2real
```

`cp -n` 不覆盖已有文件。在 `local.runtime.toml` 中设置：

```toml
calibration_file = "local.calibration.toml"
```

核对 `[network].robot_ip` 和 `[policy]` 的两份模型路径。配置相对路径相对于 runtime TOML 所在目录解析；模板中的训练导出文件不保证随仓库提供，也不会自动更新为新训练模型。

`identity` 和 `alignment` 仅作记录，不作为运动启动门槛。实际运动由 `run --live` 显式启动。

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml check
```

`check` 输出依赖、模型路径与哈希、坐标配置及缺少的运行资源。

### 3. 打开窗口

```bash
python sim2real/visualize.py
```

此入口优先使用 `sim2real/config/local.runtime.toml`，不存在时使用 `runtime.toml` 模板。也可显式指定：

```bash
python sim2real/visualize.py --config sim2real/config/local.runtime.toml
```

不连接机器人即可熟悉界面：

```bash
# 模拟反馈，使用配置中的真实 ONNX 模型进行预览
python sim2real/visualize.py --demo

# 不加载模型，只看模拟反馈与姿态
python sim2real/visualize.py --demo --no-policy
```

独立工具 CLI 提供同一窗口。工具 CLI 默认使用模板，使用本机配置时显式传入 `--config`：

```bash
python sim2real/tools/run.py --config sim2real/config/local.runtime.toml monitor
```

安装本项目后也可使用 `bpx-monitor`；新增入口需重新执行可编辑安装。

### 4. 接上网线，读取实机反馈

按厂商流程完成本体支撑、上电和必要初始化，再配置有线网卡、填写机器人 IP，点击“连接只读反馈”。网线连接机器人时，Wi-Fi 可以继续上网；有线直连接口不配置上网默认网关，具体步骤见 [操作手册](OPERATION_GUIDE.md)。

窗口只创建官方 `RequestRobotState`，不创建运动/关节控制对象。SDK 的 `connect()` 本身可能同步机器人时间。关闭窗口只结束读取，不负责让本体停车或坐下。

### 5. 连接电脑 USB/蓝牙手柄

单个手柄自动识别，沿用 MuJoCo 的 Linux 手柄映射：左摇杆上下/左右对应 vx/vy，右摇杆左右对应 wz。死区 0.1，速度范围使用 `[control].command_limit`。多个手柄时先列出，再指定设备：

```bash
python sim2real/visualize.py --list-gamepads
python sim2real/visualize.py --gamepad-index 0
# 也可指定实际设备路径
python sim2real/visualize.py --gamepad-device /dev/input/eventN
```

手柄未连接、拔出或设备不可读时显示 `—` 和原因，不伪装成零输入；恢复后重建预览。DEMO 使用模拟手柄自动切换，不读取真实设备。读取电脑手柄不会让机器人行走，也不会触发起身。

## 窗口应该怎么读

按顺序检查：**连接来源与身份 → joint/IMU 年龄 → 关节映射与姿态 → 模型状态与策略目标 → 曲线**。

| 列或区域 | 含义 |
| --- | --- |
| 关节 / P,S | 策略索引 P 和当前配置的 SDK 通道 S，均从 0 开始 |
| SDK q | 当前配置所选 SDK 通道的原始角度 |
| 官方 q | 按官方关节通道直接读取的角度，也是反馈，非默认站姿 |
| 当前 q / dq | 模型角 `sign × SDK q + offset`；模型角速度 `sign × SDK dq` |
| SDK 力矩* | 原始反馈，不做方向转换；单位以固件说明为准 |
| action | 所选 Stand/Walk 模型的原始输出 |
| 策略 q | `q_default + action_scale × action`，未接管混合、未限幅 |
| SDK 目标 | 策略目标转换回 SDK 坐标，仅展示，未发送 |
| 目标−反馈 | `策略 q − 当前 q`，不是闭环跟踪误差 |
| 灰色 / 青色骨架 | 官方默认解释 / 当前配置转换后的反馈姿态 |
| 橙色骨架 | 当前模型的目标关节姿态，尚未执行 |

角度使用 rad，角速度使用 rad/s；四元数按 xyzw 显示，RPY 按 ZYX 解算并使用度。骨架采用官方 URDF 连杆尺寸，不估计实机离地高度或足端接触。默认转换相同时三个 q 相等，灰色和青色骨架重合；这仍需与物理本体核对。

`—` 表示没有有效显示值，不表示 0。joint/IMU 年龄表示主机观察到更新标记变化后经过的时间，不是端到端网络延迟。反馈缺失、过期或会话结束时，策略预览暂停并重建历史；旧反馈和历史曲线可能仍保留在屏幕上。

窗口根据有效的机器人运动状态和电脑手柄输入自动选择 Stand/Walk/STOPPING，使用同样的切换阈值、延时和 command 变化率限制；窗口不执行实机起身，只推理选中模型。Stand command 恒为零，松开摇杆后经过 STOPPING 再回 Stand。趴卧、起身、阻尼、其他步态或手柄不可读时暂停策略预览；不按倾角停用。

顶部显示自动选择的策略、固件状态/步态/控制来源，以及只读的实时 vx/vy/wz；不提供手动策略切换、速度输入框或应用按钮。这里的“当前策略”是只读状态机选择，不代表 SDK 已确认固件执行本机 ONNX 权重。手柄输入和策略目标均未发送。模型加载失败不阻止读取反馈。

![顶部信息放大图：自动选择的策略、SDK 状态与电脑手柄实时输入](docs/window_guide/figures/12-auto-policy-input.png)

上图分别展示 DEMO 零输入时的 Stand 和持续输入时的 Walk。第二行的固件步态与第一行的本机模型选择是不同信息；第三行显示手柄输入，不是可编辑的预览指令。

详细算例、颜色含义、姿态核对与故障排查见 [窗口图解手册](WINDOW_GUIDE.md)。

### DEMO 中青色骨架为什么看起来不动

DEMO 用于演示反馈显示、坐标转换、策略推理和自动切换。它没有物理仿真，也不会把策略目标执行后生成新的反馈，因此切换到 Walk 后，青色骨架不会跟随橙色目标迈腿行走。

![DEMO Walk 骨架标注：青色模拟反馈与橙色未执行目标，以及刷新检查方法](docs/window_guide/figures/11-demo-feedback-target.png)

| 显示内容 | DEMO 中的数据来源与变化 |
| --- | --- |
| 灰色 / 青色骨架 | 同一份程序生成的关节和四元数反馈，分别按官方定义和当前配置解释；默认配置下两者重合 |
| 关节反馈 | 在默认站姿附近做幅度 `0.035 rad`（约 `2°`）的正弦变化，腿部动作很小 |
| 机身姿态 | 四元数生成缓慢的旋转变化；Roll/Pitch/Yaw 幅度分别约 `10.3° / 8.0° / 17.2°` |
| 机身位置 | 绘图原点固定，不向前移动，也不计算离地高度或足端接触 |
| 橙色骨架 | 自动选中模型的目标关节角，使用当前反馈的机身姿态绘制；目标不回写到模拟反馈 |

DEMO 的模拟手柄按 18 秒周期变化：前 6 秒零输入、中间 6 秒 `vx=0.35 m/s`、最后 6 秒零输入。状态机根据阈值和延时经历 Stand→Walk→STOPPING→Stand；这个过程只改变策略推理，不让反馈骨架执行步态。

判断界面是否正常刷新，可以选中一个关节，观察“当前 q / dq”和青色曲线，再观察姿态区 RPY 数值是否持续变化。**数值在变但骨架动作不明显**，通常是模拟关节摆幅小；如果连接状态仍显示 DEMO 运行中，而关节和 RPY 数值也一直不变，再检查刷新是否异常。主动断开后保留的最后一帧则不会继续变化。

要验证策略驱动机器人后的行走效果，使用 [MuJoCo sim2sim](../sim2sim/README.md)。接入实机时，青色骨架随实机反馈变化；只读窗口中的手柄输入和橙色目标仍不发送给机器人。

### 字号

默认正文和加粗数值均为 **32 号**，右上“字号”可选 12～48；表格可横向/纵向滚动。保存下次启动字号，在本机 runtime 配置中设置：

```toml
[monitor]
font_size = 32
```

下拉框只调整当前窗口，不写回配置。修改配置或模型文件后，需要关闭并重启窗口。

## 配置与策略契约

| 文件 / 项目 | 用途或默认值 |
| --- | --- |
| `config/runtime.toml` | 网络、模型、控制流程和窗口参数模板 |
| `config/local.runtime.toml` | 本机实际使用的运行配置 |
| `config/joint_calibration.template.toml` | 身份、整体对齐确认、IMU 和可选关节覆盖模板 |
| `config/local.calibration.toml` | 本机坐标配置，需由 runtime 的 `calibration_file` 引用 |
| SDK 默认映射 | 策略顺序对应 `[0,3,6,9,1,4,7,10,2,5,8,11]` |
| Walk / Stand 模型 | `[1,450] → [1,12]`，H=10，按观测项分别堆叠 |
| 默认每腿角度 / action scale | `[0,0.7,-1.4]` rad / `0.5` |
| 策略 / 部署发送频率 | 50 Hz / 模板 100 Hz；只读窗口不发送目标 |

当前 450 维 Actor 不使用基座线速度。未配置 `[[joints]]` 时自动使用官方通道、`sign=1`、`offset=0` 和 URDF 模型范围；旧的关节覆盖仍可加载，但逐关节 verified 标志不再参与就绪检查。

首次接入要核对默认四元数解释是否与真实方向一致；IMU 安装旋转默认单位矩阵、gyro_bias 默认零，不重复补偿固件已经处理的转换。模型范围不代表固件硬限矩、硬限速或实机可用范围已经确认。

“官方定义 / 当前配置”页提供参数对照。官方关节示例的 kp=100、kd=2 和目标 `[0,0.8,-1.8]` 是示例值，不应自动覆盖当前训练契约或增益。

## 保存与离线复查

```bash
# 连接后保存原始反馈与策略预览，使用新的文件名
python sim2real/visualize.py --output outputs/sim2real/monitor-01.jsonl

# 回放原始反馈，不连接 SDK
python sim2real/visualize.py --recording outputs/sim2real/monitor-01.jsonl

# 分析原始反馈统计
python sim2real/tools/run.py --config sim2real/config/local.runtime.toml analyze \
  outputs/sim2real/monitor-01.jsonl
```

输出文件排他创建，不覆盖已有记录；再次开启会话时换新文件名。窗口回放用 **当前配置和模型重新推理**，不会原样播放已保存的 `preview`。回放从录制读取手柄输入，不读取当前电脑手柄；旧录制未保存输入时，只显示反馈和姿态。要复现旧结果，保留原配置、模型文件和哈希；旧 preview 可在 JSONL 中直接查看。

## 实机部署

仅保留有限数/四元数模长 `[0.5,1.5]` 检查，以及退出时尝试 `kp=0、kd=1` 阻尼。关节范围、限速、限矩、倾角、反馈过期、循环超期、身份匹配和 verified 确认门槛均已移除；起身不要求高度/接地/跟踪误差验收。URDF 范围仅用于窗口对照，超范围不会拒绝接管。

```bash
# 离线资源预检查，不连接 SDK
python sim2real/run.py --config sim2real/config/local.runtime.toml run

# 关闭只读窗口后，在现场终端执行
python sim2real/run.py --config sim2real/config/local.runtime.toml run \
  --live --gamepad-index 0
```

日志自动创建，也可显式指定 `--output`；由 `--live` 选择实机执行，旧 `--arm` 参数已删除。等待实测姿态时按 RB/R1 起身，4 秒足端轨迹结束后进入 Stand，摇杆进入 Walk，松杆通过 STOPPING 回 Stand。起身继续采用 BPX 五次足端轨迹和解析 IK；实机行为不再使用仿真的高度/接地/跌倒 Supervisor。详细说明见 [操作手册](OPERATION_GUIDE.md#6-实机运动)。

退出尝试阻尼并断开，阻尼命令不保证送达。只读窗口始终不发送运动或退出阻尼命令。

部署 CLI 提供 `check` 和 `run`；独立工具 CLI 提供 `monitor`、`inspect`、`analyze` 和 `replay`，部署 CLI 仅保留这些工具命令的转发入口。

## 软件验证

```bash
python -m unittest discover -s sim2real/tests -p 'test_*.py' -v
```

测试使用模拟 SDK、时钟和策略，验证有限数/四元数、退出阻尼、取消旧门槛及完整起身状态切换，并包含部署不加载工具/窗口、新旧工具入口一致性和本地 ONNX 加载及回放测试。

## 官方依据

- [SDK 状态、运动与关节控制](https://github.com/mirrormerobotics/bpx_sdk_open)
- [关节索引和名称](https://github.com/mirrormerobotics/bpx_sdk_open/blob/master/include/motion_types.h)
- [关节控制示例](https://github.com/mirrormerobotics/bpx_sdk_open/blob/master/example/joint_level_control_example.py)
- [BPX URDF](https://github.com/mirrormerobotics/BPX/blob/master/bpx/urdf/bpx.urdf)
