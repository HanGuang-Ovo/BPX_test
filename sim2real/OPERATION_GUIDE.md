# BPX sim2real 操作手册

主流程：环境与配置 → 网络 → 官方初始化 → 只读窗口核对 → 离线回放 → 站立接管。
全部命令在项目根目录执行。窗口只读取反馈，策略输出仅供对照。

窗口的标注截图、参数解释和故障排查见 [窗口图解使用手册](WINDOW_GUIDE.md)。

## 1. 环境与配置

```bash
cd /home/hanguang/BPX_test
conda activate bpx-real
python -m pip install -e '.[inference]'
python -m pip install --no-index --find-links bpx_sdk_open-master/wheelhouse bpx-sdk-open==1.0.9
```

已有环境可跳过安装；确认终端和 IDE 使用同一个解释器。
首次使用时复制本机配置，不覆盖已有文件：

```bash
cp -n sim2real/config/runtime.toml sim2real/config/local.runtime.toml
cp -n sim2real/config/joint_calibration.template.toml sim2real/config/local.calibration.toml
mkdir -p outputs/sim2real
```

在 `local.runtime.toml` 设置 `calibration_file="local.calibration.toml"`。
模型与坐标文件的相对路径相对于 runtime TOML 所在目录。
实机只由 `run --live` 启动；普通预检查和窗口不发送运动命令。

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml check
```

确认 SDK 和 ONNX Runtime 可用、模型路径存在。部署缺项不妨碍只读窗口。
`check` 检查运行资源，不要求身份、固件保护或实机验证确认表。

## 2. 网线连接，同时用 Wi-Fi 上网

官方默认 RJ45 地址为 `10.21.20.1`。有线网卡与机器人使用匹配的子网，
例如厂商确认 /24 时，电脑可设置 `10.21.20.2/24`，直连网卡不填默认网关或 DNS。
Wi-Fi 保留上网默认路由，避免两个网络子网重叠。

```bash
ip -br addr
ip route get 10.21.20.1
ping -c 3 10.21.20.1
ss -lunp
```

路由应走实际连接机器人的有线网卡；代理/VPN 的 TUN 路由可能截获该地址。
本地普通反馈 UDP 端口为 9873；后续关节控制反馈端口为 7895。
ping 成功不等于 SDK 反馈正常；还需检查 UDP 回程、防火墙和端口占用。

## 3. 按厂商流程上电和初始化

先完成厂商规定的本体支撑、上电与初始化。
官方 `MotionLevelControl` 示例会调用 `setZeroPositionsFlag()`，其文档要求标零姿态下
足底、小腿、小腿与大腿连接处均接触地面。不能在任意姿态运行该标零示例。
官方关节示例不调用标零接口，而是读取当前角度作为第一条目标。
是否每次上电需要零位准备、是否已经由 App/固件完成，以当前固件流程为准。

本窗口不调用标零、站立、阻尼或任何运动命令。
`setZeroJointCommand()` 是清零控制命令，不是编码器/模型标零。
SDK 的 connect() 可能自动同步机器人时间。

## 4. 打开只读窗口

```bash
python sim2real/visualize.py
```

填写机器人 IP，点击“连接只读反馈”。当前机器尚未连接时，先看演示：

```bash
python sim2real/visualize.py --demo
```

DEMO 明确使用模拟反馈，模型仍是当前配置的 ONNX；不创建 SDK 对象。
`--no-policy` 可以只看反馈/姿态；模型加载失败也会继续显示反馈并报告错误。

### 4.1 关节对照

窗口按当前训练策略顺序显示 12 行，`P,S` 为策略索引、配置 SDK 索引。

- SDK q：所选 SDK 通道原始值；官方 q：按官方通道顺序直接读取。
- 当前 q/dq：应用配置中的通道、sign、offset 后的值。
- action、策略 q、SDK 目标：所选模型的原始输出及转换目标，未发送。
- 目标−反馈：单帧差值；不是机器人正在跟踪的目标误差。

首次采用官方映射、sign=1、offset=0。核对真实腿、关节与骨架方向对应。
不再要求先辨识全部 SDK 通道或用量具逐关节求 offset。
若发现偏差，先检查初始化、模型和关节排列；确实需要时才增加覆盖：

```toml
[[joints]]
name = "fr_hip_roll_joint"
sdk_index = 3
sign = 1
offset = 0.0
```

示例数值仍是官方默认值。修改配置后断开并重启窗口，避免推理时改变坐标契约。
未覆盖的关节自动采用默认值；SDK 索引必须保持 0～11 的一一映射。
范围默认来自官方 URDF；如使用更保守的工作范围，添加 minimum/maximum。

### 4.2 姿态对照

SDK 四元数按 xyzw 解算，界面显示 ZYX roll/pitch/yaw（度）、模长和机体重力方向。
灰色：SDK 默认机体→世界解释与官方关节顺序；青色：当前配置；橙色：模型目标。
首次水平姿态检查重力接近 `[0,0,-1]`、静止角速度接近零，再用小幅倾斜核对轴与方向。
没有偏差就保持单位安装矩阵和零 bias；不要求重做传感器标定。
出现偏差时再确认 quaternion_frame、sensor_to_body、gyro_bias，避免重复补偿固件已处理的旋转。
骨架只作方向对照，没有实际高度/接触估计。RPY 在 pitch=±90° 附近不唯一，以骨架和四元数为准。

### 4.3 策略预览

窗口自动读取电脑 USB/蓝牙手柄，显示映射后的 vx/vy/wz，不提供手动模型切换或速度输入框。
在反馈有效、运动状态 Motion=6、步态 Walk=0 时，与部署共用的路由自动选择 Stand/Walk/STOPPING；趴卧、起身、阻尼和其他步态时暂停。Stand command 恒为零，Walk 使用状态机限制变化率后的手柄输入，松开摇杆后经 STOPPING 回到 Stand。
只推理选中模型，首帧填满 H=10 历史，上一动作初始为零；切换时重建相应历史。顶部“当前策略”指只读状态机选择，不是对固件内部权重的确认；手柄和策略目标均未发送。

单个手柄自动识别；多个设备时使用：

```bash
python sim2real/visualize.py --list-gamepads
python sim2real/visualize.py --gamepad-index 0
```

默认沿用 MuJoCo 的手柄轴映射、死区 0.1，速度范围来自 command_limit。手柄缺失或不可读时显示原因，并暂停策略。

当前两份模型必须 `[1,450]→[1,12]`，默认每腿 `[0,0.7,-1.4]`，action scale=0.5。
窗口显示 `q_default + 0.5 × action`，没有实机接管混合、目标保护或模型驱动的反馈变化。
过去动作来自预览递推，不是机器人实际执行历史。它是影子推理，不能用来证明步态稳定。
超过模型范围的目标仅以红色提示，不限制实机发送。策略不再按倾角暂停；四元数模长须在 [0.5,1.5]。

“官方定义 / 当前配置”页对照索引、sign/offset、范围、增益、频率、模型文件和示例目标。
官方示例 kp=100、kd=2、目标 `[0,0.8,-1.8]` 仅为对照，不应自动替换已训练策略参数。

反馈/IMU 缺失显示“—”。断连或会话变化时重建窗口预览历史；数据年龄和灰色旧值仅用于显示，不作为运动保护条件。
更新率是 SDK 普通反馈通道，不因窗口调度频率就等于真实 UDP 更新率。

## 5. 保存与回放

窗口启动时指定新输出文件，可以保存原始反馈和预览：

```bash
python sim2real/visualize.py --output outputs/sim2real/monitor-01.jsonl
```

文件不覆盖。同一输出文件不能重复开启连接；下一次运行使用新文件名。
也可使用无窗口采集：

```bash
python sim2real/tools/run.py --config sim2real/config/local.runtime.toml inspect \
  --duration 10 --output outputs/sim2real/state-01.jsonl
python sim2real/tools/run.py --config sim2real/config/local.runtime.toml analyze \
  outputs/sim2real/state-01.jsonl
```

窗口回放原始录制及其中的电脑手柄输入，不连接 SDK 或读取当前手柄；旧 inspect 录制没有输入时只显示反馈/姿态：

```bash
python sim2real/visualize.py --recording outputs/sim2real/state-01.jsonl
```

播放结束停用最后一帧的预览。CLI 的严格离线策略回放：

```bash
python sim2real/tools/run.py --config sim2real/config/local.runtime.toml replay \
  outputs/sim2real/state-01.jsonl --output outputs/sim2real/policy-01.jsonl
```

录制反馈不跟随新目标，不构成闭环验证。CLI 回放和实机发送均不增加关节目标限幅、限速或限矩。

## 6. 实机运动

实机运动保护仅保留两项：

1. 反馈、观测、目标和策略输出必须为有限数，IMU 四元数模长须在 `[0.5,1.5]`，随后归一化。
2. 运行到期、Ctrl+C 或异常退出后，若已发送关节命令且连接可用，尝试发送三次 `kp=0、kd=1、dq=0、tff=0` 阻尼命令，然后断开；退出尝试不保证送达。

身份、SDK/主机版本及 alignment 仅作记录；不再要求 verified、固件限矩、看门狗、关节工作范围、限速、跟踪误差、接管姿态、倾角、反馈年龄/组时间差或循环超期确认。起身不依赖里程计高度或接地估计。`minimum/maximum` 仅用于窗口的模型范围对照。`[safety]`、`[firmware]`、`[supervisor]` 已从当前配置移除；旧配置中的这些段落也不参与运行。

模型形状、关节排列、坐标变换、端口和轨迹时序仍需符合运行接口。`command_limit` 定义手柄的机体速度范围，Stand/Walk 迟滞和 command 平滑用于策略路由；它们不限制关节目标。

先做离线资源检查：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml run
```

关闭只读窗口，避免 SDK 反馈端口占用。实机启动命令：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml run \
  --live --gamepad-index 0
```

省略 `--output` 时自动在 `outputs/sim2real` 创建带上海时间戳的 JSONL；也可指定新文件名。由 `--live` 选择实机执行，旧 `--arm` 参数已删除。

```text
固件 Motion=6 → STAND
其他固件状态 → WAITING_INIT → 按 RB/R1 → INIT
INIT → 标称轨迹 4 秒完成 → STAND
STAND → 持续摇杆输入 → WALK
WALK → 松开摇杆 → STOPPING → 1.5 秒后 STAND
```

等待阶段保持实测关节角，不运行策略。Init 捕获实测姿态，先收脚到支撑点，再抬升；沿用五次足端轨迹和 BPX 解析 IK，取消模型关节范围、速度/力矩、跟踪误差、接地、高度、倾角及起身超时保护。轨迹完成后立即进入 Stand，不额外等待稳定验收。

Sony 手柄 R1 对应 RB；左摇杆控制 vx/vy，右摇杆控制 wz。`--gamepad-deadman` 可选，按住 L1/LB 才接收摇杆速度，RB/R1 独立生效。松杆通过 STOPPING 回 Stand。手柄/SDK 读写失败属于 I/O 错误，会退出并执行阻尼尝试。

`--duration 60` 可指定接管后的总运行时长；省略则持续运行。固定指令入口可选 `--command-source constant --vx 0 --vy 0 --wz 0`，没有 RB 起身请求源；非 Motion=6 状态将保持等待。

## 7. 软件验证

```bash
python -m unittest discover -s sim2real/tests -p 'test_*.py' -v
```

测试使用模拟 SDK 和录制数据，覆盖有限数/四元数、退出阻尼、取消旧门槛后的接管、起身和策略路由；不发送实机命令。
