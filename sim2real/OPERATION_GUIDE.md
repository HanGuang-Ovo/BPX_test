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
保持 `[safety].allow_motion=false`。

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml check
```

确认 SDK 和 ONNX Runtime 可用、模型路径存在。部署缺项不妨碍只读窗口。
`check` 的退出码 0 不代表运动已就绪。

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
超过模型范围的目标会以红色提示；姿态大幅倾斜仍可显示；超过 max_tilt_rad 时暂停策略预览。

“官方定义 / 当前配置”页对照索引、sign/offset、范围、增益、频率、模型文件和示例目标。
官方示例 kp=100、kd=2、目标 `[0,0.8,-1.8]` 仅为对照，不应自动替换已训练策略参数。

反馈/IMU 缺失显示“—”。断连、数据过期、会话变化时暂停预览并重建历史。
更新率是 SDK 普通反馈通道，不因窗口调度频率就等于真实 UDP 更新率。

## 5. 保存与回放

窗口启动时指定新输出文件，可以保存原始反馈和预览：

```bash
python sim2real/visualize.py --output outputs/sim2real/monitor-01.jsonl
```

文件不覆盖。同一输出文件不能重复开启连接；下一次运行使用新文件名。
也可使用无窗口采集：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml inspect \
  --duration 10 --output outputs/sim2real/state-01.jsonl
python sim2real/run.py --config sim2real/config/local.runtime.toml analyze \
  outputs/sim2real/state-01.jsonl
```

窗口回放原始录制及其中的电脑手柄输入，不连接 SDK 或读取当前手柄；旧 inspect 录制没有输入时只显示反馈/姿态：

```bash
python sim2real/visualize.py --recording outputs/sim2real/state-01.jsonl
```

播放结束停用最后一帧的预览。CLI 的严格离线策略回放：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml replay \
  outputs/sim2real/state-01.jsonl --output outputs/sim2real/policy-01.jsonl
```

`--enforce-limits` 在相关参数填齐后增加目标保护计算。录制反馈不跟随新目标，不构成闭环验证。

## 6. 对齐后再部署

确认关节/IMU 与实际模型对应后，在本机坐标文件中设置一次：

```toml
[alignment]
verified = true
```

不再填写每关节 mapping/direction/offset/limits_verified 或 imu.verified。
只读窗口不会自动设置这个标志，也不会因为数值相等就宣布实机已经验证。
填写 SDK 读出的 SN、型号、固件版本，并备份配置和采集记录。

实际运动仍需确认固件开发模式、站姿控制权切换、增益、命令频率、看门狗、
固件限矩、退出行为，以及目标变化率、实际速度、跟踪误差和接管姿态阈值。
这些保护项与“重新做电机标定”不同，不能根据 SDK 示例自动视为已通过。
默认完整命令发送频率改为示例中的约 100 Hz，策略保持 50 Hz。

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml run
```

### 实机手柄流程

实机与 sim2sim 共用 `bpx_runtime/supervisor.py` 的状态机和
`bpx_runtime/init_controller.py` 的足端五次轨迹、解析 IK。默认操作为：

```text
水平趴姿接管 → WAITING_INIT ── RB 按下沿 ──→ INIT
INIT ── 轨迹完成且连续稳定 ──→ STAND
STAND ── 持续摇杆输入 ──→ WALK
WALK ── 松杆 ──→ STOPPING ── 稳定确认或超时 ──→ STAND
STOPPING ── 再次持续输入 ──→ WALK
异常 → DISABLED → 结束本次控制
```

- `WAITING_INIT` 保持接管时实测趴姿，速度为零，不运行 Stand/Walk 模型；
  `[control].prone_joint_position` 仅用于接管范围核对，不把本体强拉到模板姿态。
- RB (`BTN_TR=311`) 每次松开到按下触发一次；一直按住不重复起身。
  已站立、起身中或行走中按 RB 不重新起身。若本体已符合站姿接管条件，可直接进入 Stand。
- `INIT` 从实测姿态开始，先收脚到低位支撑，再竖直伸展。轨迹按发送频率更新；
  跟踪误差过大时暂停，四足接地估计未满足时停在支撑点，超时则禁用并退出。
  轨迹完成、高度/倾角/线角速度及四足接地条件连续满足 `init_confirm_time` 后进入 Stand。
- 左摇杆上下/左右对应 vx/vy，右摇杆左右对应 wz，死区 0.1；速度范围来自
  `[control].command_limit`。可选 `--gamepad-deadman`：按住 LB 才接收速度，RB 独立生效。
- Stand/Walk 迟滞、最小行走时间、STOPPING 的再启动防抖、速度稳定确认和超时
  与 sim2sim 使用同一实现。STOPPING 继续使用 Walk 模型并平滑速度至零；Stand 指令恒为零。
  历史观测记录最终成功发送的 action，起身转 Stand 时重建 Stand 历史并平滑接管。

### 实机反馈与配置确认

SDK 没有足端接触传感器接口，本实现仅支持 **BPX 在水平地面上的接地估计**。
行为图与 sim2sim 一致，反馈来源不同：实机高度为 `LegOdom.position[2] - ground_z`，
平面速度来自 `velocity_body`，姿态/角速度来自已对齐 IMU。
即使 Actor 是 450 维、观测不使用线速度，实机状态机仍要求新鲜、已验证的里程计。
缺失、过期、会话变化及手柄断开/事件丢失均结束控制，不自动重连恢复运动。

接地估计同时要求：足底运动学高度接近配置地面、反馈力矩反解的向上支撑力超过阈值，
且 Jacobian 非奇异。使用官方 BPX 连杆和足底圆柱尺寸，力矩先重排、乘方向 sign 并减去
`torque_bias_nm`，按静态近似 `tau = -J^T F` 反解支撑力。
这不是接触传感器测量；腿部重力、惯性、里程计漂移、足底形变及吊装外力都可能影响估计。
日志字段明确记录 `feet_in_contact_estimated`、足底高度、支撑力和估计有效性。
没有可靠的高度/力矩反馈时，不能用固定高度或“默认四足接地”代替验收。

本机配置新增的确认项默认保持 false，完成实际核对后再设置：

| 配置项 | 确认内容 |
| --- | --- |
| `control.prone_handover_verified` | 实际水平趴姿在接管范围内，等待期间的支撑/增益正确 |
| `init_controller.verified` | 实机收脚/抬升轨迹、起身增益、限速、限矩、暂停及超时行为 |
| `supervisor.feedback_verified` | 里程计高度、速度、Nm 力矩方向、地面和接地估计可靠 |
| 标定文件 `odometry.verified` | SDK 里程计坐标、单位及更新信息已核对 |

`ground_z` 应按里程计 z 与实测机体参考原点高度确定，不假定启动时里程计零点就是地面。
`toe_radius/toe_center_depth/toe_half_width`、接地高度容差、支撑力阈值和力矩偏置须按实际足底
及反馈核对。`[init_controller]` 的 80/2 增益、30 Nm 和 4 秒轨迹来自仿真基线，不能视为
实机已验证参数。发送增益在起身前后平滑切换，软件限矩按当次实际发送增益计算，
同时受实测固件限矩和 Init 限矩约束。

原有身份、固件保护、站姿接管及全部软件保护仍需就绪。
`[safety].max_tilt_rad` 是额外的实机硬停止阈值；若比 Supervisor 阈值严格，会提前禁用，
新增起身不会自动放宽它。当前模板 0.45 rad 比 `init_abort_tilt=0.55` 更严格。

### 启动与退出

先只做预检查（不会创建 SDK 对象）：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml run
```

`blockers` 为空、运动条件已经确认，保持本体支撑和现场停止措施，再开启
`allow_motion` 并运行；默认自动选择单个手柄：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml run \
  --live --arm \
  --output "outputs/sim2real/live-$(date +%Y%m%d-%H%M%S).jsonl"
```

启动后终端显示状态，趴姿时等待 RB。多个手柄使用 `--gamepad-index 0`，或
`--gamepad-device /dev/input/eventN`，两者互斥。省略 `--duration` 会持续运行；
指定 `--duration 60` 则从接管开始计时 60 秒，等待 RB 的时间也计入。
不要在默认手柄模式添加固定速度参数。

仅从已站姿进行固定指令试验，可显式选 `--command-source constant --vx 0 --vy 0 --wz 0`；
该模式没有起身请求源，趴姿会一直等待，不会自动起身。

松开摇杆通过 STOPPING 减速回 Stand，进程继续保持支撑。
**Ctrl+C、运行到期或故障会退出控制，并尝试清零关节指令，不先完成 STOPPING；
清零可能释放支撑力。** 先确认实际退出/看门狗行为，不能把进程退出当作保持站立或坐下。
只读窗口仍不发送运动和起身命令，也不接管这个运行中的控制进程。

## 7. 软件验证

```bash
python -m unittest discover -s sim2real/tests -p 'test_*.py' -v
```

模拟测试验证坐标、历史、新鲜度、只读 SDK 生命周期和控制保护，不验证实际动力学或固件行为。
