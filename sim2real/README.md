# BPX sim2real：先检测与标定，再部署

逐步操作与结果解读请看 [操作手册](OPERATION_GUIDE.md)，包含当前步骤 1～4，以及参数确认后的回放、点动与部署流程。

采用独立 `sim2real/` 与公共 `bpx_runtime/`。现有训练代码与策略张量契约保留；
sim2sim 和 sim2real 共用推理后端、历史缓存、动作混合与坐标数学。

当前默认配置用于检测和离线分析。`check`、`analyze`、`compare-startup`、
`calibrate`、`replay`、默认 `jog` 和默认 `run` 都不创建 SDK 连接。
`inspect` 仅创建官方 `RequestRobotState`。SDK 的 `connect()` 可能自动同步机器人时间，
因此“只读”在这里指不发送运动/关节/标零命令，不表示 SDK 完全不改变机器人系统时间。

## 工程结构

```text
bpx_runtime/                   # 仿真与实机共用，不依赖 MuJoCo/Isaac Lab/SDK
  joints.py                    # 固定训练关节顺序、SDK 文档顺序
  math.py                      # 四元数与坐标变换
  observation.py               # 45/48 维单帧契约
  policy.py                    # ONNX/TorchScript/测试策略
  history.py                   # 按项展开、旧到新的历史
  transition.py                # command 限斜率、action 混合
sim2real/
  run.py                       # 统一分阶段入口
  config/
    runtime.toml               # 网络、模型、控制周期、保护、固件确认项
    joint_calibration.template.toml  # SN、IMU、关节标定模板
  bpx_sim2real/
    config.py                  # 配置验证、observe/jog/deploy 就绪报告
    calibration.py             # 索引、符号、零位、IMU 安装方向
    state.py                   # 原始状态与逐组新鲜度
    sdk_robot.py               # 普通只读/高频关节控制两个 SDK 模式
    recording.py               # JSONL、重采样、诊断报告
    controller.py              # Stand/Walk、450 维历史、接管混合
    safety.py                  # 身份、新鲜度、限位、限速、跟踪误差
    runner.py                  # replay/live/jog 生命周期与真实时钟
    cli.py                     # 检测、标定与部署命令
  tests/                       # 模拟 SDK 与时钟，全部离线测试
```

sim2sim 中原来的 `policy/history/transition` 保留为兼容导入；其他仿真入口继续使用原路径。
实机没有沿用依赖真实离地高度与足端接触的 MuJoCo Supervisor/PD Init。
本框架实机入口从已验证站姿接管，采用 Stand → Walk → Stopping → Stand；
Stopping 将速度指令平滑归零，经过配置时长切回 Stand。它不声称通过测量接触或线速度确认停车。
起身、自动恢复、机器人端固件和机载部署权限属于后续硬件适配内容。

## 环境与入口

所有示例在项目根目录执行。Python 3.11+；只读工具只需要 NumPy 和 SDK，
策略回放/部署额外需要 ONNX Runtime。

```bash
python -m pip install -e .
# 只有需要策略推理时安装：
python -m pip install -e '.[inference]'
```

SDK 从官方目录安装与你当前 Python/架构匹配的 wheel（须确认与机器人固件兼容）：

```bash
python -m pip install --no-index --find-links bpx_sdk_open-master/wheelhouse bpx-sdk-open==1.0.9
```

也可先安装 NumPy、SDK 后直接使用 `python sim2real/run.py`，不必安装本项目包。
安装本项目后 `bpx-real` 与 `python sim2real/run.py` 是同一入口。
从源码运行默认使用 `sim2real/config/runtime.toml`；普通 wheel 安装的默认配置位于
`sys.prefix/share/bpx-deployment/config/`，应复制到工作目录并修改模型路径。
模型文件不会随部署工具安装。

`--config` 放在子命令前。配置中的相对路径均相对于 runtime.toml 所在目录。
建议复制模板为本机文件并保持模型路径正确：

```bash
cp sim2real/config/runtime.toml sim2real/config/local.runtime.toml
cp sim2real/config/joint_calibration.template.toml sim2real/config/local.calibration.toml
```

在 `local.runtime.toml` 将 `calibration_file` 改为 `local.calibration.toml`。
`local.*.toml` 与 `outputs/` 中原始采集日志被 Git 忽略。
模板方向 `sign=1`、零位 `offset=0`、IMU 单位旋转矩阵均为未验证占位值。
未测量工作限位不填写；未测量实机限矩/速度数组保持空数组。

## 当前只执行步骤 1～4

### 1. 固件和控制条件

填写 [HARDWARE_SETUP.md](HARDWARE_SETUP.md)，向厂商核实开发模式、控制权切换、
完整关节帧发送频率、增益单位、固件限矩、命令超时和退出清零行为。
先记录事实，保留 `allow_motion=false` 和所有未经验证的确认项。

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml check
```

这一步打印当前解释器依赖、模型 SHA-256、通道建议和部署缺项，不连接机器人。
SDK 版本 `1.0.9` 和机器人运动主机版本是不同版本号。

### 2. 网络和只读通信

配置实机 IP；默认有线地址 `10.21.20.1`。状态本地 UDP 端口默认 `9873`，
关节高频端口 `7895`；检查网卡路由、防火墙与端口占用。
`connect()` 后等待 `isConnected()`，等待超时会清理连接。

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml inspect \
  --duration 10 --output outputs/sim2real/boot-01.jsonl
python sim2real/run.py analyze outputs/sim2real/boot-01.jsonl
```

录制包含原始 SDK 关节顺序、IMU、序列标记/时间戳、逐组年龄、机器人身份和配置。
文件以排他方式创建，重复文件名会报错。
JSONL 与分析中的 `observed_hz` 是轮询观察到的缓存 token 变化率，不能替代抓包测量实际 UDP 频率。
逐组年龄从主机首次看到 token 变化开始计算，不是机器人端的绝对采样时间。

### 3. 上电重复性与 IMU

每次人工上电后，将机器人放在相同可复现姿态，等待稳定，再录制一份文件。
脚本不远程上电，也不执行标零。

```bash
python sim2real/run.py compare-startup \
  outputs/sim2real/boot-01.jsonl outputs/sim2real/boot-02.jsonl \
  --output outputs/sim2real/startup-comparison.json
```

输出各通道平均值、标准差和跨次录制的均值极差。检查整个录制是否静止；
比较不同物理姿态的读数不能用于判断上电零位稳定性。
上电姿态不是机械零位。确认 SN/机型/固件后填写 calibration.identity。
IMU 需通过已知水平、倾斜和旋转姿态确定四元数代表 sensor→world 还是 body→world，
填写 sensor_to_body、gyro_bias 后再设 `imu.verified=true`。
框架使用姿态四元数计算归一化重力方向，不直接把加速度计读数当重力。

### 4. 被动关节标定

在厂商允许人工移动关节的条件下，每次只改变一个关节，将摆动前后两个静止姿态分别录制。
模型角变化必须来自已知参考姿态或量具，不能凭旋转印象猜测角度。
例如要核对 FR roll 通道：

```bash
python sim2real/run.py calibrate \
  --before outputs/sim2real/fr-roll-before.jsonl \
  --after outputs/sim2real/fr-roll-after.jsonl \
  --joint fr_hip_roll_joint --model-delta 0.1 \
  --output outputs/sim2real/fr-roll-evidence.json
```

这里的 `0.1` 仅示意一个已知参考变化值，不是运动命令。
若 before 姿态中的模型角已知，可以加 `--reference-angle` 生成 offset 建议。
报告输出候选 SDK 通道、sign、offset、其他通道同时变化/姿态不稳定等问题；
不会自动覆盖 TOML，也不会把建议设为已确认。

人工复核后填写对应行的 sdk_index/sign/offset 与确认项。
工作限位要依据官方可用范围和测量确定；脚本不自动扫描机械极限。

```text
q_model = sign * q_sdk[sdk_index] + offset
dq_model = sign * dq_sdk[sdk_index]
q_sdk_target[sdk_index] = sign * (q_model_target - offset)
```

当前阶段可生成点动计划检查转换，但仍不发送指令：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml jog \
  --joint fr_hip_roll_joint --delta 0.01
```

主动点动是独立入口。需要显式 `--live --arm --output`、被测关节的完整被动标定、
本体支撑确认、实测点动参数和固件保护确认，才会创建 JointLevelControl。
点动时其他关节 kp/kd 都为零，不会自动维持站姿。
通过一个往返目标核对方向后，再检查其他关节；标定报告仍需人工复核。
当前执行 1～4 步时，保留 `allow_motion=false`，无需执行主动点动。

## 参数确认后：离线回放与实机部署

模板默认 H=10、每帧 45 维、450 维 Actor、50 Hz，不依赖基座线速度。
历史布局为“每个观测项内旧到新”，首帧填满，站立历史每次进入重新建立，
行走历史持续更新。默认模型 Walk 和 Stand 都必须是 `[1,450]→[1,12]`，
本实机入口不自动替换缺少的专家模型，也不把 Critic 的 480 维输入用于推理。

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml replay \
  outputs/sim2real/boot-01.jsonl --output outputs/sim2real/replay-01.jsonl
```

离线按 50 Hz 选择最近的过去状态，不读取未来帧；保持旧帧时增加数据年龄。
首次连接后不完整的启动样本被跳过，数量写入输出 metadata；中途反馈丢失仍会报错。
回放输出原始 action、观测、模型/SDK 目标和配置。未标定时回放只能检查软件连通性。
回放中的新策略目标不会改变已录制轨迹，所以不构成闭环运动性能验证。
`--enforce-limits` 在所有数值限位、速度、跟踪误差和固件限矩已填写后，
增加与实机相同的目标保护计算；前提仍是录制数据充分、身份一致。

部署前填写所有 `check` 列出的项，尤其：

- 每个关节的标定与工作限位，IMU 安装方向、固件身份。
- 实测 send_rate_hz、固件看门狗与硬限矩、确认可用的 kp/kd。
- target_speed_rad_s、joint_speed_rad_s、tracking_error_rad、takeover_position_error_rad。
- 站立接管与退出策略实测；本实现只支持经过实测的 `zero_joint_command` 退出方案。

退出清零可能失去支撑力；部署前必须确认这一行为适用于现场。
TCP 断开时主机无法发停止命令；故障处理依赖已确认的机器人端看门狗和物理急停。
软件估算 PD 力矩不能代替固件的硬限矩，也不能保证 SDK 发包与电机执行完成。

`run` 默认仅打印就绪项；以下是后续已完成标定时的运动入口示意：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml run
# 后续部署阶段才执行，需 allow_motion=true 与所有前提确认：
python sim2real/run.py --config sim2real/config/local.runtime.toml run \
  --live --arm --duration 10 --vx 0 \
  --output outputs/sim2real/stand-live-01.jsonl
```

创建 SDK 前加载模型，建立高频关节会话后核对 SN/型号/固件/SDK，
至少观察两帧高频反馈更新，检查初始姿态处于标定的站立接管范围。
第一条完整关节命令保持实测关节角，逐步接入策略。
每条发送目标经过工作限位与变化率限制；下一次观测使用最终成功发送的目标对应 action。
PD 在机器人端执行，目标速度与前馈力矩均为零，避免重复施加本地 PD 力矩。
推理按 50 Hz、发包按已确认发送频率运行，两者独立调度。
反馈过期、会话变化、身份错误、实际速度/跟踪误差/倾角超限或循环超期均终止本次控制。
TCP 自动重连不会自动恢复策略；重新检查并显式启动新的运行。

## 离线测试与验证范围

```bash
python -m unittest discover -s sim2real/tests -p 'test_*.py' -v
python -m unittest discover -s sim2sim -p 'test_*.py' -v
```

实机测试使用模拟 SDK、模拟时钟和注入策略，不连接任何机器人。
软件测试验证接口与故障处理，不证明固件行为、动力学、停止可靠性或步态稳定性。
当前实现未在真机上验证。实际测量前不能通过将所有确认项设为 true 来替代验证。

## 参考

- 本地官方 SDK：[README.zh-CN.md](../bpx_sdk_open-master/README.zh-CN.md)、[关节示例](../bpx_sdk_open-master/example/joint_level_control_example.py)。
- 模型契约：[历史任务说明](../ROUGH_HISTORY_TRAINING.md)、[关节顺序指南](../sim2sim/JOINT_ORDER_MAPPING_GUIDE.md)。
- 官方 SDK：https://github.com/mirrormerobotics/bpx_sdk_open
