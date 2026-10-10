# BPX sim2real 操作手册

本文对应当前项目中的 `sim2real/run.py`，按“环境准备 → 只读采集 → 被动标定 → 离线回放 → 实机接管”组织操作。当前计划只执行步骤 1～4；后续运动操作单独列在第 8 节。

所有命令默认在项目根目录 `/home/hanguang/BPX_test` 执行。手册中的角度、角速度分别使用 rad、rad/s，时间使用 s；`--vx`、`--vy` 为机体系速度指令 m/s，`--wz` 为绕机体 z 轴的角速度指令 rad/s。

## 1. 先了解入口与当前阶段

### 1.1 命令会做什么

| 命令 | 连接机器人 | 发送关节运动命令 | 用途 |
|---|---|---|---|
| `check` | 否 | 否 | 检查配置、依赖、模型文件与部署缺项 |
| `inspect` | 是，`RequestRobotState` | 否 | 采集机器人原始反馈 |
| `analyze` | 否 | 否 | 分析已录制文件 |
| `compare-startup` | 否 | 否 | 比较相同物理姿态下多次上电读数 |
| `calibrate` | 否 | 否 | 从人工摆动前后的录制生成标定建议 |
| `jog`，不加 `--live` | 否 | 否 | 打印单关节点动计划与缺项 |
| `replay` | 否 | 否 | 对录制数据运行策略推理 |
| `run`，不加 `--live` | 否 | 否 | 打印实机部署就绪检查 |
| `jog --live --arm` | 是，`JointLevelControl` | 是 | 后续阶段的单关节主动验证 |
| `run --live --arm` | 是，`JointLevelControl` | 是 | 后续阶段的策略接管 |

`inspect` 的“只读”指不发送运动、关节目标或编码器标零命令。官方 SDK 在 `connect()` 时可能自动同步机器人系统时间，因此它仍有 SDK 连接本身带来的系统行为。

当前阶段保持：

```toml
[safety]
allow_motion = false
```

不执行任何带 `--live` 的命令。默认配置中的 `sign=1`、`offset=0`、单位 IMU 旋转矩阵、`kp=40`、`kd=1` 均不能视为已经完成实机确认。

### 1.2 文件分工

| 文件/目录 | 使用方法 |
|---|---|
| [README.md](README.md) | 查看框架结构与能力范围 |
| [HARDWARE_SETUP.md](HARDWARE_SETUP.md) | 记录厂商答复、现场测量与证据路径 |
| [config/runtime.toml](config/runtime.toml) | 默认运行配置模板 |
| [config/joint_calibration.template.toml](config/joint_calibration.template.toml) | 默认关节、IMU、身份标定模板 |
| `config/local.runtime.toml` | 自己的运行配置，首次使用时复制生成 |
| `config/local.calibration.toml` | 自己的实物标定结果，首次使用时复制生成 |
| `outputs/sim2real/` | 本文示例中的原始日志、报告和回放输出 |

`local.*.toml` 和 `outputs/` 下的日志被 Git 忽略。阶段结束后需要自行备份配置、报告、原始录制以及测量照片/记录，并按机器人 SN 区分。

## 2. 环境准备与本机配置

### 2.1 确认 Python 与架构

框架要求 Python 3.11 或更新版本。SDK 是本地二进制扩展，wheel 必须同时匹配 Python 版本、操作系统与 CPU 架构。

```bash
cd /home/hanguang/BPX_test
python --version
python -c 'import sys, platform; print(sys.executable); print(platform.system(), platform.machine())'
```

如果使用 Conda，先激活自己的环境，再执行下面的安装与采集命令。编辑器选中的解释器、终端的 `python` 和安装 SDK 时使用的解释器应一致。

也可以新建独立环境，例如：

```bash
conda create -n bpx-real python=3.11 -y
conda activate bpx-real
```

后续示例统一使用 `python -m pip`，让安装目标与执行命令的 Python 一致。

### 2.2 安装框架与官方 SDK

先安装框架及 NumPy：

```bash
python -m pip install -e .
```

从本地官方 SDK wheelhouse 安装 1.0.9。这个版本来自当前本地 SDK，是否适配你的机器人固件仍需确认：

```bash
python -m pip install --no-index \
  --find-links bpx_sdk_open-master/wheelhouse bpx-sdk-open==1.0.9
```

安装完成后进行不连接机器人的导入检查：

```bash
python -c 'import numpy, bpx_sdk; print("NumPy:", numpy.__version__); print("SDK:", getattr(bpx_sdk, "__version__", "unknown"))'
```

需要离线策略回放时，再安装 ONNX Runtime：

```bash
python -m pip install -e '.[inference]'
python -c 'import onnxruntime; print(onnxruntime.__version__)'
```

安装失败时先核对 wheel 文件名中的 `cp311` 等 Python 标记和 `x86_64`/`aarch64` 架构，不要通过重命名 wheel 绕过兼容性检查。本框架的当前 CLI 使用 ONNX 模型；前 1～4 步不需要安装 Isaac Lab、MuJoCo 或 Torch。

### 2.3 首次复制本机配置

以下操作只在本机配置文件尚不存在时执行，避免覆盖已经测量的结果：

```bash
cp -n sim2real/config/runtime.toml sim2real/config/local.runtime.toml
cp -n sim2real/config/joint_calibration.template.toml sim2real/config/local.calibration.toml
mkdir -p outputs/sim2real
```

编辑 `sim2real/config/local.runtime.toml` 顶部：

```toml
calibration_file = "local.calibration.toml"
```

检查 `[network]` 中机器人 IP 是否与现场一致，并保持 `allow_motion=false`。

配置文件中的 `calibration_file`、`walk_model`、`stand_model` 相对路径均相对于运行配置文件所在目录。把配置移动到其他目录时，需要同步调整这些路径。命令行的录制路径、`--output` 路径则相对于当前终端目录。

### 2.4 检查入口

```bash
python sim2real/run.py --help
python sim2real/run.py --config sim2real/config/local.runtime.toml check
```

`--config` 必须写在子命令前，例如 `--config ... inspect`。后文所有命令都显式指定本机配置，避免无意间使用模板。

安装本项目后也可以使用 `bpx-real`，其参数与 `python sim2real/run.py` 一致。本文继续使用源码入口，便于核对文件。

## 3. 步骤 1：核对本体、固件与控制条件

### 3.1 先记录事实

在 [HARDWARE_SETUP.md](HARDWARE_SETUP.md) 记录至少以下内容：

1. 正式型号、整机 SN、运动主机软件版本、电机/运动控制器固件版本。
2. SDK 与固件的兼容组合；开发模式如何进入、如何退出。
3. SDK、遥控器、App、急停之间的控制优先级与控制权切换流程。
4. `JointLevelControl` 要求的完整关节命令频率，以及机器人端 PD 执行频率。
5. 角度、角速度、力矩、kp/kd 的单位与有效范围。
6. 固件硬限矩、硬限速、命令超时阈值，以及失联后的本体行为。
7. `setZeroJointCommand()` 的实际效果与停止/退出流程。

没有答复或没有测量的项记录为“待确认”，并保留对应 `*_verified=false`。SDK 文档顺序、仿真增益、仿真 30 Nm 限矩不能替代实机测量。

### 3.2 运行离线检查

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml check
```

输出是 JSON，主要字段如下：

| 字段 | 含义与处理方法 |
|---|---|
| `config` / `calibration` | 确认加载的是自己的两个本机文件 |
| `dependencies` | 当前解释器是否找到 NumPy、ONNX Runtime 和 SDK |
| `models.*.exists` | 模型路径是否存在；前 1～4 步不依赖模型推理 |
| `models.*.sha256` | 记录模型文件指纹，后续确认使用了哪份导出模型 |
| `observe_blockers` | SDK 不可用时，需要先解决安装问题才能采集 |
| `replay_blockers` | ONNX Runtime 依赖提示；还需自行核对模型存在与维度 |
| `deploy_blockers` | 当前禁止部署的原因；前 1～4 步存在这些缺项是预期结果 |
| `mapping` | 策略索引、候选 SDK 索引、方向、偏移和确认状态 |

`check` 执行成功只表示检查过程完成，不能据退出码 0 判断可以部署。部署是否就绪需要阅读 `deploy_blockers`。

## 4. 步骤 2：网络与只读状态采集

### 4.1 连接前检查

本地模板使用：

| 配置项 | 模板值 | 含义 |
|---|---|---|
| `network.robot_ip` | `10.21.20.1` | SDK 连接的机器人地址，按现场网络修改 |
| `network.state_port` | `9873` | 本机接收普通状态的 UDP 端口 |
| `network.joint_state_port` | `7895` | 后续关节控制高频反馈 UDP 端口 |
| `network.tcp_local_port` | `0` | 本机 TCP 端口由系统分配 |
| `network.state_rate_hz` | `100` | 向 SDK 请求的普通状态上传频率 |
| `network.connect_timeout_s` | `5.0` | 等待 TCP 建立的超时时间 |

Linux 上可以先查看网络状态：

```bash
ip -br addr
ip route get 10.21.20.1
ping -c 3 10.21.20.1
ss -lunp
```

如果修改了机器人 IP，以上命令也相应修改。电脑 IP、子网掩码、网卡选择按厂商网络说明配置，不要盲目覆盖已有网卡设置。

检查实际路由是否走向连接机器人的网卡、UDP 端口是否被其他 SDK 示例占用，以及本机防火墙是否允许反馈包。ping 成功只能说明部分网络通路可达，不能证明 SDK TCP/UDP 通信正常。

### 4.2 录制第一份状态

保持本体处于厂商允许读取状态的稳定状态，运行：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml inspect \
  --duration 10 --poll-hz 100 \
  --output outputs/sim2real/boot-01.jsonl
```

运行期间约每秒打印一次连接状态、机器人身份、SDK 原始关节角、joint/IMU 数据年龄；结束时打印汇总报告。SDK 连接启动后还会等待 `isConnected()` 成立，超时会断开并报错。

`--duration` 是连接建立后的采集时长。`--poll-hz` 是电脑读取 SDK 缓存的频率，和请求给机器人的 `state_rate_hz` 是两个设置。

所有 JSON/JSONL 输出以排他方式创建，不覆盖已有文件。再次采集使用 `boot-02.jsonl` 等新文件名。

### 4.3 分析录制

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml analyze \
  outputs/sim2real/boot-01.jsonl \
  --output outputs/sim2real/boot-01-analysis.json
```

重点阅读：

| 字段 | 如何判断 |
|---|---|
| `samples` / `duration_s` | 是否确实录到了预期时长和足够样本 |
| `disconnected_samples` | 是否出现断连，需要进一步检查网络 |
| `identities` | SN、型号、版本是否稳定且与实物一致；该字段中的身份以 JSON 字符串表示 |
| `groups.joint.observed_hz` | 轮询时看到关节更新 token 变化的频率 |
| `groups.imu.observed_hz` | 轮询时看到 IMU 更新 token 变化的频率 |
| `groups.*.max_age_s` | 缓存长期不更新时年龄会增加；需检查是否有卡顿 |
| `sdk_q_mean/std/min/max` | SDK 顺序下每个关节的角度统计 |

`observed_hz` 不是抓包得到的真实 UDP 发包率；多个包可能发生在两次轮询之间。年龄是“主机首次看到 token 变化以后过去的时间”，不是机器人采样到主机的端到端延迟。刚连接时首先看到的缓存也不能仅凭年龄为零证明它是新采样。

TCP 连通但 joint/IMU 没有更新时，优先核查状态上传配置、UDP 回程、端口占用与 SDK/固件兼容性。不要仅增加超时阈值来掩盖缺失反馈。

### 4.4 原始日志的内容

JSONL 每行是一条 JSON，通常依次包含：

- `metadata`：schema、配置快照、标定快照和采集说明。
- `sample`：主机单调时钟、会话编号、连接状态、SDK 原始关节位置/速度/力矩、IMU、里程计、更新 token、数据年龄、遥测和身份。
- `end`：`completed` 或 `error`，异常时附带错误信息。

连接初期某些反馈字段可能为 `null`，需要等待 UDP 数据到齐。分析工具仍可统计这些录制；后续回放会跳过不完整的启动前缀，中途丢失反馈则会拒绝继续。

`host_time` 是单调时钟数值，用于计算间隔，不能直接转换为日期。机器人时间戳先作为更新标记保存，不直接当作主机墙上时间使用。

## 5. 步骤 3：上电初始位置重复性与 IMU

### 5.1 多次上电重复性

此项检验“同一物理姿态下，重新上电后的读数是否一致”，不用于把上电姿态定义成机械零位。

每次按以下顺序操作：

1. 按厂商允许的方法完成上电与初始化。
2. 将机器人放到相同、可以复现的参考姿态；记录定位方法或拍照。
3. 等待关节与本体静止，只采集稳定区间。
4. 运行 `inspect`，使用新的文件名保存。
5. 先用 `analyze` 检查各通道的 `sdk_q_std`，排除姿态变化影响。

例如第二、第三次上电：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml inspect \
  --duration 10 --output outputs/sim2real/boot-02.jsonl
python sim2real/run.py --config sim2real/config/local.runtime.toml inspect \
  --duration 10 --output outputs/sim2real/boot-03.jsonl
```

每条命令之间需要人工完成对应的重新上电和参考姿态复现，连续执行两次采集本身不等于两次上电。

比较三份录制：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml compare-startup \
  outputs/sim2real/boot-01.jsonl \
  outputs/sim2real/boot-02.jsonl \
  outputs/sim2real/boot-03.jsonl \
  --output outputs/sim2real/startup-comparison.json
```

`runs` 包含各次统计；`sdk_startup_range_rad` 是每个 SDK 通道的跨次平均读数极差，即最大均值减最小均值。

脚本没有替你制定合格公差，也不会自动判断初始化方式。公差应结合量具误差、编码器分辨率和策略需要确定。脚本也不会自动排除不同 SN 的录制；对比前必须人工确认身份一致。

若读数不一致，需要先区分参考姿态没有复现、角度跨周期跳变、初始化过程未结束、编码器/固件零位发生变化等情况，再确定处理方式。

### 5.2 填写机器人身份

从只读录制核对身份后，在 `local.calibration.toml` 填写：

```toml
[identity]
serial_number = "实际录制中的 SN"
robot_model = "实际录制中的 robot_model"
firmware_version = "实际录制中的 firmware_version"
```

上面的字符串是说明性占位文本，需要替换。当前适配器使用 `BPX`、`BPXPro`、`BPW` 等型号字符串；实机控制按字符串精确匹配，不能自行把 `BPXPro` 改成营销名称 `BPX-Pro`。

`firmware_version` 对应适配器从 `getRobotVersion()` 前三个分量拼出的版本字符串；它不能替代所有电机固件版本记录。SDK 版本单独配置在 `runtime.firmware.expected_sdk_version`。

### 5.3 采集 IMU 参考姿态

在不发送运动命令的条件下，按可测量、可复现的姿态采集：

- 机体水平静止。
- 已知正/负 roll 倾角。
- 已知正/负 pitch 倾角。
- 围绕已知轴、按已知正方向缓慢旋转，用于核对陀螺仪轴和正负号。

例如水平静止：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml inspect \
  --duration 10 --output outputs/sim2real/imu-level-01.jsonl
```

SDK 四元数按 `xyzw` 读取，框架转换为 `wxyz` 进行旋转计算。需要确认四元数表示 `sensor_to_world` 还是 `body_to_world`，并确定传感器坐标轴到训练模型机体系的旋转。

在 `local.calibration.toml` 中填写：

| 字段 | 含义 |
|---|---|
| `imu.sensor_to_body` | 9 个数，行优先旋转矩阵；`v_body = R @ v_sensor` |
| `imu.gyro_bias` | 静止时原始传感器坐标系中的陀螺仪偏置，rad/s |
| `imu.quaternion_frame` | `sensor_to_world` 或 `body_to_world` |
| `imu.verified` | 完成已知姿态和旋转验证后再设为 `true` |

旋转矩阵必须满足正交性和行列式 +1，不能只给某个轴乘负号而产生镜像矩阵。模板单位矩阵仅是占位。当前没有一条命令能够自动辨识 IMU 安装方向和四元数含义。

### 5.4 用录制检查当前 IMU 转换

下面的离线代码读取水平静止录制，打印原始陀螺仪均值，以及按照当前标定转换后的重力方向和角速度均值，不连接机器人：

```bash
python - <<'PY'
import sys
sys.path.insert(0, "sim2real")
import numpy as np
from bpx_sim2real.config import load_config
from bpx_sim2real.calibration import Calibration
from bpx_sim2real.recording import load_samples

config = load_config("sim2real/config/local.runtime.toml")
cal = Calibration(config)
_, samples = load_samples("outputs/sim2real/imu-level-01.jsonl")
valid = [s for s in samples if s.connected and s.quat_xyzw is not None and s.omega is not None]
if not valid:
    raise SystemExit("没有完整 IMU 样本")
converted = [cal.imu(s.quat_xyzw, s.omega) for s in valid]
gravity = np.asarray([v[0] for v in converted])
angular = np.asarray([v[1] for v in converted])
print("raw gyro mean:", np.mean([s.omega for s in valid], axis=0))
print("body gravity mean/std:", gravity.mean(axis=0), gravity.std(axis=0))
print("body gyro mean/std:", angular.mean(axis=0), angular.std(axis=0))
PY
```

模型机体水平且坐标约定匹配时，归一化重力方向应接近 `[0, 0, -1]`，静止角速度应接近零。还要使用倾斜和旋转记录核对轴与符号；单独通过水平检查不能确认全部方向。

重力方向由姿态四元数计算，不直接把加速度计瞬时值作为重力。上述代码仅计算统计值，录制的新鲜度、姿态静止程度仍需结合 `analyze` 与现场记录检查。

## 6. 步骤 4：逐关节被动标定

### 6.1 明确两套关节顺序

策略采用“按关节类型分组”，SDK 文档采用“按腿分组”。下表只是模板映射建议，必须与自己的实物核对：

| 策略索引 | 模型关节名 | 候选 SDK 索引 |
|---|---|---|
| 0 | `fl_hip_roll_joint` | 0 |
| 1 | `fr_hip_roll_joint` | 3 |
| 2 | `hl_hip_roll_joint` | 6 |
| 3 | `hr_hip_roll_joint` | 9 |
| 4 | `fl_hip_pitch_joint` | 1 |
| 5 | `fr_hip_pitch_joint` | 4 |
| 6 | `hl_hip_pitch_joint` | 7 |
| 7 | `hr_hip_pitch_joint` | 10 |
| 8 | `fl_knee_joint` | 2 |
| 9 | `fr_knee_joint` | 5 |
| 10 | `hl_knee_joint` | 8 |
| 11 | `hr_knee_joint` | 11 |

FL/FR/HL/HR 分别为前左/前右/后左/后右，以机器人本体方向定义。模型的角度正方向要结合模型关节轴及参考姿态核对，不能只用“从屏幕看顺时针”定义。

### 6.2 每个关节要确认的内容

| 参数 | 要回答的问题 |
|---|---|
| `sdk_index` | 人工移动这个物理关节时，哪个 SDK 通道发生变化？ |
| `sign` | 模型角增加时，SDK 角增加还是减少？只能是 +1 或 -1 |
| `offset` | SDK 角怎样换算成模型角？需要已知绝对参考姿态 |
| `minimum` / `maximum` | 模型坐标系下允许的工作范围是什么？ |
| 四个 `*_verified` | 是否分别有通道、方向、零位和工作限位的证据？ |

换算公式：

```text
q_model = sign * q_sdk[sdk_index] + offset
dq_model = sign * dq_sdk[sdk_index]
offset = q_model_reference - sign * q_sdk_reference
q_sdk_target[sdk_index] = sign * (q_model_target - offset)
```

kp/kd 只重新排列通道，不乘 `sign`，也不加 `offset`。

### 6.3 单关节采集流程

先确认厂商允许当前状态下人工移动关节，并使用适合的本体支撑方式。SDK 只读不会改变本体当前已有控制状态；若关节有保持力矩，不要强行移动。

以 `fr_hip_roll_joint` 为例：

1. 将被测关节放在已知 before 姿态，保持其他关节不变。
2. 用量具或可复现定位确定 before 的模型角；仅核对方向时也至少要知道准确角变化。
3. 静止后采集 before 文件。
4. 人工将该关节移动到已知 after 姿态，记录模型角变化 `after - before`。
5. 再次静止后采集 after 文件，不把移动过程混入这两个稳定记录。

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml inspect \
  --duration 5 --output outputs/sim2real/fr-roll-before.jsonl
```

完成人工移动并稳定后，再执行：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml inspect \
  --duration 5 --output outputs/sim2real/fr-roll-after.jsonl
```

### 6.4 生成通道与方向建议

如果测得模型角增加 `0.1 rad`，运行：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml calibrate \
  --before outputs/sim2real/fr-roll-before.jsonl \
  --after outputs/sim2real/fr-roll-after.jsonl \
  --joint fr_hip_roll_joint --model-delta 0.1 \
  --output outputs/sim2real/fr-roll-evidence.json
```

`0.1` 是已测得的参考变化示例，不是让机器人运动的命令，也不是建议的固定标定幅度。若实际变化是负值，应填真实负值。

若 before 的绝对模型角也已知，例如确实为 `0 rad`，可同时生成 offset 建议：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml calibrate \
  --before outputs/sim2real/fr-roll-before.jsonl \
  --after outputs/sim2real/fr-roll-after.jsonl \
  --joint fr_hip_roll_joint --model-delta 0.1 --reference-angle 0.0 \
  --output outputs/sim2real/fr-roll-evidence-with-offset.json
```

没有可靠绝对参考角时省略 `--reference-angle`，结果中的 `suggested_offset` 为 `null`。方向确认不能自动推导绝对零位。

### 6.5 阅读建议与警告

| 输出字段 | 含义 |
|---|---|
| `suggested_sdk_index` | 前后均值变化绝对值最大的 SDK 通道 |
| `suggested_sign` | 根据该通道变化与已知模型变化的符号推算方向 |
| `suggested_offset` | 指定绝对参考角后计算的偏移；否则为空 |
| `sdk_delta` | 所有 12 个 SDK 通道的 after 均值减 before 均值 |
| `before_sdk_q` / `after_sdk_q` | 两份稳定录制的原始 SDK 平均角 |
| `warnings` | 姿态不稳定、多关节变化、单位/角变化不匹配、缺少 SN 等提示 |
| `verified` | 固定为 `false`，保留人工复核要求 |

当前脚本会在录制标准差超过 0.01 rad、其他通道变化超过主通道变化的 25% 等情况下提示警告。这些是筛查规则，不是机器人标定的正式验收标准；无警告也不代表测量必然正确。

存在警告时先检查采集与参考姿态，必要时重新采集。为了获得更可靠的结果，可以使用另一组已知姿态重复测量，并核对正、负方向一致性。

### 6.6 人工写回标定配置

报告不会自动修改 TOML。人工复核后，编辑对应 `[[joints]]` 段，例如：

```toml
[[joints]]
name = "fr_hip_roll_joint"
sdk_index = 3
sign = 1
offset = 0.0
mapping_verified = true
direction_verified = true
offset_verified = false
limits_verified = false
```

这个例子仅说明填写格式；其中索引、方向、偏移必须替换为自己的测量结果。未确认绝对零位时保留 `offset_verified=false`。全部 12 个 `sdk_index` 必须构成 0～11 的一一映射，不能重复。

工作限位依据厂商允许范围、实际机构与使用场景确定，使用模型坐标系 rad，并留出合理余量。在对应关节段中同时填写 `minimum`、`maximum`，完成验证后再设置 `limits_verified=true`。当前工具不自动扫描机械极限，不通过撞限位确定范围。

不要调用固件的编码器标零接口来替代模型坐标换算；本框架没有自动执行 `setZeroPositionsFlag`。

每次修改配置后运行：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml check
```

### 6.7 只生成点动计划

当前阶段可以离线检查一个假设的正方向点动会转换成什么 SDK 通道与符号：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml jog \
  --joint fr_hip_roll_joint --delta 0.01
```

输出包含 `mode=plan`、`sdk_index`、`delta_model_rad`、`delta_sdk_rad` 和主动点动缺项。此命令不创建 SDK 对象，不发送运动命令，不检查现场真实姿态；计划生成成功不代表可以执行。

### 6.8 步骤 1～4 完成后的交付物

至少保留以下内容：

- 填写了事实与待确认项的 `HARDWARE_SETUP.md`。
- 可重复使用的网络配置与 SDK 安装记录。
- 相同姿态的多次上电日志和比较报告。
- 水平、倾斜、旋转姿态的 IMU 日志与坐标核对记录。
- 12 个关节的 before/after 录制、参考角测量记录和标定报告。
- 按 SN 保存的本机运行配置与标定配置。

你可以在完成这些工作后停在这里，继续保持 `allow_motion=false`。尚未实测的增益、关节控制频率、看门狗、站立接管与停止行为等部署项保持未确认。

## 7. 后续：离线策略回放

这一节仍不连接机器人，可在具备模型和完整录制后执行。

### 7.1 核对模型契约

当前模板使用 Walk 与 Stand 两个 ONNX 模型，二者均应为 `[1,450] → [1,12]`：

| 项目 | 当前模型约定 |
|---|---|
| Actor 单帧 | 45 维：角速度 3、重力方向 3、command 3、相对关节位置 12、关节速度 12、上一 action 12 |
| 历史长度 | 10，每个观测项内部按旧到新展开，首帧填满历史 |
| 策略更新频率 | 50 Hz |
| 默认关节角 | roll 四个为 0；pitch 四个为 0.7；knee 四个为 -1.4 rad |
| action scale | 0.5 |
| 目标角 | `default_joint_position + 0.5 * action`，之后经过接管/混合与保护处理 |

默认 Actor 不需要基座线速度；训练 Critic 的 480 维输入不用于此处推理。若使用其他模型，需要同时核对训练契约与代码支持，不能只修改输入维度。

在 `local.runtime.toml` 核对 `walk_model` 和 `stand_model` 路径；两份专家模型都需要存在且维度正确。调用 `check` 查看路径与哈希，再运行回放检查实际 ONNX 加载。

### 7.2 零速度指令回放

优先使用接近参考站姿、姿态和关节反馈完整的录制：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml replay \
  outputs/sim2real/boot-01.jsonl \
  --vx 0 --vy 0 --wz 0 \
  --output outputs/sim2real/replay-stand-01.jsonl
```

脚本按 50 Hz 选择最近的过去状态，不提前读取未来状态；保持旧状态时会增加年龄。初始不完整样本被跳过，跳过数量记录在 metadata。中途断连、会话变化、反馈过期、输入异常等会终止回放。

未完成标定时，回放只能检查软件数据流；不能把模型输出作为可靠实机目标。躺卧或过度倾斜的日志可能触发倾角保护，不适合作为站立策略接管验证数据。

### 7.3 指令与保护计算

为观察 Walk 模式下的离线输出，可以另外生成带常量 command 的回放，例如：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml replay \
  outputs/sim2real/boot-01.jsonl --vx 0.2 --vy 0 --wz 0 \
  --output outputs/sim2real/replay-command-01.jsonl
```

`0.2` 仅用于离线 command 示例。是否进入 Walk 还受 command 限斜率、阈值、延时和录制长度影响。

当全部关节工作限位，以及目标速度、实际速度、跟踪误差和固件限矩数组已填写后，可增加：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml replay \
  outputs/sim2real/boot-01.jsonl --enforce-limits \
  --output outputs/sim2real/replay-limits-01.jsonl
```

此选项增加目标保护计算，但不执行完整的实机就绪/身份匹配认证。尤其是在静止录制上回放移动目标时，录制关节不会跟随新目标，跟踪误差检查可能因此终止。

输出 `policy` 行包括观测、action、模式、模型目标、SDK 目标和录制状态；保护启用时还包含估算力矩与是否约束目标。回放输出不同于 `inspect` 的 `sample` 录制，不能直接拿回放输出再次输入 `analyze` 或 `replay`。

离线回放是记录轨迹上的推理；新目标不会改变录制状态，因此不是闭环动力学验证，也不能证明步态稳定。

## 8. 参数与现场条件确认后的运动操作

本节属于后续部署阶段。当前只执行 1～4 步时不执行本节命令。

### 8.1 需要补齐的参数

| 配置位置 | 内容与单位 |
|---|---|
| `firmware.*_verified` | 关节控制、看门狗、退出行为、增益单位、固件限矩的验证记录 |
| `firmware.watchdog_timeout_s` | 实测机器人端命令超时，s；填写它不会替机器人配置看门狗 |
| `firmware.robot_torque_limit_nm` | 12 个实机固件限矩，Nm，采用策略顺序 |
| `firmware.shutdown_action` | 当前只支持经过确认的 `"zero_joint_command"` |
| `control.send_rate_hz` | 厂商要求且实测可用的关节发送频率，不低于策略 50 Hz |
| `control.kp/kd` | 12 个实测可用增益，采用策略顺序 |
| `control.gains_verified` | 记录增益验证完成 |
| `control.standing_handover_verified` | 站立控制权移交流程已验证 |
| `safety.target_speed_rad_s` | 12 个目标角变化率上限，rad/s |
| `safety.joint_speed_rad_s` | 12 个反馈实际角速度上限，rad/s |
| `safety.tracking_error_rad` | 12 个实际位置相对上一目标的误差上限，rad |
| `safety.takeover_position_error_rad` | 12 个接管角相对默认站姿允许误差，rad |
| `safety.max_state_age_s/max_group_skew_s` | 主机观察到的数据年龄和反馈组更新时间差阈值，s |
| `safety.max_cycle_lateness_s` | 循环超期阈值，s |
| `safety.max_tilt_rad` | 机体倾角终止阈值，rad |
| `jog.*` | 单关节点动所需增益、幅度、速度、跟踪误差与时长 |

配置中的数值限制不自动修改固件硬限矩、硬限速或看门狗。当前代码用填写的限矩比较 PD 估算值；它不证明电机实际受到了同样的固件限制。

全部确认项应对应真实测量或明确厂商依据。不要为了让 `blockers` 消失而批量设置为 `true`。

### 8.2 单关节主动验证

此入口需要被测关节已完成被动标定与工作限位确认，还需要身份、固件保护、现场支撑和点动参数就绪。其他关节的 kp/kd 都为零，不能依靠本入口保持整机站立。

只有已准备执行运动时，才将 `allow_motion` 设为 `true`，按实际条件填写 `jog` 配置。先使用默认 `jog` 查看缺项，再执行显式运动入口。下列 `0.01 rad` 是命令格式示例，实际幅度须来自自己的验证方案：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml jog \
  --joint fr_hip_roll_joint --delta 0.01 \
  --live --arm --output outputs/sim2real/fr-roll-jog-live-01.jsonl
```

目标前半程逐渐到达“实测起点 + delta”，后半程返回实测起点。时长至少满足往返距离/目标速度的要求。脚本检查端点工作范围、实际速度、跟踪误差和估算力矩；返回起点不等于已经验证了精度。

### 8.3 站立策略接管

先运行不连接机器人的预检查：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml run
```

`blockers` 应为空；同时确认 `check` 中 SDK、ONNX Runtime 与模型均可用。默认 `run` 不会验证当前机器人身份或姿态，实机连接后才会检查它们。

本实现从已验证站姿接管，没有自动起身功能。先按经过验证的厂商流程建立站姿与控制权，再启动零速度指令：

```bash
python sim2real/run.py --config sim2real/config/local.runtime.toml run \
  --live --arm --duration 10 --vx 0 --vy 0 --wz 0 \
  --output outputs/sim2real/stand-live-01.jsonl
```

运行流程为：加载模型 → 建立高频关节会话 → 核对身份与 SDK 版本 → 等待至少两次高频反馈更新 → 检查接管姿态 → 第一条目标保持实测角度 → 逐渐接入策略。

站立稳定、接管与退出已实测后，才考虑按自己的试验方案增加运动 command。

### 8.4 当前实机控制行为与边界

- 策略按 50 Hz 更新，完整关节命令按配置发送频率调度。
- 机器人端执行 PD，发送目标速度与前馈力矩均为零；主机不再次把 PD 估算力矩作为前馈发送。
- 目标经过工作限位与变化率限制，下一次 Actor 使用最终成功发送目标对应的 action。
- 反馈过期、断连、会话变化、身份不符、倾角/速度/跟踪误差/估算力矩超限、循环超期等会终止当前控制。
- 不补跑过期的历史帧；SDK 重新连接后也不自动恢复控制，需要检查后重新启动。
- CLI 的 `--vx/--vy/--wz` 是一次运行中的常量指令，当前没有键盘、手柄或在线变速命令接口。
- 控制器支持 Stand → Walk → Stopping → Stand；Stopping 平滑归零 command 后按时间切换，当前 CLI 没有运行中发出新停车 command 的交互入口。

`--duration` 到期或 Ctrl+C 会退出运行并尝试清零，不会自动先完成减速停车。`setZeroJointCommand()` 可能释放支撑力，因此现场停止方案与支撑条件必须已经确认。

如果 TCP 已断开，主机无法发送退出清零；如果清零返回失败，脚本会报错。现场故障处置依赖已验证的机器人端看门狗与物理急停，不能仅依赖 Python 进程。当前框架没有在真机上完成验证，自动起身、自动恢复及机载部署权限仍需后续适配。

## 9. 常见问题与处理

| 现象 | 处理方法 |
|---|---|
| `No module named bpx_sdk` | 核对当前解释器，使用同一 `python -m pip` 安装匹配 wheel |
| `No matching distribution found` | 核对 Python、OS、架构与 wheelhouse 支持范围 |
| 等待真实 TCP 连接超时 | 检查 IP、网卡路由、开发模式要求、SDK/固件组合与其他连接占用 |
| TCP 连通但反馈为 null/不更新 | 检查 UDP 回程、防火墙、上传端口和固件上传状态 |
| 输出文件已存在 | 更换文件名；脚本不覆盖原始证据 |
| TOML 解析/校验错误 | 检查键名、数组长度、布尔类型、限位成对填写和 SDK 索引唯一性 |
| `check` 有大量部署缺项 | 前 1～4 步正常；按证据逐项补齐，不急于解除运动门禁 |
| 多个关节同时变化警告 | 重新固定其他关节，只录制移动前后的静止区间 |
| 角变化与参考不一致 | 核对 rad/度、参考姿态、SDK 通道和角度跨周期问题 |
| 没有完整 Actor 状态 | 检查录制是否有位置、速度、四元数和角速度有效样本 |
| 回放模型路径不存在/维度不符 | 核对相对路径、两份专家模型以及训练观测契约 |
| 反馈过期/组时间差过大 | 查明反馈卡顿与轮询情况，不直接放宽阈值跳过问题 |
| 序列号回退/SDK 会话变化 | 停止本次运行，核查重启、回绕或重连后显式重新启动 |
| 初始姿态不符合站立接管范围 | 按已验证流程建立站姿；本入口不会自动起身 |
| 实际速度/跟踪误差/倾角超限 | 检查标定、增益、目标、姿态与模型适配，保留故障日志 |
| 退出清零失败 | 按现场停止方案处理，再核查固件与通信，不能假定已停止 |

详细参数帮助可直接查看，例如：

```bash
python sim2real/run.py inspect --help
python sim2real/run.py calibrate --help
python sim2real/run.py replay --help
python sim2real/run.py run --help
```

命令行通常以 0 表示操作完成，2 表示捕获到的配置/运行错误，130 表示 Ctrl+C 中断。仍需检查报告字段与日志结尾；成功退出码不等同于硬件验收通过。

## 10. 离线软件检查

以下测试使用模拟 SDK、时钟和策略，不连接机器人：

```bash
python -m unittest discover -s sim2real/tests -p 'test_*.py' -v
python -m unittest discover -s sim2sim -p 'test_*.py' -v
```

它们验证映射、历史与运行接口、门禁、目标处理及部分故障退出逻辑，不验证真实固件行为、延迟、动力学、停止可靠性和步态稳定性。

## 11. 相关本地资料

- [官方 SDK 中文说明](../bpx_sdk_open-master/README.zh-CN.md)
- [官方关节控制示例](../bpx_sdk_open-master/example/joint_level_control_example.py)
- [历史观测训练说明](../ROUGH_HISTORY_TRAINING.md)
- [sim2sim 关节顺序指南](../sim2sim/JOINT_ORDER_MAPPING_GUIDE.md)

当前阶段推荐顺序：准备环境与本机配置 → 记录硬件事实 → `check` → `inspect/analyze` → 多次上电比较 → IMU 核对 → 逐关节被动标定 → 备份证据与参数。后续部署以实测结果为依据逐项推进。
