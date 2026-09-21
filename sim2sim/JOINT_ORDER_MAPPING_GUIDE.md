# BPX 关节顺序映射检查与防错指南

本文档用于检查 BPX 策略在 Isaac Lab、导出模型和 MuJoCo 之间的关节顺序是否一致，并给出避免同类问题再次发生的配置方法。

## 1. 为什么关节顺序必须严格一致

策略的 12 维动作和观测张量只有数值，没有关节名称。第 `i` 个数究竟属于哪个关节，完全由训练和部署两端约定的顺序决定。

下面几处必须使用同一个顺序：

1. 策略输出 `action[0:12]`；
2. Isaac Lab 动作项解析出的关节；
3. Isaac Lab 观测中的 `joint_pos_rel[0:12]`；
4. Isaac Lab 观测中的 `joint_vel_rel[0:12]`；
5. 默认关节角 `q_default[0:12]`；
6. MuJoCo 中读取的关节位置和速度；
7. MuJoCo 中接收目标位置或力矩的执行器。

只要其中一项次序不同，张量维度仍然可能完全正确，程序也不会报错，但策略会把某个关节的动作发给另一个关节，并把另一个关节的状态当作当前关节状态。这类错误常表现为启动后剧烈乱动、立即倾倒或左右腿动作异常。

## 2. 当前 BPX 策略的标准顺序

当前训练模型在 Isaac Lab 中实际解析得到的是“关节类型优先”顺序，而不是“一条腿的三个关节排在一起”的顺序：

| 策略索引 | 关节名称 | 默认角度（rad） |
| ---: | --- | ---: |
| 0 | `fl_hip_roll_joint` | 0.0 |
| 1 | `fr_hip_roll_joint` | 0.0 |
| 2 | `hl_hip_roll_joint` | 0.0 |
| 3 | `hr_hip_roll_joint` | 0.0 |
| 4 | `fl_hip_pitch_joint` | 0.7 |
| 5 | `fr_hip_pitch_joint` | 0.7 |
| 6 | `hl_hip_pitch_joint` | 0.7 |
| 7 | `hr_hip_pitch_joint` | 0.7 |
| 8 | `fl_knee_joint` | -1.4 |
| 9 | `fr_knee_joint` | -1.4 |
| 10 | `hl_knee_joint` | -1.4 |
| 11 | `hr_knee_joint` | -1.4 |

该顺序已经写入 [`config/bpx_flat.toml`](config/bpx_flat.toml)。修正前，MuJoCo 配置错误地使用了 `FL roll/pitch/knee → FR roll/pitch/knee → ...` 的单腿优先顺序，因此同一维动作被发送给了错误关节。修正后，当前 ONNX 策略在 `vx=0.2 m/s` 下已能稳定运行 20 秒。

## 3. 每次部署时如何检查

### 3.1 查看 Isaac Lab 真正解析出的动作顺序

创建环境时，Isaac Lab 的 `JointPositionAction` 会输出类似下面的日志：

```text
Resolved joint names for the action term JointPositionAction: [...] [...]
```

应逐项记录列表中的关节名称，不能仅根据 URDF、USD 文件中的书写顺序或关节命名习惯推断。对于当前模型，该日志应与上一节的 12 项顺序完全一致。

当前 [`bpx_base_env_cfg.py`](../source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_base_env_cfg.py) 已让动作、关节位置观测和关节速度观测共同使用显式的 `BPX_POLICY_JOINT_NAMES`，并设置 `preserve_order=True`。新增策略时必须继续复用这份列表，不能改回 `joint_names=[".*"]`。

### 3.2 检查 MuJoCo 名称映射

在 `bpx-sim2sim` Conda 环境中运行：

```bash
cd /home/hanguang/BPX_test
conda activate bpx-sim2sim
python sim2sim/validate_setup.py
```

脚本会打印 MuJoCo 运行器使用的 `joint order`，并检查：

- 12 个关节名称是否都能在 MJCF 中找到；
- 12 个执行器是否能正确匹配；
- 观测和动作维度是否分别为 48 和 12；
- 默认角和零动作 PD 是否有效；
- 仿真中是否出现 NaN 或 Inf。

这里的打印结果必须与 Isaac Lab 的 `Resolved joint names` 逐行相同。`validate_setup.py` 单独通过只能证明 MuJoCo 内部映射有效，不能自动证明它与 Isaac Lab 相同。

### 3.3 做 one-hot 单关节动作测试

这是最可靠的语义检查。每次只令一个动作维度非零：

```text
action = [0, 0, ..., 0]
action[i] = 0.2
```

当前动作缩放为 `0.5`，所以第 `i` 个目标角应满足：

```text
q_target[i] = q_default[i] + 0.5 × 0.2
            = q_default[i] + 0.1 rad
```

从 `i=0` 到 `i=11` 逐个测试，并检查：

1. Isaac Lab 中变化的目标关节名称；
2. MuJoCo 中变化的目标关节名称；
3. 两边是否都等于标准顺序中第 `i` 项；
4. 正动作是否产生相同的关节旋转方向。

优先检查控制器内部的目标角或力矩，不要只凭画面判断实际运动。机器人存在惯性、接触和关节耦合，一个关节受控后其他关节也可能被动运动。

### 3.4 检查观测顺序

当前 48 维观测布局为：

```text
[0:3]    base_lin_vel_body
[3:6]    base_ang_vel_body
[6:9]    projected_gravity_body
[9:12]   command_vx_vy_wz
[12:24]  joint_pos_relative
[24:36]  joint_velocity
[36:48]  last_action
```

动作映射正确并不代表观测映射必然正确。应分别改变一个关节的位置或速度，确认：

- 第 `i` 个关节位置只对应 `observation[12 + i]`；
- 第 `i` 个关节速度只对应 `observation[24 + i]`；
- 上一步 `action[i]` 对应 `observation[36 + i]`。

可以使用以下命令导出 Isaac Lab 侧的实际观测：

```bash
conda activate isaaclab
python sim2sim/probe_isaaclab.py \
    --headless --device cpu --skip-policy --steps 100 \
    --json-output sim2sim/logs/isaac_zero_action.json
```

再生成 MuJoCo 日志并对拍：

```bash
conda activate bpx-sim2sim
python sim2sim/run_mujoco.py \
    --backend zero --duration 0.5 --no-realtime \
    --log sim2sim/logs/mujoco_zero_action.csv

python sim2sim/compare_trajectories.py \
    --isaac sim2sim/logs/isaac_zero_action.json \
    --mujoco sim2sim/logs/mujoco_zero_action.csv
```

轨迹不必逐帧完全相同，因为 PhysX 与 MuJoCo 的接触和积分器不同；但如果个别关节误差异常大、左右腿呈交换关系或某几列明显高度相关，应优先怀疑关节重排问题。

## 4. 从配置层避免再次出错

### 4.1 在训练配置中声明唯一的标准列表

建议在环境配置中定义一次标准顺序，并由动作项和两个关节观测项共同引用：

```python
BPX_POLICY_JOINT_NAMES = [
    "fl_hip_roll_joint",
    "fr_hip_roll_joint",
    "hl_hip_roll_joint",
    "hr_hip_roll_joint",
    "fl_hip_pitch_joint",
    "fr_hip_pitch_joint",
    "hl_hip_pitch_joint",
    "hr_hip_pitch_joint",
    "fl_knee_joint",
    "fr_knee_joint",
    "hl_knee_joint",
    "hr_knee_joint",
]
```

动作项显式保持列表顺序：

```python
joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=BPX_POLICY_JOINT_NAMES,
    preserve_order=True,
    scale=0.5,
    use_default_offset=True,
)
```

关节位置和速度观测也使用同一个 `SceneEntityCfg`：

```python
policy_joints = SceneEntityCfg(
    "robot",
    joint_names=BPX_POLICY_JOINT_NAMES,
    preserve_order=True,
)

joint_pos = ObsTerm(
    func=mdp.joint_pos_rel,
    params={"asset_cfg": policy_joints},
    noise=Unoise(n_min=-0.01, n_max=0.01),
)
joint_vel = ObsTerm(
    func=mdp.joint_vel_rel,
    params={"asset_cfg": policy_joints},
    noise=Unoise(n_min=-1.5, n_max=1.5),
)
```

`preserve_order=True` 的意义是：让解析结果遵循显式列表的顺序，而不是重新按机器人资产内部的关节索引排序。

上述代码是后续防错建议，当前训练配置尚未按此方式修改。若要修改现有环境，必须先确认显式列表与旧模型训练时的实际顺序完全一致；否则旧模型的输入、输出语义会被改变。

### 4.2 随模型导出元数据

ONNX 和 TorchScript 通常只保存张量形状，不会完整保存每一维的机器人语义。建议每次导出策略时同时生成 `policy_metadata.json`：

```json
{
  "observation_dimension": 48,
  "action_dimension": 12,
  "joint_names": ["fl_hip_roll_joint", "fr_hip_roll_joint"],
  "action_scale": 0.5,
  "default_joint_position": [0.0, 0.0],
  "physics_dt": 0.005,
  "decimation": 4
}
```

示例中数组被省略为两项；实际文件必须完整保存 12 个关节。部署程序启动时应自动比较元数据与 MuJoCo 配置，任何名称、顺序、维度或缩放不一致都立即报错，而不是继续运行机器人。

### 4.3 不依赖数组的隐含顺序

MuJoCo 端应继续按关节名称查询 `joint id`、`qpos address`、`qvel address` 和执行器，而不是假设：

- MJCF 中的关节声明顺序就是策略顺序；
- `qpos`、`qvel` 和 `ctrl` 的索引相同；
- URDF 转换为 USD 或 MJCF 后仍保留原顺序；
- 正则表达式匹配结果永远不变。

当前运行器已经按 [`bpx_flat.toml`](config/bpx_flat.toml) 中的名称建立 MuJoCo 映射，应保留这种做法。

## 5. 哪些改动后必须重新检查

出现以下任一情况，都应重新执行“Isaac 解析日志 → MuJoCo 验证 → one-hot 测试 → 观测对拍”的完整流程：

- 修改或重新转换 URDF、USD、MJCF；
- 增删关节或执行器；
- 修改关节名称、正则表达式或 `preserve_order`；
- 升级 Isaac Lab、Isaac Sim、MuJoCo；
- 重新训练或重新导出策略；
- 修改动作类型、动作缩放、默认关节角；
- 修改观测项、观测拼接顺序或历史帧数。

## 6. 快速验收清单

- [x] Isaac Lab 日志打印的 12 个动作关节与标准表一致；
- [x] `validate_setup.py` 打印的 12 个关节逐项一致；
- [x] 默认关节角数组采用同一顺序；
- [x] `joint_pos_rel`、`joint_vel_rel` 和 `last_action` 使用同一顺序；
- [x] 12 个 one-hot 动作均控制预期关节；
- [x] Isaac 与 MuJoCo 的正关节方向一致；
- [x] ONNX 输入为 48 维、输出为 12 维；
- [x] 策略文件旁保存了完整版本和关节元数据；
- [x] 模型或机器人资产发生变化后重新完成上述检查。

判断映射正确的核心标准不是“机器人没有报错”，而是“同一个策略索引在训练、观测、控制和部署的每一层始终代表同一个物理关节”。
