# BPX 四足机器人平地行走环境 —— 配置说明

本文档说明如何将原模板中的 cartpole(倒立摆)任务替换为 **BPX 四足机器人在平地上行走的速度跟踪任务**。所有改动均已通过实际运行验证(环境创建、50 步零动作仿真、2 次 PPO 训练迭代)。

---

## 1. 改动总览

| 文件 | 改动 |
|---|---|
| [bpx_base_env_cfg.py](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_base_env_cfg.py) | BPX 机器人、场景、关节映射以及三个策略共同的观测/动作接口 |
| [bpx_locomotion_env_cfg.py](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_locomotion_env_cfg.py) | 行走任务的指令、重置、奖励和终止条件 |
| [bpx_stand_env_cfg.py](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_stand_env_cfg.py) | 零指令站立任务配置 |
| [bpx_init_env_cfg.py](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_init_env_cfg.py) | 趴卧状态恢复四足稳定站姿的 Init 任务配置 |
| [mdp/\_\_init\_\_.py](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/mdp/__init__.py) | 补充导入 `feet_air_time`(该函数在 `isaaclab_tasks` 的 locomotion 模块中,不在核心库里) |
| [rsl_rl_ppo_cfg.py](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/agents/rsl_rl_ppo_cfg.py) | PPO 网络从 `[32,32]` 扩大到 `[128,128,128]`,迭代 150→1500,实验名 `cartpole_direct`→`bpx_flat` |
| [bpx_test/\_\_init\_\_.py](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/__init__.py) | 注册 locomotion、stand、Init 三个任务，并保留 `BPX-Test-v0` 兼容入口 |
| [list_envs.py](scripts/list_envs.py) | 任务过滤前缀:`Template-` → `BPX-` |
| [CHANGELOG.rst](source/BPX_test/docs/CHANGELOG.rst) | 记录本次变更 |

整体设计参考 Isaac Lab 2.2.1 官方 locomotion velocity 环境(`isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`),这是社区验证最充分的四足行走配置范式。

---

## 2. 机器人配置(`BPX_CFG`)

### 2.1 资产加载

```python
BPX_USD_PATH = .../BPX_structure/usd/bpx.usd   # 相对本文件定位,不怕装到别处
spawn = sim_utils.UsdFileCfg(
    usd_path=BPX_USD_PATH,
    activate_contact_sensors=True,          # 启用接触上报(奖励/终止需要)
    rigid_props=...,                        # 关闭额外阻尼、穿透恢复速度 1.0 m/s
    articulation_props=...,                 # 自碰撞开启、位置求解器 4 次迭代
)
```

USD 资产本身由 URDF 转换而来(见 `usd/config.yaml`),转换时关节驱动增益被设为 **0**——所以 PD 增益必须在 Isaac Lab 这边的执行器里配置(见 2.3)。toe(足端)在转换时**未**与 calf 合并(`merge_fixed_joints: false`),因此是独立刚体,可以直接用作足端接触检测,共 17 个刚体(1 躯干 + 4 腿 × 4 连杆)。

### 2.2 默认站姿(初始关节角)

```python
joint_pos = {
    ".*_hip_roll_joint":  0.0,    # 髋外展居中
    ".*_hip_pitch_joint": 0.7,    # 大腿后摆 0.7 rad
    ".*_knee_joint":     -1.4,    # 膝盖弯曲 -1.4 rad
}
```

选取依据(几何计算):
- 大腿长 0.23 m + 小腿长 0.2316 m(含足端)。髋俯仰 +0.7、膝 -1.4 时,足端落在髋关节**正下方**,高度 `-(0.23+0.2316)·cos(0.7) ≈ -0.353 m`,加上足端碰撞体半径后足底约在基座下方 **0.39 m**;
- 因此初始生成高度设为 `pos=(0.0, 0.0, 0.40)`,落地即站稳(冒烟测试中零动作 50 步无翻倒,验证了这一点);
- 关节限位为髋俯仰 [-0.79, 2.61]、膝 [-2.75, -0.55],该姿态处于限位中部,给策略探索留足空间。

### 2.3 执行器(12 关节隐式 PD)

```python
actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
        effort_limit_sim=30.0,    # URDF 力矩上限 30 N·m
        velocity_limit_sim=20.0,  # URDF 速度上限 20 rad/s
        stiffness=40.0,           # PD 比例增益
        damping=1.0,              # PD 微分增益
    ),
}
```

- **隐式执行器**把 PD 控制交给 PhysX 求解器,比显式执行器快 1 个数量级左右,是 locomotion 训练的标准做法;
- 增益按整机质量插值:Go2(约 7 kg)用 25/0.5,ANYmal(约 30 kg)用 80/2,BPX 约 15 kg 取 **40/1.0**;
- 另设 `soft_joint_pos_limit_factor=0.95`,把软限位收紧到硬限位 95%,避免训练中撞击关节限位产生尖峰力。

---

## 3. MDP 设计

### 3.1 观测(48 维,带噪声)

| 项 | 维数 | 噪声 | 说明 |
|---|---|---|---|
| `base_lin_vel` | 3 | U(-0.1, 0.1) | 基座线速度(基座坐标系) |
| `base_ang_vel` | 3 | U(-0.2, 0.2) | 基座角速度 |
| `projected_gravity` | 3 | U(-0.05, 0.05) | 重力在基座系的投影(姿态感知) |
| `velocity_commands` | 3 | 无 | 速度指令 (vx, vy, ωz) |
| `joint_pos` (rel) | 12 | U(-0.01, 0.01) | 关节角相对默认姿态偏差 |
| `joint_vel` (rel) | 12 | U(-1.5, 1.5) | 关节速度 |
| `actions` | 12 | 无 | 上一步动作 |

平地行走不需要高度扫描(`height_scan`),所以没有加 RayCaster——观测更小、训练更快。

### 3.2 动作(12 维)

```python
joint_pos = JointPositionActionCfg(scale=0.5, use_default_offset=True)
# 目标关节角 = 默认关节角 + 0.5 × 动作 ∈ [默认±0.5 rad]
```

策略输出的每个分量是**相对默认站姿的归一化偏移**,这比直接输出绝对角度好学得多。

### 3.3 速度指令

```python
ranges: lin_vel_x=(-1, 1), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1, 1) m/s 或 rad/s
每 10 s 重采样;2% 概率强制原地站立(命令清零)
```

指令在场景中可视化(`debug_vis=True`):机器人上方 0.5 m 处画一对箭头——**绿色 = 期望速度指令**,**蓝色 = 当前实际速度**,长度与速度大小成正比。策略学得好时蓝箭头会紧跟绿箭头。

### 3.4 奖励(核心设计)

**任务奖励(正)**

| 项 | 权重 | 作用 |
|---|---|---|
| `track_lin_vel_xy_exp` | +1.0 | 跟踪线速度指令,`exp(-误差²/0.25)` 核 |
| `track_ang_vel_z_exp` | +0.5 | 跟踪偏航角速度指令 |
| `feet_air_time` | +0.5 | 落地瞬间奖励摆动相时长(>0.4 s),鼓励抬脚迈步而不是拖地滑行;零指令时自动失效 |

**正则惩罚(负)**

| 项 | 权重 | 作用 |
|---|---|---|
| `lin_vel_z_l2` | -2.0 | 抑制上下弹跳 |
| `ang_vel_xy_l2` | -0.05 | 抑制横滚/俯仰角速度 |
| `flat_orientation_l2` | -0.5 | 保持躯干水平 |
| `dof_torques_l2` | -1e-5 | 降低能耗 |
| `dof_acc_l2` | -2.5e-7 | 动作平滑(抑制关节加速度尖峰) |
| `action_rate_l2` | -0.01 | 动作平滑(抑制相邻步动作突变) |
| `joint_deviation_l1` | -0.05 | 姿态接近默认站姿 |
| `undesired_contacts` | -1.0 | 惩罚躯干/髋/大腿/小腿触地(只有脚该着地) |

权重基本沿用官方 velocity 环境的配比(任务奖励主导、惩罚只做塑形),这是经过大量实验检验的平衡点。

### 3.5 终止条件

- `time_out`:20 s 回合结束(只标记不惩罚);
- `base_contact`:躯干触地(力 > 1 N)→ 判定摔倒;
- `bad_orientation`:躯干倾角 > 1 rad → 判定翻倒。

### 3.6 事件(域随机化)

| 事件 | 时机 | 内容 |
|---|---|---|
| `physics_material` | 启动 | 摩擦系数随机化(静摩擦 0.6–1.2,64 桶) |
| `actuator_gains` | 启动 | PD 增益 ±10% 缩放 |
| `reset_base` | 重置 | 基座位置 ±0.5 m、偏航 ±π、速度 ±0.5 随机 |
| `reset_robot_joints` | 重置 | 默认关节角 ×(0.5–1.5) 随机缩放 |
| `push_robot` | 每 10–15 s | 基座施加 ±0.5 m/s 随机水平速度冲击 |

随机化让策略对模型误差更鲁棒,是 sim2real 的基础。

---

## 4. 训练配置

| 项 | 值 | 说明 |
|---|---|---|
| 物理频率 | 200 Hz(`dt=0.005`) | |
| 控制频率 | 50 Hz(decimation=4) | locomotion 标准配置 |
| 并行环境数 | 2048 | 针对 8 GB 的 RTX 4060 Laptop;大显存可加到 4096 |
| PPO 网络 | [128, 128, 128] ELU | actor/critic 同构 |
| 每环境步数 | 24 | |
| 迭代次数 | 1500 | 平地行走一般几百次迭代即可见效 |
| 学习率 | 1e-3 自适应(目标 KL 0.01) | |
| entropy | 0.01 | 鼓励探索 |
| 实验名 | `bpx_locomotion` | 新行走任务日志在 `logs/rsl_rl/bpx_locomotion/<时间戳>/`；旧兼容任务仍使用 `bpx_flat` |

---

## 5. 使用方法

```bash
# 进入 isaaclab conda 环境(本机已将 BPX_test 以 editable 方式安装)
conda activate isaaclab
cd /home/hanguang/BPX_test

# 查看已注册任务
python scripts/list_envs.py

# 分别训练三个策略（可先减少迭代数做冒烟测试）
python scripts/rsl_rl/train.py --task BPX-Locomotion-v0 --headless
python scripts/rsl_rl/train.py --task BPX-Stand-v0 --headless
python scripts/rsl_rl/train.py --task BPX-Init-v0 --headless
python scripts/rsl_rl/train.py --task BPX-Locomotion-v0 --max_iterations 300

# 用 TensorBoard 看训练曲线
tensorboard --logdir logs/rsl_rl

# 回放最新 checkpoint(带可视化)
python scripts/rsl_rl/play.py --task BPX-Locomotion-v0

# 单机器人回放 play.py 本身就支持，加 --num_envs 1 即可：
python scripts/rsl_rl/play.py --task BPX-Locomotion-v0 --num_envs 1 --real-time

# 零动作/随机动作 sanity check
python scripts/zero_agent.py --task BPX-Stand-v0
python scripts/random_agent.py --task BPX-Init-v0

#查看 TensorBoard 曲线
训练时或训练后，另开一个终端：
conda activate isaaclab
cd /home/hanguang/BPX_test
tensorboard --logdir logs/rsl_rl
然后浏览器打开 http://localhost:6006 即可。它会在终端里打印出这个网址。

几个要点：
可以和训练同时开，曲线会实时刷新(每 24 步一个数据点)；
三个新任务分别写入 `logs/rsl_rl/bpx_locomotion`、`bpx_stand` 和 `bpx_init`；旧的
`BPX-Test-v0` 仍写入 `bpx_flat`。TensorBoard 指向 `logs/rsl_rl` 可以同时查看所有策略。
重点看的曲线(左侧 SCALARS 标签页)：
Episode Rewards/track_lin_vel_xy_exp 和 Episode Rewards/track_ang_vel_z_exp —— 上升 = 学会跟踪速度指令
Train/Mean episode length —— 逼近 1000(20 s × 50 Hz)= 几乎不再摔倒
Policy/mean_noise_std —— 持续下降 = 策略逐渐收敛
Episode Rewards/total —— 总奖励综合走势
```


**判断训练成功的信号**(TensorBoard 中):
- `Episode Rewards/track_lin_vel_xy_exp` 持续上升并趋平;
- `Policy/Mean noise std` 逐渐下降;
- 回合长度(`Train/Mean episode length`)逼近 1000 步(= 20 s × 50 Hz),说明摔倒越来越少。

---

## 6. 已做的验证

1. **冒烟测试**:4 环境创建 + 50 步零动作仿真
   - 观测空间 `(4, 48)`、动作空间 `(4, 12)` ✓
   - 12 关节 / 17 刚体,默认关节角 `[0, 0.7, -1.4]` 正确下发 ✓
   - 零动作 50 步无翻倒(初始站姿稳定)✓
2. **短训练**:64 环境 × 2 次 PPO 迭代
   - 训练循环、日志落盘(`logs/rsl_rl/bpx_flat/`)、checkpoint 保存全部正常 ✓

---

## 7. 常见调整方向

| 想要的效果 | 调哪里 |
|---|---|
| 走不快/想跑起来 | 扩大指令范围 `lin_vel_x=(‑2, 2)`,并同步加大 `episode_length_s` |
| 学不出抬脚步态 | 提高 `feet_air_time` 权重(0.5→1.0)或降低 `threshold`(0.4→0.3) |
| 动作抖动大 | 加大 `action_rate_l2`/`dof_acc_l2` 权重 |
| 站不稳就摔 | 先减小指令范围只训站立(`lin_vel_x=(‑0.2, 0.2)`),再逐步放大(指令课程) |
| 显存不足(OOM) | `--num_envs 1024` 或 512 |
| 想上粗糙地形 | 给 `terrain` 换 `ROUGH_TERRAINS_CFG`,加回 `height_scanner` 观测与 `terrain_levels` 课程(参考官方 velocity 环境) |
| 想做 sim2real | 加 `randomize_rigid_body_mass`、执行器延迟(`DelayedPDActuatorCfg`)、推力幅度加大 |
