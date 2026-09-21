# BPX 历史观测与 Sim-to-Real 状态估计方案

整理日期：2026-09-21；工程背景来自 2026-09-20 的讨论与源码检查。

本文整理三个相互关联的研究方向：历史观测、非对称 Actor–Critic，以及基于 IMU 和接触约束的速度估计。工程现状依据本次讨论中检查的源码；参数和实施顺序是待验证的实验建议，不代表已经实现或取得性能提升。论文链接指向原文、作者提交版本或正式会议页面。

## 1. 核心建议

优先验证“历史观测＋非对称 Actor–Critic”，再比较隐式速度推断、学习式速度估计与 EKF/InEKF。

三个方向可以组合：

- **历史观测**：补充单帧观测缺失的时序信息，帮助推断接触变化和动力学响应。
- **非对称 Actor–Critic**：Actor 只使用部署时可获得的信息，Critic 在训练中额外使用仿真真值。
- **速度估计**：通过学习式估计器或接触辅助滤波器，为 Actor 和上层状态机提供可部署的速度信息。

“真实速度只给 Critic”与“Actor 使用估计速度”完全兼容。需要区分不给 Actor **真实速度**，以及不给 Actor **任何显式速度估计**这两种设计。

## 2. 当前工程基线

### 2.1 观测和控制

当前公共配置只有 `policy` 观测组，使用 48 维单帧观测，尚未单独配置 Critic 的特权观测。

| 观测项 | 维度 | 真机来源或处理方式 |
|---|---:|---|
| 基座线速度 `base_lin_vel` | 3 | 需要状态估计 |
| 基座角速度 `base_ang_vel` | 3 | IMU 陀螺仪，需标定和坐标变换 |
| 重力投影 `projected_gravity` | 3 | 姿态估计；不能直接把运动中的加速度计输出当作重力 |
| 速度指令 `[vx, vy, wz]` | 3 | 手柄或上层控制器 |
| 相对关节位置 | 12 | 关节编码器 |
| 关节速度 | 12 | 电机反馈或编码器处理 |
| 上一时刻动作 | 12 | 控制程序内部记录 |

- 物理仿真频率：200 Hz。
- 策略频率：50 Hz，控制周期为 0.02 s。
- 策略使用 MLP，默认隐藏层为 `[128, 128, 128]`。
- 训练侧线速度来自仿真状态，并叠加均匀噪声。
- MuJoCo 部署侧直接读取仿真真实速度，尚未通过实际状态估计器生成速度观测。

因此，现有 sim2sim 结果尚不能说明策略能够适应真机速度估计误差。

### 2.2 相关代码入口

| 文件 | 与本方案的关系 |
|---|---|
| [公共环境配置](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_base_env_cfg.py) | 当前观测定义、动作定义和控制频率 |
| [复杂地形配置](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_rough_env_cfg.py) | 盲走地形、指令与课程 |
| [PPO 配置](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/agents/rsl_rl_ppo_cfg.py) | Actor/Critic 网络和 PPO 参数 |
| [MuJoCo 机器人接口](sim2sim/bpx_sim2sim/robot.py) | 速度坐标转换与 48 维观测构造 |
| [MuJoCo 运行器](sim2sim/bpx_sim2sim/runner.py) | 策略调用、观测传递和状态机输入 |
| [Supervisor](sim2sim/bpx_sim2sim/supervisor.py) | 停止与站稳判断中的速度依赖 |
| [地形部署配置](sim2sim/config/bpx_terrain.toml) | 观测维度、控制周期和策略路径 |

## 3. 方向一：加入历史观测

### 3.1 为什么可能有效

单帧关节状态和姿态，往往不足以区分脚刚碰到障碍、支撑脚正在打滑或机身受到持续推力等情况。历史观测和历史动作提供“施加动作之后发生了什么”的信息，使网络有机会推断接触状态和环境变化。

RMA 使用近期状态与动作估计环境隐变量，是这一路线的典型例子。[RMA 原文](https://ashish-kmr.github.io/rma-legged-robots/rma-locomotion-final.pdf)

### 3.2 BPX 的起步方案

先采用**历史帧堆叠＋MLP**建立基线，再根据收益决定是否引入时间卷积编码器或 GRU。

定义包含当前帧的历史窗口：

```text
history_t = [o_(t-H+1), ..., o_(t-1), o_t]
action_t = actor(history_t)
```

| 窗口帧数 H | 首末帧时间跨度（50 Hz） | 删除线速度后堆叠维度 |
|---|---:|---:|
| 5 | 0.08 s | 225 |
| 10 | 0.18 s | 450 |
| 20 | 0.38 s | 900 |

上述窗口通常可近似称为 0.1、0.2、0.4 秒窗口；严格的首末帧跨度为 `(H-1) × 0.02 s`。这些数值只是实验起点。

部署时需要统一：

- 历史帧和观测项的排列顺序；
- 历史更新频率，按策略周期更新；
- episode 复位时的清空和填充规则；
- Init、Stand、Walk 策略切换时的历史维护规则；
- 观测缩放、噪声、裁剪和动作记录语义。

### 3.3 能力边界

历史信息主要改善接触后的判断和适应，不能可靠预知尚未接触的沟槽或落脚点。如果未来目标包括高速穿越、精确踩点，需要考虑深度相机等外感知。Miki 等人的工作讨论了本体感知与外感知的互补性。[论文](https://arxiv.org/abs/2201.08117)

### 3.4 推荐论文

| 论文 | 发表信息 | 阅读重点与 BPX 关联 |
|---|---|---|
| [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251) — Joonho Lee et al. | Science Robotics, 2020 | 盲走复杂地形、本体感知时序信息、教师—学生训练，以及真实自然地形验证 |
| [RMA: Rapid Motor Adaptation for Legged Robots](https://www.roboticsproceedings.org/rss17/p011.html) — Ashish Kumar et al. | RSS, 2021 | 历史状态与动作如何生成环境隐变量；判断是否需要独立适应模块 |
| [Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild](https://arxiv.org/abs/2201.08117) — Takahiro Miki et al. | Science Robotics, 2022 | 循环编码器融合本体与外感知；作为后续加入地形感知的参考 |

## 4. 方向二：真实线速度只给 Critic

### 4.1 应该删除哪一种速度

建议讨论的目标是三维**基座线速度真值**。基座角速度、关节速度和速度指令应保留，它们具有可用的真机来源，承担不同作用。

删去 `base_lin_vel` 后，Actor 单帧观测由 48 维变为 45 维：

```text
o_t = [base_ang_vel, projected_gravity, velocity_command,
       joint_pos_rel, joint_vel, last_action]
```

初始实验可让 Actor 接收上述历史，让 Critic 接收相同历史以及真实基座线速度。后续再按需要加入接触状态、地形高度等特权信息。

### 4.2 非对称结构的作用与限制

Critic 使用更充分的状态信息来辅助训练，Actor 始终使用可部署的观测。这是非对称 Actor–Critic 的基本思想。[Pinto et al., RSS 2018](https://www.roboticsproceedings.org/rss14/p08.html)

需要明确：

- Critic 看到真值不会使部署时的 Actor 自动获得这些信息；Actor 仍需依赖历史推断或显式估计器。
- 速度跟踪奖励可以继续使用仿真真实速度。奖励不需要受到 Actor 传感器集合的限制。
- Actor 不接收真实速度，必须从训练阶段就体现；不能只在部署阶段把已有策略的速度输入清零。
- 使用真实速度监督估计器，与向 Actor 输入真实速度是不同的操作。

### 4.3 两条可比较的策略路线

| 路线 | Actor 输入 | 优点 | 主要问题 |
|---|---|---|---|
| 隐式推断 | 45 维本体观测历史 | 结构简单，不需要独立速度输出 | 难以直接诊断策略是否正确理解速度 |
| 显式学习式估计 | 当前观测或历史＋估计速度＋可选隐变量 | 速度误差可量化；可为其他控制模块提供输出 | 需要估计器训练，并验证策略对估计误差的适应性 |

显式估计的一种结构是：

```text
本体观测历史 → 估计器 → 估计速度 v_hat、环境隐变量 z
当前本体观测 + v_hat + z → Actor → 动作
观测信息 + 仿真真实速度 + 可选特权信息 → Critic（仅训练）
```

DreamWaQ 将历史本体观测、速度估计、环境隐变量和非对称 Actor–Critic 放在同一框架内。其 Actor 使用估计速度，并非完全不使用速度。[DreamWaQ 原文](https://arxiv.org/pdf/2301.10602)

### 4.4 推荐论文

| 论文 | 发表信息 | 阅读重点与 BPX 关联 |
|---|---|---|
| [Asymmetric Actor Critic for Image-Based Robot Learning](https://www.roboticsproceedings.org/rss14/p08.html) — Lerrel Pinto et al. | RSS, 2018 | 理解完整状态 Critic 与部分观测 Actor 的训练逻辑；原实验为视觉机器人任务 |
| [DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning](https://arxiv.org/abs/2301.10602) — I Made Aswin Nahrendra et al. | ICRA, 2023 | 与当前需求最直接相关；重点看历史、估计速度、潜变量及特权 Critic 的连接方式 |
| [Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481) — Gwanghyeon Ji et al. | IEEE RA-L, 2022 | 策略与估计器共同训练；估计基座速度、足端高度和接触概率 |

## 5. 方向三：IMU＋腿部运动学＋EKF/InEKF

### 5.1 为什么仅有 IMU 和滤波器还不够

IMU 提供角速度与比力。积分速度会受到 IMU 零偏和姿态误差影响，EKF 本身不能凭空提供修正信息。足式机器人常用接触辅助估计：

```text
IMU → 状态预测
关节编码器 + 腿部运动学 + 接触判断 → 观测更新
滤波器 → 姿态、速度、偏置等估计
```

当支撑脚不滑动时，其相对于地面的速度接近零，可用于约束机身运动。Bloesch 的经典工作与 Hartley 的 InEKF 工作给出了相关融合方法。[Bloesch 2012](https://www.roboticsproceedings.org/rss08/p03.html)、[Hartley 2020](https://arxiv.org/abs/1904.09251)

### 5.2 BPX 的工程重点

1. **接触和打滑处理。** 错把滑动的足端视为固定接触会引入错误约束，需要考虑接触可信度和滤波更新权重。
2. **模拟实际估计误差。** 现有独立均匀噪声不足以表达持续偏差、相关噪声、延迟和接触切换异常。先建立误差模型，进一步在仿真中运行实际估计器。
3. **统一坐标系和参考点。** 当前 MuJoCo 接口已对根 link 原点与根 link 质心的线速度进行转换。真机估计器需要与其统一，并处理 IMU 安装位置和姿态外参。
4. **统一时间戳与采样。** 估计器更新周期可以不同于策略周期，但送入策略的数据必须具有清楚的时序含义。
5. **保留整机层面的状态估计需求。** 当前 Supervisor 使用平面速度进行停止和站稳判断；即使 Actor 不接收速度，这部分仍需要可部署的速度来源。

在真机传感器规格、接触信息质量和计算平台尚未确认前，不能断言 EKF 或学习式估计一定更适合 BPX，应通过统一场景中的闭环表现比较。

### 5.3 推荐论文

| 论文 | 发表信息 | 阅读重点与 BPX 关联 |
|---|---|---|
| [State Estimation for Legged Robots — Consistent Fusion of Leg Kinematics and IMU](https://www.roboticsproceedings.org/rss08/p03.html) — Michael Bloesch et al. | RSS, 2012 | 入门首选：IMU 预测、运动学更新、接触点状态与可观测性一致性 |
| [Contact-Aided Invariant Extended Kalman Filtering for Robot State Estimation](https://arxiv.org/abs/1904.09251) — Ross Hartley et al. | IJRR, 2020；相关会议版本为 RSS 2018 | 推荐完整期刊版本：接触增删、IMU 偏置、InEKF 一致性与收敛性质 |
| [Legged Robot State Estimation using Invariant Kalman Filtering and Learned Contact Events](https://proceedings.mlr.press/v164/lin22b.html) — Tzu-Yuan Lin et al. | CoRL 2021，论文集发表于 2022 | 学习接触事件并结合 InEKF；适合缺少可靠足端接触传感器的情况 |

## 6. 建议的对照实验

### 6.1 实验分组

| 实验 | Actor 输入 | 主要问题 |
|---|---|---|
| A：当前基线 | 单帧＋仿真线速度 | 当前能力是多少？ |
| B：增加历史 | 历史＋仿真线速度 | 历史本身是否改善地形适应？ |
| C：删除真实线速度 | 45 维本体观测历史 | 历史能否补偿缺失的速度观测？ |
| D1：学习式速度估计 | 与 C 相同的历史＋学习式估计速度 | 显式学习式估计是否改善闭环表现？ |
| D2：滤波速度估计 | 与 C 相同的历史＋EKF/InEKF 估计速度 | 传统估计器在相同策略条件下表现如何？ |

为了隔离变量，B/C/D1/D2 使用相同的 Critic 输入、奖励、地形课程、训练预算和评估场景。严格衡量 A→B 的历史收益时，应补充一个与 B 使用相同 Critic 的单帧对照；原始 A 保留为工程基线。

先使用直接堆叠的共同结构进行比较。若之后改成 DreamWaQ 式隐变量编码器，应记录为额外结构变化，不把全部收益归因于速度估计。

### 6.2 评估指标

- **地形通过率**：在固定评估地形上测试，并包含未用于训练的地形组合。
- **速度跟踪误差**：分别记录前向、横向与偏航跟踪误差。
- **跌倒和恢复**：记录跌倒率、推扰后的恢复时间。
- **停止与交接稳定性**：检查 Walk→Stop→Stand 和策略切换后的表现。
- **动作与执行器表现**：记录动作变化、力矩及饱和情况。
- **估计器表现**：D1/D2 额外记录速度估计误差、偏差、延迟及接触切换时的异常。

使用多个随机种子报告均值与波动，不只比较训练奖励。估计误差较小也不自动意味着闭环控制更好，两类指标应同时记录。

## 7. 实施顺序与阅读顺序

建议按以下顺序推进：

1. 固定当前模型和评估场景，建立可重复基线。
2. 保留仿真速度，增加历史观测，比较窗口长度。
3. 分离 Actor/Critic 观测，删除 Actor 的真实线速度并重新训练。
4. 增加学习式速度估计，观察速度跟踪、扰动恢复和停车表现。
5. 明确真机 IMU、编码器、接触信息与时间同步条件，建立 EKF/InEKF 对照。
6. 将选定的观测链路同步到 sim2sim，并验证历史、估计器和多策略切换的一致性。

优先阅读：**DreamWaQ → RMA → Bloesch 2012 → Hartley 2020**。其余论文用于补充时序策略、非对称训练原理、联合估计训练和外感知扩展。

已有的 [工程改进路线图](BPX_IMPROVEMENT_ROADMAP.md) 和 [运动改进文献指南](BPX_LOCOMOTION_LITERATURE.md) 提供更广泛背景；本文聚焦本轮讨论的历史观测与状态估计。
