# BPX 四足机器人运动改进文献指南

本文按 2026-09-17 的 BPX 源码更新工程背景，并核对所列主要论文的原始页面。它是一份围绕现有工程的选读指南，不是最新文献的完整综述；“对 BPX 的启发”与实施顺序是本项目的工程建议，不表示论文方法已在 BPX 上实现或验证。

## 1. 当前系统概况

BPX 使用 Isaac Lab 与 RSL-RL/PPO 训练 12 自由度四足机器人，目前有平地行走、零指令站立和趴卧起身三个任务，并在 MuJoCo 中用六状态 Supervisor 切换策略。以下奖励介绍主要针对行走任务。

- 控制频率：50 Hz，物理仿真频率：200 Hz；
- 动作：相对默认关节姿态的关节位置目标；
- 观测：基座速度、角速度、重力投影、速度指令、关节状态和上一时刻动作；
- 正向奖励：线速度跟踪、偏航速度跟踪、足端腾空时间；
- 惩罚项：机身垂向速度、横滚/俯仰角速度、姿态倾斜、力矩、关节加速度、动作变化、关节偏离和非足端接触；
- 鲁棒性训练：公共摩擦系数、PD 增益和观测噪声，以及任务专属重置；行走/站立另有周期推扰；
- PPO：三层 `[128, 128, 128]` ELU 网络，自适应 KL 学习率；行走和站立默认 1500 次迭代，Init 默认 3000 次；
- 行走指令：10% 静止、25% 纯旋转、65% 混合运动；Init 已有基于终帧成功率的力矩惩罚课程。

相关实现：

- [公共机器人、观测和动作配置](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_base_env_cfg.py)
- [行走奖励和随机化配置](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_locomotion_env_cfg.py)
- [起身任务与课程](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/bpx_init_env_cfg.py)
- [跨仿真运行说明](sim2sim/README.md)
- [PPO 配置](source/BPX_test/BPX_test/tasks/manager_based/bpx_test/agents/rsl_rl_ppo_cfg.py)
- [训练曲线解读](RL_TRAINING_CURVES.md)

旧文档记录的一次早期行走训练在第 1499 轮附近有以下指标。本次未重新读取原事件文件或复算，运行标识和统计口径不完整，故仅保留为历史记录；不能作为当前三个策略的验收结果。

| 指标 | 结果 |
|---|---:|
| 平均回合长度 | 约 991/1000 步 |
| 正常超时结束比例 | 约 98.7% |
| 线速度跟踪误差 | 约 0.158 m/s |
| 偏航角速度跟踪误差 | 约 0.259 rad/s |
| 线速度跟踪奖励 | 约 0.963 |
| 偏航跟踪奖励 | 约 0.454/0.5 |

这些旧记录不足以证明当前模型已覆盖完整速度范围或跨仿真切换。先用固定评估核对起身、交接、停车和各方向运动，再考虑以下研究方向：

1. 让步态更自然、平滑且可控；
2. 改善状态估计和真实传感器适配；
3. 扩充动力学随机化并缩小 sim-to-real 差距；
4. 增强未知地形、扰动和负载变化下的适应能力；
5. 进一步学习跳跃、攀爬等敏捷运动。

## 2. 最优先阅读的五篇文献

### 2.1 Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning

Nikita Rudin et al., 2022.

- [论文页面与 PDF](https://proceedings.mlr.press/v164/rudin22a.html)
- 与当前 Isaac Lab、RSL-RL、大规模并行 PPO 技术路线最接近；
- 重点关注课程学习、地形难度递增、奖励设计、域随机化和大规模并行训练；
- 可用于设计 BPX 后续速度范围与地形课程；当前 Init 的成功率课程是本项目的具体实现，不能直接等同于论文中的课程。

### 2.2 Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior

Gabriel B. Margolis and Pulkit Agrawal, 2023.

- [论文页面](https://proceedings.mlr.press/v205/margolis23a.html)
- 适合解决步态单一、动作不够自然以及步态难以显式控制的问题；
- 将踏频、相位关系、抬脚高度和身体姿态等作为可调行为参数；
- 对 BPX 最直接的启发是：不要只依赖 `feet_air_time` 让步态自行涌现，而应考虑 gait-conditioned policy。

### 2.3 Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion

Gwanghyeon Ji et al., 2022.

- [论文](https://arxiv.org/abs/2202.05481)
- 联合训练控制策略和状态估计器；
- 从关节、IMU 和历史观测估计基座线速度、足端高度及接触状态；
- 当前 BPX actor 直接使用仿真提供的 `base_lin_vel`，因此这篇论文对真机部署非常重要。

### 2.4 Sim-to-Real: Learning Agile Locomotion for Quadruped Robots

Jie Tan et al., 2018.

- [论文页面](https://www.roboticsproceedings.org/rss14/p10.html)
- 系统介绍系统辨识、执行器模型、控制延迟、紧凑观测和动力学随机化；
- BPX 已有摩擦、PD 增益、观测噪声、重置扰动与任务专属推扰，尚未覆盖完整质量/质心变化、真实电机响应、通信延迟和传感器偏置；
- 适合作为扩展 sim-to-real 配置的操作指南。

### 2.5 Learning Agile and Dynamic Motor Skills for Legged Robots

Jemin Hwangbo et al., 2019.

- [论文](https://arxiv.org/abs/1901.08652)
- 四足机器人强化学习 sim-to-real 的经典工作；
- 重点关注 learned actuator model、动力学随机化、能耗、快速运动和跌倒恢复；
- 有助于分析理想隐式 PD 与真实 BPX 电机之间的差异。论文中的跌倒恢复范围不能直接套用于 BPX：当前 Init 只覆盖腹部朝地且接近水平的趴姿。

## 3. 按改进目标选择文献

| 改进目标 | 推荐文献 | 对 BPX 的主要启发 |
|---|---|---|
| 多种步态与步态参数控制 | [Walk These Ways](https://proceedings.mlr.press/v205/margolis23a.html) | 加入步态相位、踏频、占空比、抬脚高度和姿态指令 |
| 大规模并行训练与课程学习 | [Learning to Walk in Minutes](https://proceedings.mlr.press/v164/rudin22a.html) | 参考并行训练与地形课程，设计 BPX 难度调整实验 |
| 真机状态估计 | [Concurrent Training of a Control Policy and a State Estimator](https://arxiv.org/abs/2202.05481) | 用观测历史估计基座速度、足高和接触状态 |
| 基础 sim-to-real | [Sim-to-Real: Learning Agile Locomotion](https://www.roboticsproceedings.org/rss14/p10.html) | 系统辨识、延迟建模、执行器模型与随机化 |
| 高动态 sim-to-real | [Learning Agile and Dynamic Motor Skills](https://arxiv.org/abs/1901.08652) | learned actuator model、能耗和跌倒恢复 |
| 快速适应负载和动力学变化 | [RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034) | 从近期观测历史推断隐含动力学参数 |
| 无视觉复杂地形运动 | [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251) | privileged learning、本体感知历史和复杂地形训练 |
| 仅依靠有限传感器适应地形 | [DreamWaQ](https://arxiv.org/abs/2301.10602) | 从本体感知历史学习隐式地形表示 |
| 本体感知与外部地形信息融合 | [Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild](https://arxiv.org/abs/2201.08117) | 鲁棒融合本体感知和外部地形感知 |
| 跳跃、攀爬和钻越 | [ANYmal Parkour](https://arxiv.org/abs/2306.14874) | 低层运动技能与高层技能选择的分层控制 |
| 端到端视觉跑酷 | [Extreme Parkour with Legged Robots](https://arxiv.org/abs/2309.14341) | 从深度图直接产生敏捷运动控制 |
| 理解接触力与动态步态 | [Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control](https://biomimetics.mit.edu/publications/e9ca891c-35de-4be5-896c-780feae67fd4/) | 支撑相、摩擦锥、地面反力和预测控制 |
| 理解当前 PPO 优化器 | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | 理解 clipped objective、优势函数和策略更新幅度 |

## 4. 推荐阅读与实施顺序

### 前置：先验证当前三策略闭环

无需先扩展论文方法，就应完成固定模型与接口核对、趴姿等待、RB 起身、Init→Stand 交接以及停车评估。记录每份模型的训练参数和成功率，区分训练 `recovered` 与部署稳定确认。现有 Supervisor 是规则状态机，不是学习得到的高层规划策略。

### 阶段一：提高平地步态质量

首先阅读 **Walk These Ways**，然后为训练和回放增加以下评价指标：

- 四足触地时序与对角腿相位关系；
- 步频、占空比和足端腾空高度；
- 左右、前后腿动作对称性；
- 足端接触期间的水平滑移速度；
- 足端落地冲击和峰值接触力；
- 关节力矩、关节速度及其饱和比例；
- 单位运输能耗（Cost of Transport）；
- 不同速度区间下的跟踪误差。

在这些指标存在以后，再比较：

1. 当前自由涌现步态；
2. 加入步态相位或 gait clock 的策略；
3. 加入踏频、抬脚高度和身体高度指令的 gait-conditioned policy。

### 阶段二：解决状态估计问题

阅读 **Concurrent Training of a Control Policy and a State Estimator**。以下是 BPX 可比较的设计路线，联合估计器与 teacher-student 是不同方案，不应把列表理解成该论文原样采用的单一流程：

- 为策略加入若干步观测历史或循环网络；
- 使用 teacher policy 读取仿真真值；
- 让部署用 student policy 只读取真机可获得的传感器量；
- 逐步移除 actor 对仿真真值 `base_lin_vel` 的依赖。

### 阶段三：增强 sim-to-real

结合 **Tan 2018**、**Hwangbo 2019** 和 **RMA**，逐项加入：

- 机身和各腿质量随机化；
- 质心位置与惯量随机化；
- 电机强度、摩擦和阻尼随机化；
- 动作延迟、观测延迟与随机丢帧；
- IMU 偏置和关节编码器噪声；
- 地面摩擦、恢复系数及坡度变化；
- 电池电压、负载和单腿扰动变化；
- 更符合真实电机特性的执行器模型。

每次只引入一类随机化，并在固定测试集合上记录性能变化，避免随机化过强导致策略保守。

### 阶段四：扩展复杂地形

依次阅读：

1. **Learning Quadrupedal Locomotion over Challenging Terrain**；
2. **DreamWaQ**；
3. 需要视觉时再读 **Learning Robust Perceptive Locomotion in the Wild**。

对应实施顺序建议为：

1. 加入斜坡、台阶、随机高度场和离散障碍；
2. 加入地形难度课程；
3. 使用 teacher 的特权地形信息训练 student；
4. 比较纯本体感知、隐式地形估计和显式高度扫描三条路线。

### 阶段五：敏捷运动

当普通复杂地形行走已经可靠后，再参考 **ANYmal Parkour** 和 **Extreme Parkour**：

- 分别训练跳跃、攀爬、钻越等低层技能；
- 为不同技能设计专用课程、终止条件和目标；
- 最后训练高层策略选择和组合低层技能。

## 5. 对当前奖励调整的注意事项

早期记录中的 `feet_air_time` 分项略为负值，不能据此推断当前模型的步态。现用奖励在落足时计算腾空时间减去 `0.4 s`，且仅在平面指令模长 >0.1 m/s 时启用；纯旋转和静止不启用。负值可能表示落足前腾空较短，不等于应直接增大权重。

直接提高该项权重可能产生：

- 过度抬腿；
- 步频过低；
- 落地冲击变大；
- 为获取腾空奖励而牺牲速度跟踪；
- 仿真中好看但真机能耗和稳定性变差。

因此，应先记录步频、占空比、足端高度、滑移、冲击和能耗，再决定是修改阈值、奖励权重，还是采用显式步态条件策略。

## 6. 最短阅读路径

如果只准备先读五篇，推荐顺序为：

1. **Learning to Walk in Minutes**：理解当前训练范式；
2. **Walk These Ways**：改进和控制步态；
3. **Concurrent Training of a Control Policy and a State Estimator**：解决真实观测问题；
4. **Sim-to-Real: Learning Agile Locomotion**：建立完整 sim-to-real 方法；
5. **Learning Agile and Dynamic Motor Skills**：理解执行器建模与动态运动迁移。

需要在线适应时再读 **RMA**。阅读顺序应服从实际评估结果：先验证当前三策略闭环，随后改善步态、解决状态估计和现实差异，最后扩展复杂地形及敏捷运动。
