本文对照 2026-10-15 查阅的 GitHub 主分支源码，分析 BPX 当前楼梯训练的改进方向。结论基于代码和现有 stairs_v3 日志，没有复现外部项目训练，也没有对 BPX 重新训练。链接指向当时访问的 main/master 分支，并非固定提交快照。项目作者展示的通过能力不等于 BPX 的实测能力。

**最值得先验证的是训练信息、技能反馈和课程的配合。** BPX 已经具备楼梯相对高度、局部净空、接触过滤、落脚冲击、执行延迟和历史观测等设计。当前问题不能仅归结为缺少某个通用奖励项；尤其 v3 已经有归一化的停驻线速度和角速度惩罚。

**开源参照的任务与输入有明显区别。**

| 参照 | 已核对的实现事实 | 与 BPX 的比较价值和限制 |
| --- | --- | --- |
| 宇树官方 unitree_rl_gym | Go2 配置继承基类；查到的基类默认地形是 plane，Go2 类没有覆盖这一项 | 可参考关节动作和基础 PPO；类名含 Rough 不足以证明默认训练楼梯。[Go2 配置](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py)、[基类配置](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot_config.py) |
| 宇树官方 unitree_rl_lab | 查到的 Go2 地形字典只启用了 flat，楼梯项被注释；Critic 增加线速度、关节力矩，部分本体量无噪声 | 适合参考训练信息分配，不能把该默认配置当成官方高楼梯复现方案。[Go2 配置](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py) |
| 宇树官方 unitree_rl_mjlab | rough 基础配置的 Actor 读取高度扫描和步态相位；Critic 关闭观测噪声，并增加足端相关信息；行走和站立使用不同约束 | 可参考按指令启用奖励及更完整的 Critic；直接采用其 Actor 会改变 BPX 的盲走输入约定。[基础配置](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/velocity_env_cfg.py)、[Go2 配置](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/config/go2/env_cfgs.py) |
| ETH legged_gym | 标准 rough 实现组合多种地形，按行走距离调整难度；摆动时间奖励在零移动指令下关闭 | 适合参考地形课程和运动奖励的启用条件；其通过目标与 BPX 的中途严格停驻不同。[环境实现](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)、[地形生成](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/utils/terrain.py) |
| MIT Improbable-AI / Walk These Ways，研究作者仓库 | 历史适应模块估计训练特权量；训练入口启用步态命令、时钟输入和 30 帧历史 | 可参考历史状态估计与动作平滑，当前策略并不是与 BPX 同输入的纯速度接口。[Actor/Critic](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym_learn/ppo_cse/actor_critic.py)、[训练入口](https://github.com/Improbable-AI/walk-these-ways/blob/master/scripts/train.py) |
| Extreme Parkour，研究作者仓库 | 先训练扫描地形输入策略，再进行深度视觉蒸馏；包含历史编码、目标方向进度和边缘接触代价 | 可参考分阶段训练及针对性接触奖励；它的感知链与高动态任务不能直接替代 BPX 盲走停驻。[训练流程](https://github.com/chengxuxin/extreme-parkour)、[网络](https://github.com/chengxuxin/extreme-parkour/blob/main/rsl_rl/rsl_rl/modules/actor_critic.py)、[奖励实现](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot.py) |
| ashutoshpanpalia / unitree_go2_stairs，社区项目 | Actor 输入线速度和 187 点高度扫描，台阶设置为 2–10 cm；作者明确说明是仿真项目 | 是直接的上下楼梯示例，但缺少与 BPX 相同的严格停驻任务与实机证据。[配置](https://github.com/ashutoshpanpalia/unitree_go2_stairs/blob/main/source/unitree_go2_stairs/unitree_go2_stairs/tasks/stairs_walking/unitree_go2_stairs/unitree_go2_stairs_env_cfg.py)、[观测与奖励](https://github.com/ashutoshpanpalia/unitree_go2_stairs/blob/main/source/unitree_go2_stairs/unitree_go2_stairs/tasks/stairs_walking/unitree_go2_stairs/unitree_go2_stairs_env.py)、[作者说明](https://github.com/ashutoshpanpalia/unitree_go2_stairs) |

补充参照是 ICRA 2024 的 [Reinforcement Learning for Blind Stair Climbing with Legged and Wheeled-Legged Robots](https://arxiv.org/html/2402.06143v1)。它使用带地形、接触和摩擦信息的特权 Critic，以及位置目标式任务；Go1 结果属于仿真，实机示范使用 Ascento。本文只将其作为论文证据，没有将其描述成已经核验的开源训练仓库，也不把其单台阶结果等同于连续上下楼及中途停驻。

**BPX 目前的训练约束更集中在低速停走。**

BPX 的 Actor 是本体感知 450 维、H=10，Critic 是 480 维且仍有观测噪声。台阶高度和深度在同一梯段内逐阶随机，75% 楼梯回合要求在首、中、末阶段之一停车。当前地形等级又由包含停驻验收的整回合成绩决定。这些组合比固定踏面、连续通过、带高度输入的示例更难直接学习。

现有 stairs_v3 日志中，两条 stop_success 在 5000 轮内全部为零；最近 100 轮有效停驻观测时长的日志均值约 1.39 秒，角速度分项约 25.6%，线速度约 50.7%，支撑约 66.5%。这些是 reset 批次均值的再次平均，包含无样本填零，不能当作事件加权的正式评估率。它们支持优先调查停驻精度，而不能证明唯一原因是探索噪声或某个奖励权重。

**优先改进一：建立能够区分原因的评估，并对齐奖励单位。**

先对原行走模型和 stairs_v3/model_4999 在同一组场景中分别执行均值动作与训练采样动作。保持相同地形、指令脚本、随机化清单，并固定成对的随机数流，避免采样动作消耗随机数后连带改变后续场景。分别统计上/下楼、首/中/末阶停车、难度和速度区间。

每次停车记录是否被安排、是否到达保持、是否完成保持、有效观测时长、五条件联合通过比例、重启情况及最终出口结果。至少分开报告 P(进入保持 | 安排停车)、P(验收通过 | 完成保持)、P(完整通过且停驻达标 | 安排停车)。分子分母按事件累加后相除，不再对空样本批次的零值做等权平均。最终验收继续使用 0.08 m/s、0.20 rad/s、三足承重和 80% 联合比例。

当前 failure 的权重为 -5，但普通 RewardManager 还会乘以 0.02 秒，所以实际一次失败只扣 0.1；completion_bonus 则做了步长补偿，一次确实加 5。这是应先明确的设计口径。如果预期失败代价是每事件 -5，应做事件补偿并重新评估代价预算；否则应修正文档和注释。不能把外部仓库的终止权重数字直接复制进来。

本项涉及 `scripts/validate_stairs_skills.py`、`mdp/stairs.py` 的 reset 诊断和事件奖励，以及 `bpx_stairs_base_env_cfg.py` 的 failure 配置。

**优先改进二：分别培养通过和停驻，保持最终任务验收。**

当前 `stair_completion_reward` 只有完整驶出且要求的停驻通过才给事件奖励；`StairMasteryCurriculum` 也用这套组合条件判断升降级。停驻为零时，大量已经会走的回合仍被计作课程失败，地形等级长期停留在低位；免停驻回合又可能偶然连续成功而升级。

建议将训练能力拆为通过能力与停驻能力：通过能力统计完整驶出、跟踪和非物理失败；停驻能力统计真正完成的停车及联合通过质量。最终完整任务仍要求两者达标，免停驻连续成功不应成为联合任务掌握的唯一依据。

可先保留 25% 平地回放，在低阶楼梯上分别采样连续通过、平坦支撑区停走和跨阶停走。停车难度从平坦区、低阶稳定支撑位置，逐渐扩展到首/中/末阶边缘；每次仍由真实行走到达，避免任意位姿跨阶重置。首次实验不必同时提高高度、逐阶随机性、停车位置难度和域随机化。

训练奖励可拆为驶出事件、完成合格停驻并重启的事件、完整任务附加事件。重新分配总预算，并保证每事件只结算一次，排除同帧物理失败。保持原有最终成功定义，以验证实际能力提高。上述分解是针对 BPX 症状提出的设计方案，不是声称所有参考仓库都有同样配方。

**优先改进三：在停车当下提供与联合验收相关的反馈。**

v3 已有独立的归一化线/角速度代价，因此不建议再加一套相同惩罚。缺少的是局部联合质量反馈和合格停车的即时事件反馈。可测试零指令期间的平滑稳定度奖励，例如将线/角速度指数核与支撑、漂移、直立因子结合，明确上限及启用条件；随后在完成保持并重启时结算单次事件。

需要同时审计现有项是否冲突：抑制关节运动应允许必要的补步和卸载调整；从行走转为静止时，可以对运动正则平滑加权，保持期间逐渐加强稳定约束。不要用强制回到平地默认关节角的方式替代跨阶平衡。宇树 mjlab 的 [stand_still 实现](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/mdp/rewards.py) 实际约束的是默认关节角偏差，因此名字相同也不代表它解决了跨阶静止问题。

出口奖励的反馈较晚：以现有 gamma=0.995、50 Hz 计算，10 秒后一个 +5 事件的直接折扣贡献约为 0.408。这个计算仅说明单个事件的折扣，不能替代对 Critic/GAE 效果的实测，但支持先测试局部反馈。即使改进奖励，也不能预先声称 stop_success 一定上升。

**优先改进四：首先清理 Critic 噪声，再增加特权信息。**

当前 Actor 和 Critic 都配置角速度、重力、关节位置及速度噪声，Critic 只比 Actor 多基座线速度。它缺少地形、接触和停驻计分状态，难以分辨看似相同的零指令观测将来能否领取成功奖励。宇树 mjlab 的 Critic 配置及上述盲爬楼论文提供了训练侧信息增强的直接参照。

低成本对照是只关闭 Critic 的观测噪声，保留 Actor 噪声、450/480 维度和全部奖励。下一步再在 Critic 加入当前/前方局部地形高度、足端高度和承重状态，以及停驻相位、计时、剩余时长、固定参考点偏差、已观测/已达标时长。之后才评估增益、延迟和摩擦等随机参数是否需要显式输入 Critic。

这些特权量仅进入训练时的价值网络，Actor 的 450 维部署接口可以保留。扩大 Critic 会改变模型维度；当前 `scripts/rsl_rl/finetune.py` 严格检查所有张量，要求 450/480，必须同步支持只迁移兼容 Actor、重建或部分初始化 Critic，并使用新优化器。不能只改观测后继续按旧接口恢复训练。

**后续改进五：审计奖励的实际尺度，分别约束运动和静止。**

当前 swing_clearance 是以米为单位的不足量平方。单脚缺 5 cm 时，权重 -2 对应的惩罚率只有 -0.005/秒。若诊断证实抬脚不足，可测试以 4 cm 等尺度归一化后重新选权重，保留当前地形相对查询、摆动脚和运动指令条件。将误差除以 0.04 后，平方尺度放大 625 倍，所以不能原封不动保留 -2。

当前 action_change=-0.01、torque=-1e-5 比一些参考的名义系数小，但公式、动作缩放、刚度和机器人不同。例如 Extreme Parkour 的 action_rate 使用 L2 范数，而 BPX 使用平方和，数值不能直接相比。只有确认停驻高频动作导致角速度超标后，才逐项测试动作变化、目标角二阶变化或关节运动代价。抑制过强会妨碍过阶和纠正平衡。

低速区也需要单独看：指令 vx=0.10 m/s、实际完全静止时，当前水平跟踪奖励仍为 2*exp(-0.1²/0.2²)=1.558/秒，约为满分的 77.9%。这不证明策略选择了不走，但表明其低速区的收益差值得量化。若出现卡阶，测试仅在要求移动、远离停车过程时启用的抗停滞反馈或归一化进度；当前已有带符号进度，不应重复添加一项相同奖励。不得在受令停车时处罚停滞。

脚步时间或接触节律可以作为运动阶段候选约束，零指令期间要关闭。当前设计没有 Actor 相位输入，直接搬用 WTW 或 mjlab 的时钟式步态奖励会改变原任务假设；若要引入，应作为独立实验，不能一边隐藏必要的时钟一边要求精确同步。

**后续改进六：阶段性扩大随机化，并正确处理探索恢复。**

现有摩擦、增益、延迟、零偏及观测噪声有价值。停驻尚未形成时，可在保持真实物理的条件下暂时缩小部分随机化范围，技能稳定后逐项扩大，并始终在完整随机化验证集上评估。Extreme Parkour 的配置为引入动作延迟设置了阶段时机，可作为渐进安排的例子；不能据此推出某个固定日程适合 BPX。[配置源码](https://github.com/chengxuxin/extreme-parkour/blob/main/legged_gym/legged_gym/envs/base/legged_robot_config.py)

mass/COM 和外力随机化可在后续加入，但范围应由 BPX 实测误差和使用条件确定。当前已经有的随机化不必全部同时增强。

当前 checkpoint 动作标准差均值约 0.03325，已低于初始 0.08。weights-only 微调函数会重新写入新配置的标准差；若继续使用 0.08，探索标准差将约为当前值的 2.41 倍。因此重启实验时应显式记录和控制 std。只有配对回放显示采样动作明显更差时，才优先考虑降低或按零指令调节探索；状态相关 std 必须同时体现在 PPO 的采样分布、log probability、熵和更新中。

**后续改进七：再考虑历史编码、状态估计或教师—学生。**

WTW 的适应模块与 Extreme Parkour 的历史编码值得参考，但“历史输入”不等于已经训练了状态估计器。BPX 的 10 帧历史跨度约 0.18 秒，可先保留接口测试带速度/接触辅助目标的历史编码，再考虑更长窗口或 GRU。更长历史、循环网络、教师蒸馏都需要单独衡量收益和部署成本；特权教师不能保证盲学生能推断接触前不可观测的台阶。

实现上，BPX 的观测是按 term 拼接各自历史的布局。若使用时序编码器，需要先恢复为逐时间帧排列；直接把 450 维 reshape 成 10×45 会错配信号。Critic 优化也不要求同步放大 Actor 网络，当前 [128,128,128] 可以先作为控制组。

地形方面，可先训练每梯段等高等深的规则楼梯，再渐进增加逐阶高度和深度变化，验证集保留当前独立随机楼梯。之后用未见地形种子、目标 15 cm 高度、上下楼及停车位置分层验证。几何库变大不能替代训练分布与验证集的分离。

**部署还有一个具体的不一致需要后续处理。**

训练让楼梯 Actor 在零指令时继续控制。当前 `sim2real/bpx_sim2real/controller.py` 在进入 stand 后改用独立 stand 策略；`bpx_runtime/routing.py` 在低指令持续后会进入 stopping，再进入 stand。后续楼梯部署模式应让同一楼梯 Actor 执行减速、静止和重启，并保持相应历史连续。否则验证到的是另一套控制组合，无法直接检查训练中的停驻技能。

**建议的实验顺序如下。** 所有变体固定起点、环境数、训练 transition 预算、验证清单及最终验收；先用一个训练种子筛选，再以多个独立训练种子复核，而不是把不同评估场景种子当作训练重复。

| 顺序 | 对照内容 | 主要判断 |
| --- | --- | --- |
| E0 | 原行走与当前楼梯模型的成对回放，建立事件级基线 | 停驻失败来自到达率、均值策略、采样扰动还是接触条件 |
| E1 | 当前训练仅改为无噪声 Critic | 在维度和奖励不变时，价值估计与停驻精度是否改善 |
| E2 | 在 E1 上增加 Critic 的地形、接触和计分状态 | 额外训练信息能否提高样本效率；部署 Actor 保持原输入 |
| E3 | 在同一起点单独测试停驻局部质量/事件反馈 | 停驻是否在较短训练预算内产生非零成功 |
| E4 | 在同一起点单独测试通过/停驻课程分解与位置难度安排 | 是否解除低等级停驻造成的课程阻塞，同时保留严格联合验收 |
| E5 | 组合有效改动，再逐项测试净空尺度、动作平滑及随机化日程 | 收益能否泛化到未见楼梯，且不牺牲通过能力 |

failure 事件预算如需改变，应先作为独立对照或明确记录为所有新基线共同变更，不与 E1 的单因素对照混用。选模型主要看事件加权的上下楼通过、严格停驻、重启、完整任务成功及未见地形表现，累计 mean reward 只作辅助。本文建议均需 BPX 仿真验证，目前没有执行新训练或证实性能提升。
