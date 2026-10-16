# 从零学习上下楼梯与平地混合课程

任务保持 `BPX-Locomotion-Stairs-History-v0`，使用 H=10 非对称历史观测接口：Actor 450 维、Critic 480 维、10 帧历史；动作、关节顺序和随机出生朝向不变。

先看根目录 [README 的工程导读](README.md#当前上下楼梯工程怎么读) 理解文件分工；本页说明训练参数与操作。MuJoCo 场景和回放见 [楼梯 sim2sim](sim2sim/STAIRS_SIM2SIM.md)。

## 开训前检查与依赖安装

在项目根目录、已激活的 Isaac Lab Python 环境运行：

```bash
python scripts/check_dependencies.py               # 快速检查依赖、驱动、CUDA 与 USD
bash scripts/install_training_deps.sh              # 一键补齐当前 Isaac Lab 环境
python scripts/check_dependencies.py --smoke-test  # 4 环境、两轮 PPO、核对模型保存
```

检查失败返回 `1`，通过返回 `0`。真实验证的日志和 checkpoint 单独保存在 `outputs/training_env_check/check_*/`；通过只说明小规模训练链路可运行，不能保证 512/1024 环境的显存足够。新机器从零安装使用 Python 3.11 独立环境和 `bash scripts/install_training_deps.sh --full`；参数、版本与资产准备见 [README 环境准备](README.md#环境准备)。

## 回合分配与防遗忘

每次 reset 独立采样地形类别，整个回合不切换：

| 类别 | 新回合采样概率 | 指令 | 初始难度 |
| --- | ---: | --- | --- |
| 上楼 `up` | 初始 30%，平地达标后 40% | 初始只前进，达标后前进/后退各 50% | 所有模式第 0 级，约 2–3.625 cm |
| 下楼 `down` | 初始 30%，平地达标后 40% | 初始前进，达标后前进/后退各 50%，vy=wz=0 | 第 0 级，3–5 cm 台阶，0.15–0.30 m/s |
| 平地 `flat` | 初始 40%，平地达标后 20% | terrain 行走速度分布；其中 10% 为 `(0,0,0)` | 固定第 0 级，无台阶 |

上楼前进 0.3–0.6 m/s、后退 0.2–0.4 m/s；下楼前进/后退使用相同的分级速度幅值，后退取负 vx。上下楼均不采样侧走或转弯，因此不再出现恒定转弯轨迹无法到达外圈的问题。平地仍从 ±1.0/±0.5/±1.0 的 terrain 分布采样，保留侧走和旋转。后退指机器人倒着移动，上楼/下楼由地形决定。

比例是**新回合数的采样概率**；下楼回合较长时，运行中的下楼环境比例和采集步数占比会高于 40%。平地回合内部沿用 terrain 行走策略的 10% 静止、25% 纯偏航、65% 三轴混合分布；持续保留上楼和平地数据用于减轻遗忘，仍须检查它们的成功率和实际回放。

回合上限和指令保持时间改为 **40 秒**。在 0.15 m/s 和随机朝向下，20 秒不足以从中央走完到外圈平台。训练入口关闭楼梯任务的随机初始 episode 长度，避免刚出生便因随机计时超时。

## 场地与独立等级

每块地形 10×10 m，中央实际平台 2×2 m。台阶有竖直踢面，踏面深度始终为 **20 cm**；11 个踢面在方形半径 1.0、1.2、……、3.0 m。外圈平台从半径 3.0 m 延伸至 5.0 m，为稳定落地留出空间。

- 上楼：中央低位平台，向外逐级升高，使用 Isaac Lab 的倒金字塔楼梯。
- 下楼：中央高位平台，向外逐级降低，使用正金字塔实体楼梯。出生站高相对中央平台为 0.40 m。
- 平地：整块水平地面。
- 地形库为 8 行、6 列，列顺序固定为上楼×2、下楼×2、平地×2。几何列数比例与回合采样概率分别配置。
- 上楼单级高度保持原来的 2–15 cm 范围；第 0 行约 2–3.625 cm，最高行约 13.375–15 cm。

每个环境维护 `上楼/下楼/平地 × 直行/侧走/后退/转弯` 的等级和连续结果。上下楼仅使用直行和后退槽位，各自独立升级；平地始终为零级。切换类别和移动模式时恢复相应的难度，reset 先统计旧回合，再设置新地形和出生 origin，最后更新指令。

下楼的高度和速度随自己的等级共同提高：

| 下楼等级 | 单级高度范围 | vx 幅值范围（m/s，后退取负） |
| ---: | --- | --- |
| 0 | 3.00–5.00 cm | 0.150–0.300 |
| 1 | 4.43–6.43 cm | 0.171–0.343 |
| 2 | 5.86–7.86 cm | 0.193–0.386 |
| 3 | 7.29–9.29 cm | 0.214–0.429 |
| 4 | 8.71–10.71 cm | 0.236–0.471 |
| 5 | 10.14–12.14 cm | 0.257–0.514 |
| 6 | 11.57–13.57 cm | 0.279–0.557 |
| 7 | 13.00–15.00 cm | 0.300–0.600 |

同一块台阶的各级高度一致；同一行的两列在高度带内随机取值。**低速区间属于新增训练分布，加载旧模型后不意味着立即能稳定完成。**

同环境、同类别、同模式连续 **2 次成功升一级**，连续 **2 次未完成降一级**；最高级封顶，最低级保底。越界但未落稳、耗尽时间、物理跌倒、跟踪不合格都算未完成，不再把未落稳越界作为中性回合。

## 楼梯稳定落地与平地整回合质量

先满足以下条件，并**连续保持 0.4 秒**：

1. 机身到达外圈平台，方形半径至少 3.5 m、尚未越过 4.5 m 截断边界。
2. 四足都已越过最后一级台阶：每足方形半径至少 3.05 m，且仍在边界内。
3. 至少两足的即时竖直接触力各大于 5 N，机身高度距外圈平台默认站高的绝对误差不超过 0.15 m。
4. 总倾角不超过 0.35 rad（约 20°），竖直速度绝对值不超过 0.2 m/s，横滚/俯仰合角速度不超过 1 rad/s。

允许机器人稳定走过平台，不要求停住或四足同时支撑。条件中断会将连续落地时间清零；跳过边界、腾空经过目标区域、仍跨着最后一级台阶都不能通过。

回合成功还要求：无物理失败；楼梯区域累计活动至少 1 秒（平地豁免）；平均偏航跟踪误差 ≤0.5 rad/s；沿指令方向的实际平均速度 ≥目标平均速度的 60%；平均水平跟踪误差上楼/平地 ≤0.35 m/s、下楼 ≤0.10 m/s。后两项避免低速指令下原地停住也因绝对误差小而晋级。

楼梯落稳会结束回合；4.5 m 越界仍会截断回合以避免进入相邻地块，但不会自动获得成功。

平地使用**整回合稳定质量**，不再要求结束前恰好连续稳定 1 秒。稳定的单步定义仍使用上面的至少两足支撑、合理高度、倾角、竖直速度及横滚/俯仰角速度条件；平地无需到达外圈平台。

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `flat_quality_warmup` | 1.0 s | 复位后首秒仅豁免稳定质量统计，让身体完成初始过渡 |
| `flat_min_quality_time` | 2.0 s | 首秒之后至少观察 2 秒，避免短回合轻易通过 |
| `flat_min_stable_fraction` | 0.90 | 有效观察期间，所有稳定条件同时通过的时间占比至少 90% |
| `flat_max_unstable_time` | 0.5 s | 有效观察期间最长一次连续不稳定不能超过 0.5 秒 |

稳定占比为 `累计稳定时间 / 有效观察时间`，单次超限只减少占比，不清空之前的成绩；最长失稳时长限制防止较长回合用大量正常时间掩盖一段持续失稳。例如有效观察 10 秒、其中两次各 0.04 秒超限，稳定占比 99.2%，可通过稳定质量检查；连续失稳 0.6 秒，即使稳定占比超过 90% 也不能通过。阈值是当前课程初始设置，后续应结合分项日志和回放校准。

平地持续训练至越界、40 秒时限或物理失败，质量达标不会提前终止。物理失败仍直接否决成功，包括首秒；速度跟踪及专项约束仍按**整个回合**计算，不豁免首秒。对平均目标偏航幅值 >0.1 rad/s 的回合，沿目标方向累计实际转角须达到目标的 60%，累计绝对角速度误差不得超过目标转角的 50%；不转、反转都不能通过纯旋转任务。零指令站立回合的平均线速度误差和角速度误差均须 ≤0.1（分别为 m/s、rad/s）。普通移动的水平/偏航平均误差和沿指令 60% 进度要求保持不变。

楼梯仍须在合法平台连续落稳 0.4 秒，升降级计数及方向/采样门槛保持不变。平地的新成功结果同时供方向门槛和平地采样比例门槛使用。

## 从零训练

```bash
conda activate isaaclab
source /home/hanguang/Isaac_Sim/setup_conda_env.sh
python scripts/rsl_rl/train.py --task BPX-Locomotion-Stairs-History-v0 \
  --num_envs 512 --max_iterations 10000 --headless
```

不传 `--finetune` 或 `--resume`，从随机权重和零级直行开始。10000 轮是初始训练预算，不保证学完所有阶段。当前配置先使用 30/30/40 的上楼/下楼/平地采样概率；平地移动和站立各自连续两个达标窗口后，课程自动切换到 40/40/20。

`sim2sim/config/bpx_stairs_history.toml` 是部署回放配置，其中 15 cm 台阶和模型路径不控制训练课程。

## 微调、续训与回放

在已有 Isaac Lab 环境中，从仓库根目录执行，显式指定微调来源：

```bash
conda activate isaaclab
source /home/hanguang/Isaac_Sim/setup_conda_env.sh
python scripts/rsl_rl/train.py --task BPX-Locomotion-Stairs-History-v0 \
  --finetune logs/rsl_rl/bpx_stairs_history/2026-10-16_01-21-10/model_5999.pt \
  --num_envs 512 --max_iterations 3000 --headless
```

只有显式传入 `--finetune 模型路径` 才会**加载 Actor/Critic 和策略标准差权重**，新建优化器、迭代计数从 0 开始。未传 `--finetune` 或 `--resume` 时，从随机权重开始运行当前混合课程；代码不预设 checkpoint 路径。初始学习率为 `3e-4`，使用 adaptive 调度，不恢复旧优化器，也不重新把策略标准差设为 1。新目录后缀默认 `up_down_flat_from_scratch`，不会覆盖旧训练目录。上面的命令表示新增 3000 轮。

更换微调来源只需指定模型路径：

```bash
python scripts/rsl_rl/train.py --task BPX-Locomotion-Stairs-History-v0 \
  --finetune logs/rsl_rl/bpx_stairs_history/YOUR_RUN/model_XXXX.pt \
  --num_envs 512 --max_iterations 3000 --headless
```

继续**新的混合任务**，恢复优化器、迭代号和上下楼课程：

```bash
python scripts/rsl_rl/train.py --task BPX-Locomotion-Stairs-History-v0 \
  --resume --checkpoint logs/rsl_rl/bpx_stairs_history/YOUR_MIXED_RUN/model_XXXX.pt \
  --num_envs 512 --max_iterations 1000 --headless
```

`--resume` 与 `--finetune` 互斥。恢复课程状态时应保持 `num_envs` 和统计窗口大小一致；改变这些参数请用 `--finetune`，重建课程但保留权重。版本 2 的前进/后退混合课程支持自动迁移至当前版本 3；更早的旧楼梯课程仍需通过 `--finetune` 建立新课程。

Isaac Sim 回放使用当前环境配置，`play.py` 只加载策略，不自动恢复保存的地形等级和方向阶段。默认从低难度前进开始，因此不能用默认回放代表高等级或后退能力。

```bash
python scripts/rsl_rl/play.py --task BPX-Locomotion-Stairs-History-v0 \
  --checkpoint /path/to/model.pt --num_envs 16 --real-time
```

MuJoCo 通过独立 TOML 设置场景；更换模型只需更新策略路径，不必新增逐 checkpoint 的配置。

## 观察哪些曲线

前缀为 `Curriculum/terrain_levels/`，分别查看 `up/`、`down/`、`flat/`：

| 后缀 | 含义 |
| --- | --- |
| `success_rate` | 楼梯稳定落地／平地整回合稳定质量、速度跟踪、有效运动且无物理失败的比例 |
| `timeout_rate` | 耗尽 40 秒的比例，不包含场地边界截断 |
| `failure_rate` | 物理失败终止比例；未落稳越界可能不是物理失败，但成功率会下降 |
| `actual_speed` | 沿指令方向的回合平均速度，单位 m/s |
| `command_speed` | 目标水平速度模长的回合平均值 |
| `speed_xy`、`velocity_x`、`velocity_y` | 速度模长及机身坐标系有符号速度 |
| `terrain_level` | 已完成回合实际使用的等级均值 |
| `stored_terrain_level` | 所有环境缓存的该类别课程等级均值；上下楼均统计前进和后退槽位 |
| `active_fraction` | 当前环境中该地形类别的比例，与新回合采样概率可能不同 |
| `window_samples`、`episode_count` | 有效窗口样本数与累计已完成回合数；零样本时统计零值不表示失败 |

环境还计算 `flat/stable_fraction`（每回合稳定占比的窗口均值）、`flat/max_unstable_time`（每回合最长连续失稳秒数的窗口均值）、`flat/quality_time`（每回合有效观察秒数的窗口均值），以及各模式细分。不能把 `max_unstable_time` 的窗口均值理解成窗口内最坏回合，也不能仅凭均值反推出通过率。默认 TensorBoard 只保留前两项平地汇总质量指标，更多细分在完整日志模式下查看。

### TensorBoard 默认精简为26条曲线

楼梯训练的 `agent.tensorboard_scalar_filter` 控制写入项，其他任务保持原来的日志设置：

| 保留内容 | 完整tag或后缀 | 条数 |
| --- | --- | ---: |
| 三类任务完成与运动表现 | `Curriculum/terrain_levels/{up,down,flat}/{success_rate,failure_rate,timeout_rate,actual_speed,command_speed}` | 15 |
| 上下楼难度 | `Curriculum/terrain_levels/{up,down}/terrain_level` | 2 |
| 课程阶段 | `Curriculum/terrain_levels/{command_stage,terrain_sampling_phase}` | 2 |
| 平地质量 | `Curriculum/terrain_levels/flat/{stable_fraction,max_unstable_time}` | 2 |
| 回报和回合长度 | `Train/{mean_reward,mean_episode_length}` | 2 |
| PPO健康 | `Policy/mean_noise_std`、`Loss/value_function`、`Loss/learning_rate` | 3 |

重复的方向细分、恒0平地等级、逐项奖励、速度分量、统计计数、Perf和按时间重复记录的曲线不再写入默认TensorBoard。过滤只发生在日志写入层，环境统计、奖励、训练更新及checkpoint内容不受影响；终端仍可输出完整诊断值。需要记录完整曲线时，在训练命令最后添加 `agent.tensorboard_scalar_filter=null`，从该次运行开始生效。此设置只过滤TensorBoard，W&B/Neptune保持其原有行为。

已有日志可创建独立精简快照，原始event文件保留：

```bash
python scripts/filter_tensorboard_logs.py \
  --source logs/rsl_rl/bpx_stairs_history \
  --output logs/tensorboard_stairs_core \
  --filter-file outputs/stairs_reward_analysis/tensorboard_core_filter.txt
tensorboard --logdir logs/tensorboard_stairs_core
```

本次已生成 `logs/tensorboard_stairs_core`。快照不会自动跟随源日志增长；再次生成时须选择新的输出目录，脚本拒绝覆盖已有目录。旧运行尚未写出的新指标不会凭空补齐；只按同一保留规则筛选已有数据。原TensorBoard若继续指向旧日志目录，旧曲线仍会显示，可以使用更新后的 `tensorboard_core_filter.txt` 在tag过滤框中隐藏它们。新训练/续训使用精简写入后，其新日志直接显示核心曲线。

另外保留 `up/forward/success_rate`、`up/backward/success_rate`、`flat/sideward/actual_speed`、`down/forward/terrain_level` 等细分日志。每个类别×模式保留最近 256 个回合；类别总指标合并各模式窗口，不能理解为按当前采样概率加权的全局指标。日志仅统计已完成回合，速度先按回合实际时长归一化；训练器还会对一轮内日志求均值。

方向课程只有两个阶段：stage 0 上下楼均只前进；stage 1 上下楼均为前进/后退 = 50/50%，之后不再开放其他方向。平地从一开始保留所有方向。

晋级要求上楼前进、下楼前进以及平地四种统计模式各累计至少 128 个当前阶段回合，且各自成功率 ≥80%，连续两个互不重叠的统计窗口达标。楼梯仅使用等级 ≥1 的回合验证方向能力，平地不受等级限制。旧阶段的在途回合不计入新门槛；后退从自己的零级开始。`command_stage`、`command_good_windows` 记录进度，`command_sideward_ramp` 保留为兼容日志字段并固定为 0。checkpoint 保存并恢复方向阶段、窗口计数及各模式等级。

本次课程状态版本为 **3**。`--resume` 版本 2 时保留 Actor/Critic、优化器、迭代号（训练器负责恢复）、上下楼等级及其统计/连续结果、已解锁方向和采样阶段；自动清空旧平地结果与稳定质量统计、平地连续结果、平地采样考核窗口以及方向晋级考核窗口，避免混用两种成功口径。迁移不修改原 checkpoint，下一次保存写出 v3。v3 检查点正常恢复完整课程状态；更早版本或无版本状态仍需 `--finetune`。

已经启动的训练进程不会因修改文件而切换判据。新规则需要重新启动训练或从 checkpoint 续训后生效；本次代码修改不会停止或重启已有进程。


重点检查下楼成功率与实际速度是否随等级提高，以及上楼、平地成功率是否下降。混合训练是防遗忘措施，实际步态和迁移效果须通过正式训练、逐足日志与回放确认。

## 逐足换步约束

为减少单腿长期卸载、长期贴地不换步的三足解，楼梯任务增加两项持续成本：

| 奖励项 | 权重 | 无惩罚时间 | 线性渐增区间 | 单足成本上限 |
| --- | ---: | ---: | --- | ---: |
| `feet_air_time_excess` | −0.1 | 连续腾空 ≤0.6 s | 0.6–1.0 s | 1 |
| `feet_contact_time_excess` | −0.1 | 连续接触 ≤1.0 s | 1.0–1.5 s | 1 |

每项原始成本为 `sum(clamp((每足连续时间 − 容忍时间) / ramp_time, 0, 1))`，先逐足截断，再对四足求和。单足超过上限后保持成本，不随秒数无限放大。例如连续腾空 0.8 秒时单足成本约 0.5，1.0 秒后为 1；乘 −0.1 权重后分别为 −0.05、−0.1，RewardManager 再按策略步长积分。

只在指令水平速度模长 **>0.1 m/s** 时启用，覆盖直行、左右侧走、后退及前进转弯；静止与纯偏航指令豁免。门控使用目标速度，因此有移动指令但实际卡住时，长期不换步仍有成本。

复用 `contact_forces` 的 `current_air_time` 和 `current_contact_time`，要求 `track_air_time=True`（当前已启用）；接触力阈值沿用传感器默认的 1 N。每只脚接触/离地时独立更新对应连续时间，episode reset 清零。无需等待再次落地便能惩罚持续悬空，四足可以自行选择步态相位。

配置位于 `BpxStairsHistoryRewardsCfg`，可分别调整 `max_air_time`、`max_contact_time`、`ramp_time` 和权重。当前阈值高于诊断中常见的短腾空周期，并为上楼持续支撑预留 1 秒；它们是初始训练参数，需结合新模型的逐足换步与登顶表现再评估。

TensorBoard 分别记录 `Episode_Reward/feet_air_time_excess`、`Episode_Reward/feet_contact_time_excess`。这两个日志是各自四足总和的加权回合统计。原有 `feet_air_time` 落地奖励保持阈值 0.4 s、权重 0.5；本次增加持续约束。新奖励须在重新训练或继续训练后才可能改变策略，单纯回放旧 checkpoint 不会修正步态。

## 验证

```bash
python -m unittest discover -s scripts -p 'test_*stairs_curriculum.py' -v
python scripts/validate_stairs_task.py --headless --device cpu \
  --finetune logs/rsl_rl/bpx_stairs_history/2026-10-16_01-21-10/model_5999.pt
```

真实仿真验证包含原观测/动作接口、台阶几何、顶部出生、独立难度、旧回合日志归属、有限物理步进、指定 checkpoint 的权重一致性、两轮 PPO 更新与新课程 checkpoint 恢复。验证只证明代码流程和数值可运行，不代表下楼技能已训练完成。
