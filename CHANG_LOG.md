# 修改日志

本文档用于记录项目每次代码修改的设计思路、核心改动点和验证方式。后续修改代码时，请继续在本文档中追加记录，便于回顾训练效果、定位问题和说明实验依据。

## 记录格式

```text
## YYYY-MM-DD - 修改标题

### 修改思路
- 为什么要做这次修改。
- 希望解决什么训练、模型、奖励、特征或工程问题。

### 核心修改点
- 改动了哪些文件或模块。
- 关键逻辑、配置、接口或行为变化。

### 验证方式
- 执行过的命令、测试结果或未验证原因。

### 备注
- 风险、兼容性问题、后续优化方向。
```

## 2026-06-23 - v1.2a 训练回落后的保守拆塔调参

### 修改思路
- 从 `logs/tencentarena.com_Archive [26-06-23 11-29-48].har` 提取到 v1.2a 末段训练指标：common_ai 胜率最高到过约 0.78，但末端回落到约 0.47-0.50，未达到 v1.1 `0.84` 的目标线。
- 指标显示击杀、伤害和推塔都有提升，死亡较低，但终局 `win_result` 长期为负，说明策略没有稳定把优势转化为拆塔胜利。
- `unsafe_dive` / `unsafe_dive_severity` 惩罚长期偏大，且原实现用同一个距离阈值同时表示“可推塔窗口”和“危险越塔范围”，容易把接近敌塔推进也压成风险行为。

### 核心修改点
- 在 `agent_ppo/conf/conf.py` 中把训练改得更保守：
  - 学习率从 `5e-4` 降到 `3e-4`，目标学习率从 `1e-4` 降到 `5e-5`。
  - 模型保存间隔从 1800 秒缩短到 600 秒，避免错过中途峰值 checkpoint。
- 调整 reward 权重，使目标更偏向“低风险拆塔”：
  - 提高 `enemy_tower_hp_down`、`tower_destroy`、`push_window_tower_damage`、`win_result` 和 `timeout_tower_gap`。
  - 降低 `self_tower_hp_down`、`kill`、`money`、`exp`、`death`、`unsafe_dive` 和 `unsafe_dive_severity`，减少保守惩罚和换血/击杀对目标的干扰。
- 在 `agent_ppo/feature/reward_process.py` 中拆分距离阈值：
  - `PUSH_WINDOW_DISTANCE = 9000.0` 保留为安全推塔窗口判断。
  - `DIVE_DANGER_DISTANCE = 7000.0` 用于危险越塔惩罚，避免过早惩罚接近敌塔。
- 更新 `utils/v1_2_preflight.py` 和相关测试，使 preflight 对齐新的 v1.2 默认配置。
- 更新 `docs/v1.2-runbook.md` 和 `docs/v1.2-implementation-plan.md`，说明 v1.2a 回落后默认采用 `3e-4` 并更频繁保存 checkpoint。

### 验证方式
- 待运行：
  - `python3 utils/v1_2_preflight.py --md /tmp/hok_v1_2_preflight_after.md --csv /tmp/hok_v1_2_preflight_after.csv`
  - `python3 -m unittest discover -s tests -p 'test_ppo_optimization.py'`
  - `python3 -m unittest discover -s tests -p 'test_v1_2_preflight.py'`

### 备注
- 这次不建议直接进入 v1.2b；应先用新配置重跑 v1.2a，并重点观察 common_ai 胜率是否不再峰值后回落，以及敌塔血是否继续下降。

## 2026-06-21 - v2 风险敏感 PPO 与对手课程

### 修改思路
- `results/v1` 中 common_ai 评估胜率能上升但后期平台化，self-play 胜率长期震荡，说明策略有一定学习能力但泛化和稳定性不足。
- 击杀、伤害上升的同时死亡也明显上升，说明上一版奖励把进攻意愿拉起来了，但没有足够惩罚低血、敌塔附近和敌人贴近时的高风险进攻。
- 本轮不继续简单叠加奖励项，而是引入风险敏感奖励、势函数差分塑形、课程化对手和轻量时序差分特征，让训练信号更稳定。

### 核心修改点
- 在 `agent_ppo/conf/conf.py` 中调整奖励权重：
  - `damage_to_enemy` 从 2.0 降到 1.0，避免过度鼓励无脑换血。
  - `kill` 从 8.0 降到 5.0，`death` 从 -8.0 加强到 -10.0。
  - 新增 `risk_exposure`，用于惩罚低血时靠近敌塔或敌方英雄。
  - 新增 common_ai 到 self-play 的课程调度参数。
- 在 `agent_ppo/feature/reward_process.py` 中加入风险敏感奖励：
  - 新增 `risk_exposure` 计算，低血、靠近敌塔、靠近敌方英雄时风险升高。
  - 对 `damage_to_enemy` 加入风险门控，高风险状态下对敌伤害奖励会被打折。
  - 将 `retreat_low_hp` 和 `under_enemy_tower` 改为势函数差分形式，减少每帧固定塑形导致的刷分倾向。
- 为兼容平台旧 checkpoint，保留 50 维模型输入：
  - 不再将特征维度扩展到 58，避免 `concat_mlp_fc1.weight` 从 `(256, 50)` 变成 `(256, 58)`。
  - 本轮主要保留风险奖励、势函数差分、课程化对手和行为监控，不改变模型第一层 shape。
- 在 `agent_ppo/workflow/train_workflow.py` 中加入训练对手课程：
  - 训练局前期更高概率对战 `common_ai`，之后逐步过渡到 self-play。
  - 评估局仍保持环境配置原逻辑，避免评估曲线不可比。
  - 新增低血敌塔暴露比例、平均风险暴露、塔下死亡次数的行为统计。
- 在 `agent_ppo/conf/monitor_builder.py` 中新增行为诊断面板。
- 在 `tests/test_ppo_core.py` 中扩展测试：
  - 覆盖 50 维特征输出。
  - 覆盖模型第一层保持 `(256, 50)`，兼容旧 checkpoint。
  - 覆盖风险暴露奖励。
  - 覆盖 common_ai 比例随 episode 衰减。

### 验证方式
- 待运行 `python tests/test_ppo_core.py` 验证配置、特征、奖励、课程调度和 PPO learn 基础路径。

### 备注
- 本轮仍未真正启用 LSTM 主干，也不再改变模型输入维度；如果 v2 仍平台化，再考虑新开无旧 checkpoint 的 recurrent PPO 或 population self-play。
- 风险项权重需要结合后续训练曲线继续微调，重点观察死亡数、低血敌塔暴露比例和 `hurt_to_hero / hurt_by_hero`。

## 2026-06-21 - PPO 主线特征、奖励和训练稳定性优化

### 修改思路
- 原始 PPO 版本的状态表达较弱，特征维度只有 10 维，主要包含己方英雄和防御塔的少量信息，难以支撑 1v1 场景中的换血、撤退、压塔、防守等复杂决策。
- 原始奖励函数较单一，主要依赖防御塔血量和前进奖励，容易让智能体只学习推塔或盲目前进，缺少击杀、死亡、伤害、承伤、低血回撤和塔下危险等训练信号。
- 原始配置中样本维度、特征维度和动作维度大量手写，扩展特征或动作后容易出现 shape 不一致。
- PPO loss 中存在一些设备和数值稳定性细节问题，例如直接创建 CPU tensor、advantage 未归一化、学习率调度调用方式不够规范。
- 需要增加基础测试，保证特征输出、奖励项、样本维度和一次 PPO learn 调用能跑通。

### 核心修改点
- 在 `agent_ppo/conf/conf.py` 中扩展奖励权重：
  - 保留 `tower_hp_point` 和 `forward`。
  - 新增 `hero_hp_point`、`damage_to_enemy`、`damage_taken`、`kill`、`death`、`retreat_low_hp`、`under_enemy_tower`。
  - 奖励设计从“只关注推塔”扩展到“推塔 + 换血 + 生存 + 击杀死亡 + 位置安全”。
- 在 `agent_ppo/feature/reward_process.py` 中重构奖励计算：
  - 增加 `safe_div` 和 `distance`，避免除零和缺失字段导致异常。
  - `damage_to_enemy` 通过敌方英雄血量下降计算正奖励。
  - `damage_taken` 通过己方英雄血量下降计算惩罚项，权重为负。
  - `kill` 和 `death` 使用英雄是否死亡作为终局/强信号。
  - `retreat_low_hp` 鼓励低血时靠近己方塔。
  - `under_enemy_tower` 惩罚靠近敌方塔，低血时惩罚更强。
- 在 `agent_ppo/feature/feature_process/__init__.py` 中扩展特征：
  - 将特征维度从 10 扩展到 50。
  - 新特征覆盖己方英雄、敌方英雄、双方防御塔、双方相对距离、血量差、等级差、推进进度、塔下风险、帧进度和技能可用性。
  - 增加特征长度断言，确保输出维度与 `Config.FEATURE_DIM` 一致。
- 在 `agent_ppo/conf/conf.py` 中改造维度配置：
  - 新增 `FEATURE_DIM` 和 `REDUCED_LEGAL_ACTION_DIM`。
  - `DATA_SPLIT_SHAPE`、`SERI_VEC_SPLIT_SHAPE`、`data_shapes`、`SAMPLE_DIM` 改为由配置自动推导，减少手写 shape 的维护成本。
- 在 `agent_ppo/model/model.py` 中增强 PPO loss 稳定性：
  - 根据模型输出所在 device 创建 tensor，减少 CPU/GPU device mismatch 风险。
  - 对 advantage 按训练帧做归一化。
  - 用 `torch.clamp(..., min=1.0)` 防止分母过小。
  - 清理重复的 value loss 计算。
  - 使用 `nn.init.orthogonal_` 初始化线性层权重。
- 在 `agent_ppo/agent.py` 中修复设备转换：
  - 推理输入 tensor 显式放到 `self.device`。
  - 模型输出转 numpy 前先 `detach().cpu()`，兼容 GPU 推理。
- 在 `agent_ppo/algorithm/algorithm.py` 中调整训练返回值：
  - `scheduler.step()` 使用标准调用方式。
  - `learn()` 返回包含 `total_loss` 等指标的结果，方便测试和监控。
- 在 `agent_ppo/workflow/train_workflow.py` 中增加奖励分项累计：
  - 每局累计各奖励子项。
  - 监控上报时增加 `reward_xxx`，便于判断训练信号来自哪一类奖励。
- 新增 `tests/test_ppo_core.py`：
  - 检查配置 shape 一致性。
  - 检查特征输出维度和取值范围。
  - 检查奖励管理器输出所有奖励项。
  - 使用 fake sample 验证一次 PPO learn 可以正常执行。
- 新增 `planning/suggestion.md`：
  - 记录项目结构分析、主要问题和后续优化建议。

### 验证方式
- 最近提交中新增了 `tests/test_ppo_core.py`，用于覆盖配置、特征、奖励和 PPO learn 的基础路径。
- 本次只补充修改日志，没有重新运行测试。

### 备注
- 当前优化重点是让奖励和特征更符合 1v1 对抗任务，但奖励权重仍需要通过实际训练曲线和对局回放继续调参。
- `Model` 中虽然保留了 LSTM 接口，但策略和 value 主干仍主要依赖当前帧特征；后续如果要真正利用 16 帧序列，需要进一步改造模型 forward。
- 新增特征依赖环境观测字段，不同环境版本字段名可能不完全一致；目前代码已对部分字段做兼容和默认值处理，但仍建议结合真实 observation 抽样检查。
