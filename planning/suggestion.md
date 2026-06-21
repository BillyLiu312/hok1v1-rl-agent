代码结构

入口是 train_test.py:1，通过 run_train_test() 选择 ppo 或 diy。算法注册在 conf/algo_conf_hok1v1.toml:1，应用级 buffer、batch、模型保存等配置在 conf/configure_app.toml。

agent_ppo 是核心实现：

- agent_ppo/workflow/train_workflow.py:28：训练主循环。负责创建环境配置、循环跑 episode、调用 agent 预测、收集帧、送样本、定期保存模型。
- agent_ppo/agent.py:1：智能体封装。负责模型推理、动作采样、合法动作 mask、模型保存加载、状态维护。
- agent_ppo/feature/definition.py:1：特征、奖励和样本处理。FeatureProcess 生成 10 维特征；FrameCollector 做 GAE、拼接 16 帧序列样本。
- agent_ppo/model/model.py:21：PPO 网络和 loss。输入 10 维特征，输出 6 个动作头和 value。
- agent_ppo/algorithm/algorithm.py:1：学习步骤。把样本拆分成 tensor，前向、算 loss、反传、梯度裁剪、scheduler step、监控上报。
- agent_ppo/conf/conf.py:11：PPO 维度、动作空间、学习率、GAE 参数、奖励权重。

agent_diy 目前不是完整算法：model.py 只有空网络，algorithm.learn() 是 pass，agent.reset() / _model_inference() / observation_process() 都是模板逻辑。因此除非你准备自己实
现 DIY，否则主要维护对象应是 PPO。

训练数据流

一局游戏开始后，workflow() 创建 EpisodeRunner，每帧调用 agent.predict() 或 agent.exploit()。agent.predict() 先做 observation_process()，再走 _model_inference() 得到动作。
训练模式下，build_frame() 把特征、动作、旧概率、value、reward、LSTM 状态存进 FrameCollector。局末 FrameCollector.sample_process() 计算 GAE，把 16 帧拼成一个训练样本，再由
send_sample_data() 交给 learner。learner 侧调用 Algorithm.learn() 做 PPO 更新。

主要问题

1. Model 定义了 LSTM，但实际没有使用。
    agent_ppo/model/model.py:56 创建了 LSTM，forward 里只是把传入的 hidden/cell 原样保存并返回，策略和 value 直接来自 concat_mlp(feature_vec)，见 agent_ppo/model/
    model.py:81。这意味着现在虽然样本按 16 帧组织，但模型本身是无记忆 MLP。

2. 特征过少，状态表达很弱。
    配置里 DimConfig.DIM_OF_FEATURE = [10]，SERI_VEC_SPLIT_SHAPE = [(10,), (85,)]，见 agent_ppo/conf/conf.py:28。当前特征主要是己方英雄生存/坐标和敌方塔少量信息，缺少敌方
    英雄、血量、技能 CD、兵线、金币/等级、距离、朝向、可攻击目标等关键 1v1 信息。

3. 奖励非常稀疏且偏单一。
    当前奖励只有 tower_hp_point 和 forward，见 agent_ppo/conf/conf.py:14。这会导致智能体更像“推塔/前进”策略，难学到换血、补刀、撤退、击杀、防守等行为。

4. PPO loss 里有一些设备和数值细节不够稳。
    agent_ppo/model/model.py:184 等位置直接创建 torch.tensor(0.0)，如果模型在 GPU 上，可能出现 device mismatch 风险。value_cost 在 agent_ppo/model/model.py:172 先算一次又
    马上覆盖，冗余且容易误导。

5. 配置和维度强耦合。
    特征维度、动作头、sample shape 多处手写。只要扩展特征，就要同步改 DimConfig、DATA_SPLIT_SHAPE、SERI_VEC_SPLIT_SHAPE、data_shapes，出错概率高。

优化建议

优先级最高的是把 PPO 主线先做扎实：


3. 重做奖励项。
    在现有推塔奖励基础上加入：击杀/死亡、对敌英雄伤害、受到伤害惩罚、补刀/经济、靠近安全区域、低血撤退、技能命中奖励、塔下危险惩罚。奖励权重要从小规模实验逐步调，不建议一
    次加太多大权重。

4. 增加训练可观测性。
    监控里除了 loss/reward，建议上报 episode 长度、胜率、击杀/死亡、塔血变化、平均 action entropy、非法动作 mask 后的动作分布。这样能判断是探索不足、奖励错、特征不足还是环
    境配置问题。

5. 清理 DIY 或标注为模板。
    当前 agent_diy 会让读代码的人误以为有第二套算法。可以在 README 明确“DIY 未实现”，或者直接补全一个最小可运行 baseline。

6. 给样本和 loss 加单元检查。
    最值得加的是：FeatureProcess 输出长度必须等于配置特征维度；FrameCollector 拼接样本长度必须等于 Config.SAMPLE_DIM；legal action mask 和 logits split 维度一致；PPO loss
    在 CPU/GPU 都能跑一批 fake sample。

总体看，这个项目的工程骨架是标准的“环境交互 -> 采样 -> GAE -> PPO 更新”，但当前智能体能力主要受限于三点：特征太少、奖励太粗、LSTM 未真正生效。建议先修正模型/样本结构的一致性，再扩特征和奖励，否则训练结果很难稳定提升。