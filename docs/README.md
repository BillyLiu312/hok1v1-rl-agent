# hok1v1 文档树

保存日期：2026-06-22

## 智能体决策1v1实验开发指南

1. [项目简介](dev-guide/intro.md)
2. [环境详述](dev-guide/env.md)
3. [智能体详述](dev-guide/agent_lite.md)
4. [数据协议](dev-guide/protocol.md)
5. [胜率优化管线](optimization-pipeline.md)
6. [v1.1 训练分析](v1.1-training-analysis.md)
7. [v1.2 实施计划](v1.2-implementation-plan.md)
8. [v1.2 Runbook](v1.2-runbook.md)

## v1.2 本地工具

- `utils/analyze_training_logs.py`：汇总 `logs/v*/step-*.md` 为训练摘要。
- `utils/evaluation_matrix.py`：生成 checkpoint x matchup x 换边 x 召唤师技能的固定评估清单。
- `utils/evaluation_config_export.py`：把评估矩阵导出为 `usr_conf` JSONL、TOML 配置片段和 TOML metadata sidecar。
- `utils/analyze_run_records.py`：聚合训练账本中的 episode/matchup/reward 分解，并输出死亡高分位、己塔低分位和超时率等风险尾部指标。
- `utils/select_checkpoint.py`：根据训练摘要、矩阵评估结果和风险尾部指标排序 checkpoint。
- `utils/compare_experiment_reports.py`：横向比较多个 v1.2 证据包，输出 baseline delta、resolved reward 权重哈希、消融判读和 Markdown 摘要。
- `utils/v1_2_baseline.py`：从 `logs/v1.1/step-*.md` 自动生成 v1.2 验收门槛使用的 v1.1 baseline JSON/Markdown。
- `utils/evaluate_v1_2_candidate.py`：根据 v1.2 验收标准逐项判定候选 checkpoint。
- `utils/evaluate_summoner_skill_policy.py`：检查召唤师技能推荐是否有足够样本、胜率增益和风险对照，决定是否进入规则表审阅。
- `utils/v1_2_preflight.py`：训练前一次性检查 v1.2-a 配置、reward、实验方案、启动清单命令、工具和同步 preset。
- `utils/v1_2_experiment_plan.py`：生成 v1.2 科研主线、消融组、固定矩阵、成功指标和比较命令。
- `utils/v1_2_launch_manifest.py`：生成训练启动清单，记录 commit、同步包 hash、环境变量、reward/profile 覆盖和报告命令。
- `utils/run_metadata_summary.py`：汇总训练启动配置快照、reward profile、实际解析后的 reward 权重摘要/哈希、对手课程和关键配置 hash。
- `utils/opponent_curriculum_summary.py`：按实际对手来源统计 v1.2-b 课程采样比例、胜率、死亡和塔血。
- `utils/checkpoint_matrix.py`：生成 checkpoint-vs-opponent 胜率矩阵和 Elo 排名。
- `utils/summoner_skill_results.py`：按 matchup 和召唤师技能选择聚合胜率、死亡和推塔指标，并输出技能推荐。
- `utils/summoner_skill_policy_patch.py`：把技能推荐导出为可审阅的 matchup-conditioned 规则表草案。
- `utils/build_experiment_report.py`：一键生成 v1.2 训练摘要、科研叙事摘要、评估矩阵、召唤师技能网格和 checkpoint 排名证据包。
- `utils/offline_sync.py check --preset v1.2`：训练前检查离线同步包是否包含 v1.2 必备代码、工具、测试和文档。

## 腾讯开悟强化学习框架

1. [综述](taa-rl-fw/intro.md)
2. [环境](taa-rl-fw/rl_env.md)
3. [智能体 / 综述](taa-rl-fw/rl_agent/info.md)
4. [智能体 / 特征处理](taa-rl-fw/rl_agent/feature.md)
5. [智能体 / 算法开发](taa-rl-fw/rl_agent/algorithm.md)
6. [智能体 / 模型开发](taa-rl-fw/rl_agent/model.md)
7. [智能体 / 工作流开发](taa-rl-fw/rl_agent/workflow.md)
8. [智能体 / 智能体开发](taa-rl-fw/rl_agent/agent.md)
9. [分布式计算框架](taa-rl-fw/distributed_computing_fw.md)
10. [其他工具 / 监控与日志](taa-rl-fw/other_tools/monitor_log.md)
11. [强化学习系统系列技术标准](taa-rl-fw/rl_standard.md)
