# hok1v1-rl-agent

王者荣耀 1v1 智能体训练项目，基于腾讯 AI Arena / KaiwuDRL 环境组织 PPO 与 DIY 两套智能体实现。

## 项目结构

```text
.
├── agent_ppo/      # PPO 智能体、特征处理、模型与训练流程
├── agent_diy/      # DIY 智能体、模型与训练流程
├── conf/           # 应用与算法配置
├── docs/           # 开发指南、优化方案与实验分析
├── paper/          # 项目 minipaper、poster 与 LaTeX 源文件
├── train_test.py   # 本地训练测试入口
└── kaiwu.json      # Kaiwu 相关配置
```

## 快速开始

运行训练测试前，先在 `train_test.py` 中确认算法名称：

```python
algorithm_name = "ppo"  # 可选: "ppo" 或 "diy"
```

然后执行：

```bash
python train_test.py
```

## 开发文档

完整文档入口见 [docs/README.md](docs/README.md)。

常用入口：

- [项目简介](docs/dev-guide/intro.md)：任务目标、地图、英雄、胜负规则。
- [环境详述](docs/dev-guide/env.md)：环境配置、观测、动作空间和监控指标。
- [智能体详述](docs/dev-guide/agent_lite.md)：特征处理、奖励设计、召唤师技能和评估模式。
- [数据协议](docs/dev-guide/protocol.md)：原始帧状态、英雄/NPC/技能字段定义。
- [胜率优化管线](docs/optimization-pipeline.md)：面向胜率提升的工程优化路线。

## 论文与海报

项目展示材料入口见 [paper/README.md](paper/README.md)。

常用文件：

- [Minipaper PDF](paper/Honor-of-King-Paper/minipaper.pdf)：项目研究小论文，包含方法、实验、结果、参考文献和附录。
- [Minipaper LaTeX](paper/Honor-of-King-Paper/minipaper.tex)：minipaper 源文件，基于 NeurIPS 模板。
- [Poster PDF](paper/poster.pdf)：项目学术海报。
- [Poster LaTeX](paper/poster.tex)：poster 源文件。

## 说明

- `agent_ppo` 是默认训练方案。
- `agent_diy` 可用于自定义算法实验。
- 训练参数与环境配置主要位于 `conf/` 和各智能体目录下的 `conf/`。
