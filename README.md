# hok1v1-rl-agent

王者荣耀 1v1 智能体训练项目，基于腾讯 AI Arena / KaiwuDRL 环境组织 PPO 与 DIY 两套智能体实现。

## 项目结构

```text
.
├── agent_ppo/      # PPO 智能体、特征处理、模型与训练流程
├── agent_diy/      # DIY 智能体、模型与训练流程
├── conf/           # 应用与算法配置
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

## 说明

- `agent_ppo` 是默认训练方案。
- `agent_diy` 可用于自定义算法实验。
- 训练参数与环境配置主要位于 `conf/` 和各智能体目录下的 `conf/`。
