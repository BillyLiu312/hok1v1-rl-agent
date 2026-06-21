#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder
from agent_ppo.conf.conf import GameConfig


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    monitor.title("智能决策1v1").add_group(
        group_name="算法指标",
        group_name_en="algorithm",
    )
    monitor.add_panel(name="累积回报", name_en="reward", type="line").add_metric(
        metrics_name="reward",
        expr="round(avg(reward{}), 0.01)",
    ).end_panel()
    monitor.add_panel(name="总损失", name_en="total_loss", type="line").add_metric(
        metrics_name="total_loss",
        expr="round(avg(total_loss{}), 0.01)",
    ).end_panel()
    monitor.add_panel(name="价值损失", name_en="value_loss", type="line").add_metric(
        metrics_name="value_loss",
        expr="round(avg(value_loss{}), 0.01)",
    ).end_panel()
    monitor.add_panel(name="策略损失", name_en="policy_loss", type="line").add_metric(
        metrics_name="policy_loss",
        expr="round(avg(policy_loss{}), 0.01)",
    ).end_panel()
    monitor.add_panel(name="熵损失", name_en="entropy_loss", type="line").add_metric(
        metrics_name="entropy_loss",
        expr="round(avg(entropy_loss{}), 0.01)",
    ).end_panel()
    for reward_name in GameConfig.REWARD_WEIGHT_DICT:
        metric_name = f"reward_{reward_name}"
        monitor.add_panel(name=f"回报_{reward_name}", name_en=metric_name, type="line").add_metric(
            metrics_name=metric_name,
            expr=f"round(avg({metric_name}{{}}), 0.0001)",
        ).end_panel()
    config_dict = monitor.end_group().build()
    return config_dict
