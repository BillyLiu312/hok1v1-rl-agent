#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("智能决策1v1")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="round(avg(reward{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="奖励分解",
            name_en="reward_detail",
            type="line",
        )
        .add_metric(
            metrics_name="tower_hp",
            expr="round(avg(reward_tower_hp_point{}), 0.01)",
        )
        .add_metric(
            metrics_name="enemy_tower_hp_down",
            expr="round(avg(reward_enemy_tower_hp_down{}), 0.01)",
        )
        .add_metric(
            metrics_name="self_tower_hp_down",
            expr="round(avg(reward_self_tower_hp_down{}), 0.01)",
        )
        .add_metric(
            metrics_name="tower_destroy",
            expr="round(avg(reward_tower_destroy{}), 0.01)",
        )
        .add_metric(
            metrics_name="hp",
            expr="round(avg(reward_hp_point{}), 0.01)",
        )
        .add_metric(
            metrics_name="money",
            expr="round(avg(reward_money{}), 0.01)",
        )
        .add_metric(
            metrics_name="exp",
            expr="round(avg(reward_exp{}), 0.01)",
        )
        .add_metric(
            metrics_name="kill",
            expr="round(avg(reward_kill{}), 0.01)",
        )
        .add_metric(
            metrics_name="death",
            expr="round(avg(reward_death{}), 0.01)",
        )
        .add_metric(
            metrics_name="forward",
            expr="round(avg(reward_forward{}), 0.01)",
        )
        .add_metric(
            metrics_name="push_window_tower_damage",
            expr="round(avg(reward_push_window_tower_damage{}), 0.01)",
        )
        .add_metric(
            metrics_name="unsafe_dive",
            expr="round(avg(reward_unsafe_dive{}), 0.01)",
        )
        .add_metric(
            metrics_name="win_result",
            expr="round(avg(reward_win_result{}), 0.01)",
        )
        .add_metric(
            metrics_name="timeout_tower_gap",
            expr="round(avg(reward_timeout_tower_gap{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="round(avg(total_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="round(avg(value_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="round(avg(policy_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="round(avg(entropy_loss{}), 0.01)",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
