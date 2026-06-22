#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.conf.runtime_config import runtime_value


BASE_REWARD_WEIGHT_DICT = {
    "tower_hp_point": 4.0,
    "enemy_tower_hp_down": 8.0,
    "self_tower_hp_down": 8.0,
    "tower_destroy": 20.0,
    "hp_point": 1.0,
    "money": 0.5,
    "exp": 0.5,
    "kill": 2.0,
    "death": 4.0,
    "forward": 0.05,
    "push_window_tower_damage": 2.0,
    "unsafe_dive": 2.0,
    "unsafe_dive_severity": 1.0,
    "push_window_active": 0.0,
    "unsafe_dive_active": 0.0,
    "win_result": 20.0,
    "timeout_tower_gap": 8.0,
}

REWARD_PROFILE_OVERRIDES = {
    "v1.2": {},
    "default": {},
    "no_window_reward": {
        "push_window_tower_damage": 0.0,
        "unsafe_dive": 0.0,
        "unsafe_dive_severity": 0.0,
    },
    "no_terminal_reward": {
        "win_result": 0.0,
        "timeout_tower_gap": 0.0,
    },
    "death_only_risk": {
        "push_window_tower_damage": 0.0,
        "unsafe_dive": 0.0,
        "unsafe_dive_severity": 1.0,
        "self_tower_hp_down": 0.0,
        "death": 4.0,
    },
}


def parse_reward_weight_overrides(raw_overrides):
    if not raw_overrides:
        return {}

    overrides = {}
    for item in raw_overrides.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip()
        if key not in BASE_REWARD_WEIGHT_DICT:
            continue
        try:
            overrides[key] = float(value.strip())
        except (TypeError, ValueError):
            continue
    return overrides


def build_reward_weight_dict(profile=None, raw_overrides=None):
    profile = profile or "v1.2"
    weights = dict(BASE_REWARD_WEIGHT_DICT)
    weights.update(REWARD_PROFILE_OVERRIDES.get(profile, {}))
    weights.update(parse_reward_weight_overrides(raw_overrides))
    return weights


class GameConfig:
    REWARD_PROFILE = runtime_value("HOK_REWARD_PROFILE", "v1.2")
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT = build_reward_weight_dict(
        profile=REWARD_PROFILE,
        raw_overrides=runtime_value("HOK_REWARD_WEIGHT_OVERRIDES"),
    )
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 0
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    DIM_OF_FEATURE = [83]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    FEATURE_DIM = DimConfig.DIM_OF_FEATURE[0]
    LEGAL_ACTION_DIM = 85
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        FEATURE_DIM + LEGAL_ACTION_DIM,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (LEGAL_ACTION_DIM,)]
    INIT_LEARNING_RATE_START = 5e-4
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(FEATURE_DIM + LEGAL_ACTION_DIM) * 16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
