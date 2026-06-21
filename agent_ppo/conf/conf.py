#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT = {
        "tower_hp_point": 5.0,
        "hero_hp_point": 2.0,
        "damage_to_enemy": 2.0,
        "damage_taken": -1.5,
        "kill": 8.0,
        "death": -8.0,
        "forward": 0.01,
        "retreat_low_hp": 0.05,
        "under_enemy_tower": -0.02,
    }
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 0
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    DIM_OF_FEATURE = [50]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    FEATURE_DIM = DimConfig.DIM_OF_FEATURE[0]
    REDUCED_LEGAL_ACTION_DIM = 85
    DATA_SPLIT_SHAPE = []
    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (REDUCED_LEGAL_ACTION_DIM,)]
    INIT_LEARNING_RATE_START = 1e-3
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

    data_shapes = []

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = 0


Config.REDUCED_LEGAL_ACTION_DIM = sum(Config.LABEL_SIZE_LIST)
Config.SERI_VEC_SPLIT_SHAPE = [(Config.FEATURE_DIM,), (Config.REDUCED_LEGAL_ACTION_DIM,)]
Config.DATA_SPLIT_SHAPE = (
    [Config.FEATURE_DIM + Config.REDUCED_LEGAL_ACTION_DIM, 1, 1]
    + [1 for _ in Config.LABEL_SIZE_LIST]
    + Config.LABEL_SIZE_LIST
    + [1 for _ in Config.LABEL_SIZE_LIST]
    + [1, Config.LSTM_UNIT_SIZE, Config.LSTM_UNIT_SIZE]
)
Config.data_shapes = (
    [[shape * Config.LSTM_TIME_STEPS] for shape in Config.DATA_SPLIT_SHAPE[:-2]]
    + [[Config.LSTM_UNIT_SIZE], [Config.LSTM_UNIT_SIZE]]
)
Config.SAMPLE_DIM = sum(Config.DATA_SPLIT_SHAPE[:-2]) * Config.LSTM_TIME_STEPS + sum(Config.DATA_SPLIT_SHAPE[-2:])
