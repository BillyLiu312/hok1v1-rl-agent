#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config, GameConfig
from agent_ppo.feature.feature_process import FeatureProcess
from agent_ppo.feature.reward_process import GameRewardManager
from agent_ppo.model.model import Model


class SampleData:
    def __init__(self, sample):
        self.sample = sample


def fake_frame_state(enemy_hp=1000, frame_no=1):
    return {
        "frame_no": frame_no,
        "hero_states": [
            {
                "runtime_id": 1,
                "config_id": 112,
                "camp": 1,
                "hp": 1000,
                "max_hp": 1000,
                "level": 3,
                "exp": 80,
                "max_exp": 160,
                "money": 1200,
                "location": {"x": -10000, "z": 0},
                "skill_state": [{"usable": True}, {"cooldown": 2, "max_cooldown": 10}],
            },
            {
                "runtime_id": 2,
                "config_id": 133,
                "camp": 2,
                "hp": enemy_hp,
                "max_hp": 1000,
                "level": 3,
                "exp": 80,
                "max_exp": 160,
                "money": 1200,
                "location": {"x": 10000, "z": 0},
            },
        ],
        "npc_states": [
            {"camp": 1, "sub_type": 21, "hp": 5000, "max_hp": 5000, "location": {"x": -30000, "z": 0}},
            {"camp": 2, "sub_type": 21, "hp": 5000, "max_hp": 5000, "location": {"x": 30000, "z": 0}},
        ],
    }


def build_fake_sample():
    per_frame = []
    feature = torch.zeros(Config.FEATURE_DIM)
    legal_action = torch.ones(Config.REDUCED_LEGAL_ACTION_DIM)
    seri_vec = torch.cat([feature, legal_action])
    per_frame.append(seri_vec.repeat(Config.LSTM_TIME_STEPS))
    per_frame.append(torch.ones(Config.LSTM_TIME_STEPS))
    per_frame.append(torch.linspace(-1.0, 1.0, Config.LSTM_TIME_STEPS))

    for label_size in Config.LABEL_SIZE_LIST:
        per_frame.append(torch.zeros(Config.LSTM_TIME_STEPS))

    for label_size in Config.LABEL_SIZE_LIST:
        per_frame.append(torch.full((label_size * Config.LSTM_TIME_STEPS,), 1.0 / label_size))

    for _ in Config.LABEL_SIZE_LIST:
        per_frame.append(torch.ones(Config.LSTM_TIME_STEPS))

    per_frame.append(torch.ones(Config.LSTM_TIME_STEPS))
    per_frame.append(torch.zeros(Config.LSTM_UNIT_SIZE))
    per_frame.append(torch.zeros(Config.LSTM_UNIT_SIZE))
    return torch.cat(per_frame).float()


def test_config_shapes_are_consistent():
    assert Config.FEATURE_DIM == Config.SERI_VEC_SPLIT_SHAPE[0][0]
    assert Config.REDUCED_LEGAL_ACTION_DIM == sum(Config.LABEL_SIZE_LIST)
    assert Config.DATA_SPLIT_SHAPE[0] == Config.FEATURE_DIM + Config.REDUCED_LEGAL_ACTION_DIM
    assert Config.SAMPLE_DIM == sum(shape[0] for shape in Config.data_shapes)


def test_feature_process_outputs_configured_dim():
    processor = FeatureProcess(camp=1)
    feature = processor.process_feature({"frame_state": fake_frame_state()})
    assert len(feature) == Config.FEATURE_DIM
    assert all(0.0 <= value <= 1.0 for value in feature)


def test_reward_manager_emits_all_reward_items():
    manager = GameRewardManager(main_hero_runtime_id=1)
    reward_1 = dict(manager.result(fake_frame_state(enemy_hp=1000, frame_no=1)))
    reward_2 = dict(manager.result(fake_frame_state(enemy_hp=900, frame_no=2)))
    assert set(GameConfig.REWARD_WEIGHT_DICT).issubset(reward_2.keys())
    assert "reward_sum" in reward_2
    assert reward_2["damage_to_enemy"] > reward_1["damage_to_enemy"]


def test_reward_manager_handles_tower_risk_rewards():
    frame_state = fake_frame_state(enemy_hp=900, frame_no=2)
    frame_state["hero_states"][0]["hp"] = 200
    frame_state["hero_states"][0]["location"] = {"x": 29500, "z": 0}

    manager = GameRewardManager(main_hero_runtime_id=1)
    reward = manager.result(frame_state)

    assert "retreat_low_hp" in reward
    assert "under_enemy_tower" in reward
    assert reward["under_enemy_tower"] > 0.0


def test_ppo_learn_accepts_fake_sample():
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.INIT_LEARNING_RATE_START)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    algorithm = Algorithm(model, optimizer, scheduler, device=torch.device("cpu"))
    sample = SampleData(build_fake_sample())
    assert sample.sample.numel() == Config.SAMPLE_DIM
    result = algorithm.learn([sample])
    assert "total_loss" in result
    assert torch.isfinite(torch.tensor(result["total_loss"]))


if __name__ == "__main__":
    test_config_shapes_are_consistent()
    test_feature_process_outputs_configured_dim()
    test_reward_manager_emits_all_reward_items()
    test_reward_manager_handles_tower_risk_rewards()
    test_ppo_learn_accepts_fake_sample()
