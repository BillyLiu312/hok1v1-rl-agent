#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest

from agent_ppo.conf.conf import Config
from agent_ppo.conf.summoner_skill import (
    DEFAULT_SUMMONER_SKILL_BY_HERO,
    MATCHUP_SUMMONER_SKILL_OVERRIDES,
    select_summoner_skill,
)
from agent_ppo.feature.feature_process import FeatureProcess
from agent_ppo.feature.reward_process import GameRewardManager


def make_frame_state():
    hero_blue = {
        "runtime_id": 1001,
        "config_id": 112,
        "camp": 1,
        "hp": 3000,
        "max_hp": 4000,
        "ep": 100,
        "max_ep": 200,
        "level": 4,
        "exp": 120,
        "money": 800,
        "money_cnt": 1800,
        "phy_atk": 220,
        "phy_def": 120,
        "mov_spd": 400,
        "atk_spd": 600,
        "kill_cnt": 1,
        "dead_cnt": 0,
        "revive_time": 0,
        "total_hurt": 3000,
        "total_hurt_to_hero": 1500,
        "total_be_hurt_by_hero": 800,
        "location": {"x": -12000, "z": -2000},
        "skill_state": {
            "slot_states": [
                {"usable": True, "cooldown": 0, "cooldown_max": 1000},
                {"usable": False, "cooldown": 3000, "cooldown_max": 6000},
                {"usable": True, "cooldown": 0, "cooldown_max": 8000},
                {"usable": False, "cooldown": 5000, "cooldown_max": 10000},
            ]
        },
    }
    hero_red = {
        **hero_blue,
        "runtime_id": 2001,
        "config_id": 133,
        "camp": 2,
        "hp": 2500,
        "max_hp": 4200,
        "level": 3,
        "money_cnt": 1500,
        "kill_cnt": 0,
        "dead_cnt": 1,
        "revive_time": 0,
        "total_hurt": 2000,
        "total_hurt_to_hero": 800,
        "total_be_hurt_by_hero": 1500,
        "location": {"x": 4000, "z": 3000},
    }
    blue_tower = {
        "sub_type": 21,
        "camp": 1,
        "hp": 9000,
        "max_hp": 10000,
        "location": {"x": -18000, "z": 0},
        "attack_target": 2001,
    }
    red_tower = {
        "sub_type": 21,
        "camp": 2,
        "hp": 7000,
        "max_hp": 10000,
        "location": {"x": 18000, "z": 0},
        "attack_target": 3001,
    }
    blue_minion = {
        "runtime_id": 3001,
        "sub_type": 1,
        "camp": 1,
        "hp": 1000,
        "max_hp": 1200,
        "location": {"x": 15000, "z": 0},
    }
    red_minion = {
        "runtime_id": 4001,
        "sub_type": 1,
        "camp": 2,
        "hp": 800,
        "max_hp": 1200,
        "location": {"x": 2000, "z": 0},
    }
    return {
        "frame_no": 42,
        "hero_states": [hero_blue, hero_red],
        "npc_states": [blue_tower, red_tower, blue_minion, red_minion],
    }


class PpoOptimizationTest(unittest.TestCase):
    def test_feature_length_matches_config(self):
        feature = FeatureProcess(1).process_feature({"frame_state": make_frame_state()})
        self.assertEqual(len(feature), Config.FEATURE_DIM)
        self.assertGreaterEqual(min(feature), 0.0)
        self.assertLessEqual(max(feature), 1.0)

    def test_reward_contains_all_configured_items(self):
        reward = GameRewardManager(1001).result(make_frame_state())
        for key in (
            "tower_hp_point",
            "enemy_tower_hp_down",
            "self_tower_hp_down",
            "tower_destroy",
            "hp_point",
            "money",
            "exp",
            "kill",
            "death",
            "forward",
            "push_window_tower_damage",
            "unsafe_dive",
            "win_result",
            "timeout_tower_gap",
            "reward_sum",
        ):
            self.assertIn(key, reward)

    def test_terminal_reward_uses_win_and_timeout(self):
        manager = GameRewardManager(1001)
        frame_state = make_frame_state()
        manager.result(frame_state)
        win_reward = manager.terminal_reward(frame_state, win=1, truncated=False)
        timeout_reward = manager.terminal_reward(frame_state, win=0, truncated=True)
        self.assertEqual(win_reward["win_result"], 1.0)
        self.assertEqual(win_reward["timeout_tower_gap"], 0.0)
        self.assertGreater(timeout_reward["timeout_tower_gap"], 0.0)
        self.assertIn("reward_sum", timeout_reward)

    def test_summoner_skill_tables_are_deterministic(self):
        self.assertEqual(DEFAULT_SUMMONER_SKILL_BY_HERO[112], 80115)
        self.assertEqual(DEFAULT_SUMMONER_SKILL_BY_HERO[133], 80110)
        self.assertEqual(DEFAULT_SUMMONER_SKILL_BY_HERO[199], 80110)
        self.assertEqual(MATCHUP_SUMMONER_SKILL_OVERRIDES[(199, 133)], 80107)
        self.assertEqual(select_summoner_skill(199, 133), 80107)


if __name__ == "__main__":
    unittest.main()
