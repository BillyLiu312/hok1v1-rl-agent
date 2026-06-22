#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest

from agent_ppo.conf.conf import Config, GameConfig, build_reward_weight_dict, parse_reward_weight_overrides
from agent_ppo.conf.summoner_skill import (
    DEFAULT_SUMMONER_SKILL_BY_HERO,
    MATCHUP_SUMMONER_SKILL_OVERRIDES,
    select_summoner_skill,
)
from agent_ppo.feature.feature_process import FeatureProcess, V1_2_REQUIRED_FEATURES, feature_index_map, feature_schema
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

    def test_feature_schema_matches_vector_layout(self):
        schema = feature_schema()
        index_map = feature_index_map()
        self.assertEqual(len(schema), Config.FEATURE_DIM)
        self.assertEqual(len(index_map), Config.FEATURE_DIM)
        self.assertEqual(index_map["main.hero_id_112"], 0)
        self.assertEqual(index_map["enemy_tower.distance"], 47 + 11 + 6)
        self.assertEqual(index_map["push_window.main_lane_near_enemy_tower_count"], Config.FEATURE_DIM - 4)

    def test_v1_2_required_feature_schema_entries_are_present(self):
        schema = set(feature_schema())
        for feature_name in V1_2_REQUIRED_FEATURES:
            self.assertIn(feature_name, schema)

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
            "unsafe_dive_severity",
            "push_window_active",
            "unsafe_dive_active",
            "win_result",
            "timeout_tower_gap",
            "reward_sum",
        ):
            self.assertIn(key, reward)

    def test_push_window_and_unsafe_dive_diagnostics_are_unweighted(self):
        frame_state = make_frame_state()
        frame_state["hero_states"][0]["location"] = {"x": 16000, "z": 0}
        frame_state["hero_states"][1]["location"] = {"x": 40000, "z": 0}
        reward = GameRewardManager(1001).result(frame_state)

        self.assertEqual(reward["push_window_active"], 1.0)
        self.assertEqual(reward["unsafe_dive_active"], 0.0)

        frame_state["hero_states"][0]["hp"] = 500
        frame_state["hero_states"][1]["location"] = {"x": 17000, "z": 0}
        frame_state["npc_states"] = [
            npc for npc in frame_state["npc_states"] if npc.get("camp") != 1 or npc.get("sub_type") == 21
        ]
        frame_state["npc_states"][1]["attack_target"] = 1001
        reward = GameRewardManager(1001).result(frame_state)

        self.assertEqual(reward["push_window_active"], 0.0)
        self.assertEqual(reward["unsafe_dive_active"], 1.0)
        self.assertEqual(reward["unsafe_dive"], -1.0)
        self.assertEqual(reward["unsafe_dive_severity"], -1.5)

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

    def test_v1_2_conservative_ppo_hyperparameters(self):
        self.assertEqual(Config.INIT_LEARNING_RATE_START, 5e-4)
        self.assertEqual(Config.TARGET_LR, 1e-4)
        self.assertEqual(Config.GAMMA, 0.995)
        self.assertEqual(Config.BETA_START, 0.025)
        self.assertEqual(Config.CLIP_PARAM, 0.2)
        self.assertTrue(Config.USE_GRAD_CLIP)
        self.assertEqual(Config.GRAD_CLIP_RANGE, 0.5)

    def test_reward_profile_defaults_to_v1_2_weights(self):
        self.assertEqual(GameConfig.REWARD_PROFILE, "v1.2")
        self.assertEqual(GameConfig.REWARD_WEIGHT_DICT["win_result"], 20.0)
        self.assertEqual(GameConfig.REWARD_WEIGHT_DICT["push_window_tower_damage"], 2.0)
        self.assertEqual(GameConfig.REWARD_WEIGHT_DICT["unsafe_dive"], 2.0)
        self.assertEqual(GameConfig.REWARD_WEIGHT_DICT["unsafe_dive_severity"], 1.0)

    def test_reward_profiles_support_ablation_weights(self):
        no_window = build_reward_weight_dict(profile="no_window_reward")
        self.assertEqual(no_window["push_window_tower_damage"], 0.0)
        self.assertEqual(no_window["unsafe_dive"], 0.0)
        self.assertEqual(no_window["unsafe_dive_severity"], 0.0)
        self.assertEqual(no_window["win_result"], 20.0)

        no_terminal = build_reward_weight_dict(profile="no_terminal_reward")
        self.assertEqual(no_terminal["win_result"], 0.0)
        self.assertEqual(no_terminal["timeout_tower_gap"], 0.0)
        self.assertEqual(no_terminal["push_window_tower_damage"], 2.0)

        death_only = build_reward_weight_dict(profile="death_only_risk")
        self.assertEqual(death_only["unsafe_dive"], 0.0)
        self.assertEqual(death_only["unsafe_dive_severity"], 1.0)

    def test_reward_weight_overrides_are_scoped_to_known_keys(self):
        overrides = parse_reward_weight_overrides("death:5,unknown:99,bad,push_window_tower_damage:x,money:0.25")
        self.assertEqual(overrides, {"death": 5.0, "money": 0.25})

        weights = build_reward_weight_dict(profile="no_window_reward", raw_overrides="push_window_tower_damage:1.5")
        self.assertEqual(weights["push_window_tower_damage"], 1.5)


if __name__ == "__main__":
    unittest.main()
