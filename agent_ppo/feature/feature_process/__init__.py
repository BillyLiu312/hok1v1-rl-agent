#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess
from agent_ppo.feature.feature_process.soldier_process import SoldierProcess


HERO_FEATURE_SCHEMA = [
    "main.hero_id_112",
    "main.hero_id_133",
    "main.hero_id_199",
    "main.is_alive",
    "main.hp_rate",
    "main.ep_rate",
    "main.level",
    "main.exp",
    "main.money",
    "main.money_cnt",
    "main.phy_atk",
    "main.phy_def",
    "main.mov_spd",
    "main.atk_spd",
    "main.pos_x",
    "main.pos_z",
    "main.kill_cnt",
    "main.dead_cnt",
    "enemy.hero_id_112",
    "enemy.hero_id_133",
    "enemy.hero_id_199",
    "enemy.visible",
    "enemy.is_alive",
    "enemy.hp_rate",
    "enemy.level",
    "enemy.money_cnt",
    "enemy.rel_x",
    "enemy.rel_z",
    "enemy.distance",
    "enemy.kill_cnt",
    "enemy.dead_cnt",
    "skill.0.usable",
    "skill.0.cooldown_rate",
    "skill.1.usable",
    "skill.1.cooldown_rate",
    "skill.2.usable",
    "skill.2.cooldown_rate",
    "skill.3.usable",
    "skill.3.cooldown_rate",
    "matchup.hp_advantage",
    "matchup.level_advantage",
    "matchup.money_advantage",
    "matchup.damage_to_hero_advantage",
    "matchup.hero_damage_taken_advantage",
    "matchup.main_revive_time",
    "matchup.enemy_revive_time",
    "matchup.main_low_hp",
]

TOWER_FEATURE_FIELDS = [
    "alive",
    "hp_rate",
    "pos_x",
    "pos_z",
    "rel_x",
    "rel_z",
    "distance",
    "targets_main_hero",
    "targets_main_lane",
    "targets_enemy_hero",
    "main_hero_in_danger_range",
]

ORGAN_FEATURE_SCHEMA = [
    f"{tower}.{field}"
    for tower in ("main_tower", "enemy_tower")
    for field in TOWER_FEATURE_FIELDS
]

SOLDIER_FEATURE_SCHEMA = [
    "lane.main_count",
    "lane.enemy_count",
    "lane.main_hp_sum",
    "lane.enemy_hp_sum",
    "lane.main_nearest_hero_distance",
    "lane.enemy_nearest_hero_distance",
    "lane.main_forward_mean",
    "lane.enemy_forward_mean",
    "lane.enemy_lowest_hp_rate",
    "lane.hp_advantage",
    "push_window.main_lane_near_enemy_tower_count",
    "push_window.main_lane_near_enemy_tower_hp_sum",
    "push_window.enemy_lane_near_enemy_tower_count",
    "push_window.main_lane_nearest_enemy_tower_distance",
]

FEATURE_SCHEMA = HERO_FEATURE_SCHEMA + ORGAN_FEATURE_SCHEMA + SOLDIER_FEATURE_SCHEMA

V1_2_REQUIRED_FEATURES = [
    "enemy_tower.distance",
    "enemy_tower.targets_main_hero",
    "enemy_tower.targets_main_lane",
    "enemy_tower.main_hero_in_danger_range",
    "push_window.main_lane_near_enemy_tower_count",
    "push_window.main_lane_near_enemy_tower_hp_sum",
    "push_window.enemy_lane_near_enemy_tower_count",
    "push_window.main_lane_nearest_enemy_tower_distance",
    "matchup.money_advantage",
    "matchup.level_advantage",
    "matchup.damage_to_hero_advantage",
    "matchup.hero_damage_taken_advantage",
    "matchup.main_revive_time",
    "matchup.enemy_revive_time",
]


def feature_schema():
    return list(FEATURE_SCHEMA)


def feature_index_map():
    return {name: index for index, name in enumerate(FEATURE_SCHEMA)}


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.soldier_process = SoldierProcess(camp)

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.soldier_process = SoldierProcess(camp)

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_soldier_feature(self, frame_state):
        return self.soldier_process.process_vec_soldier(frame_state)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        main_camp_hero_vector_feature = self.process_hero_feature(frame_state)
        organ_feature = self.process_organ_feature(frame_state)
        soldier_feature = self.process_soldier_feature(frame_state)

        feature = main_camp_hero_vector_feature + organ_feature + soldier_feature

        return feature
