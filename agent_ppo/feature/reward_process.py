#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
from agent_ppo.conf.conf import GameConfig


TOWER_SUB_TYPE = 21
MAX_MONEY_DELTA = 1000.0
MAX_EXP_DELTA = 1000.0
TOWER_WINDOW_DISTANCE = 9000.0
HERO_THREAT_DISTANCE = 7000.0


def _safe_div(value, denom):
    if denom <= 0:
        return 0.0
    return value / denom


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight


def init_calc_frame_map():
    return {key: RewardStruct(weight) for key, weight in GameConfig.REWARD_WEIGHT_DICT.items()}


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.enemy_hero_camp = -1
        self.m_reward_value = {}
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG

    def result(self, frame_data):
        self.frame_data_process(frame_data)
        self.get_reward(self.m_reward_value)

        frame_no = frame_data["frame_no"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    def set_cur_calc_frame_vec(self, calc_frame_map, frame_data, camp):
        main_hero, enemy_hero = self._get_heroes(frame_data, camp)
        main_tower, enemy_tower = self._get_towers(frame_data, camp)

        for reward_name, reward_struct in calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value

            if reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = 1.0 - self._hp_rate(enemy_tower)
            elif reward_name == "enemy_tower_hp_down":
                reward_struct.cur_frame_value = 1.0 - self._hp_rate(enemy_tower)
            elif reward_name == "self_tower_hp_down":
                reward_struct.cur_frame_value = 1.0 - self._hp_rate(main_tower)
            elif reward_name == "tower_destroy":
                reward_struct.cur_frame_value = 1.0 if enemy_tower is not None and enemy_tower.get("hp", 0) <= 0 else 0.0
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = self._hp_rate(main_hero)
            elif reward_name == "money":
                reward_struct.cur_frame_value = self._norm(main_hero.get("money_cnt", 0) if main_hero else 0, MAX_MONEY_DELTA)
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self._norm(self._level_exp_score(main_hero), MAX_EXP_DELTA)
            elif reward_name == "kill":
                reward_struct.cur_frame_value = float(main_hero.get("kill_cnt", 0) if main_hero else 0)
            elif reward_name == "death":
                reward_struct.cur_frame_value = float(main_hero.get("dead_cnt", 0) if main_hero else 0)
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower, enemy_hero)
            elif reward_name == "push_window_tower_damage":
                reward_struct.cur_frame_value = self._push_window_tower_damage(
                    main_hero, enemy_hero, enemy_tower, frame_data, camp, reward_struct.last_frame_value
                )
            elif reward_name == "unsafe_dive":
                reward_struct.cur_frame_value = self._unsafe_dive(main_hero, enemy_hero, enemy_tower, frame_data, camp)
            elif reward_name in ("win_result", "timeout_tower_gap"):
                reward_struct.cur_frame_value = 0.0

    def calculate_forward(self, main_hero, main_tower, enemy_tower, enemy_hero):
        if main_hero is None or main_tower is None or enemy_tower is None:
            return 0.0
        if main_hero.get("hp", 0) <= 0:
            return 0.0

        hero_pos = self._pos(main_hero)
        main_tower_pos = self._pos(main_tower)
        enemy_tower_pos = self._pos(enemy_tower)
        dist_hero_to_enemy_tower = math.dist(hero_pos, enemy_tower_pos)
        dist_tower_to_tower = max(math.dist(main_tower_pos, enemy_tower_pos), 1.0)
        progress = 1.0 - min(dist_hero_to_enemy_tower / dist_tower_to_tower, 1.5)

        hp_rate = self._hp_rate(main_hero)
        enemy_alive_nearby = False
        if enemy_hero is not None and enemy_hero.get("hp", 0) > 0:
            enemy_alive_nearby = math.dist(hero_pos, self._pos(enemy_hero)) < 7000

        if hp_rate < 0.35 and enemy_alive_nearby:
            return -0.5
        return progress

    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                main_camp = hero["camp"]
            else:
                enemy_camp = hero["camp"]

        self.main_hero_camp = main_camp
        self.enemy_hero_camp = enemy_camp
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    def get_reward(self, reward_dict):
        reward_dict.clear()
        reward_sum = 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            main_value = self.m_main_calc_frame_map[reward_name]
            enemy_value = self.m_enemy_calc_frame_map[reward_name]

            if reward_name == "forward":
                reward_struct.value = main_value.cur_frame_value - main_value.last_frame_value
            elif reward_name == "enemy_tower_hp_down":
                reward_struct.value = main_value.cur_frame_value - main_value.last_frame_value
            elif reward_name == "self_tower_hp_down":
                reward_struct.value = -1.0 * (main_value.cur_frame_value - main_value.last_frame_value)
            elif reward_name == "death":
                reward_struct.value = -1.0 * (main_value.cur_frame_value - main_value.last_frame_value)
            elif reward_name == "push_window_tower_damage":
                reward_struct.value = main_value.cur_frame_value - main_value.last_frame_value
            elif reward_name == "unsafe_dive":
                reward_struct.value = -1.0 * main_value.cur_frame_value
            elif reward_name in ("win_result", "timeout_tower_gap"):
                reward_struct.value = 0.0
            else:
                reward_struct.cur_frame_value = main_value.cur_frame_value - enemy_value.cur_frame_value
                reward_struct.last_frame_value = main_value.last_frame_value - enemy_value.last_frame_value
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = reward_sum

    def terminal_reward(self, frame_data, win, truncated=False):
        main_tower, enemy_tower = self._get_towers(frame_data, self.main_hero_camp)
        if truncated:
            tower_gap = self._hp_rate(main_tower) - self._hp_rate(enemy_tower)
            timeout_tower_gap = max(-1.0, min(1.0, tower_gap))
            win_result = 0.0
        else:
            timeout_tower_gap = 0.0
            win_result = 1.0 if win else -1.0

        reward = {
            "win_result": win_result,
            "timeout_tower_gap": timeout_tower_gap,
        }
        reward["reward_sum"] = (
            win_result * GameConfig.REWARD_WEIGHT_DICT.get("win_result", 0.0)
            + timeout_tower_gap * GameConfig.REWARD_WEIGHT_DICT.get("timeout_tower_gap", 0.0)
        )
        return reward

    def _get_heroes(self, frame_data, camp):
        main_hero, enemy_hero = None, None
        for hero in frame_data["hero_states"]:
            if hero["camp"] == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        return main_hero, enemy_hero

    def _get_towers(self, frame_data, camp):
        main_tower, enemy_tower = None, None
        for npc in frame_data["npc_states"]:
            if npc.get("sub_type") != TOWER_SUB_TYPE:
                continue
            if npc.get("camp") == camp:
                main_tower = npc
            else:
                enemy_tower = npc
        return main_tower, enemy_tower

    def _push_window_tower_damage(self, main_hero, enemy_hero, enemy_tower, frame_data, camp, last_value):
        if not self._has_push_window(main_hero, enemy_hero, enemy_tower, frame_data, camp):
            return last_value
        return 1.0 - self._hp_rate(enemy_tower)

    def _unsafe_dive(self, main_hero, enemy_hero, enemy_tower, frame_data, camp):
        if main_hero is None or enemy_tower is None or main_hero.get("hp", 0) <= 0:
            return 0.0
        low_hp = self._hp_rate(main_hero) < 0.35
        no_wave = self._nearby_lane_count(frame_data, camp, enemy_tower) <= 0
        in_tower_range = self._distance(main_hero, enemy_tower) <= TOWER_WINDOW_DISTANCE
        enemy_threat = enemy_hero is not None and enemy_hero.get("hp", 0) > 0 and self._distance(main_hero, enemy_hero) <= HERO_THREAT_DISTANCE
        tower_targets_hero = enemy_tower.get("attack_target") == main_hero.get("runtime_id")
        return 1.0 if in_tower_range and no_wave and (low_hp or tower_targets_hero or enemy_threat) else 0.0

    def _has_push_window(self, main_hero, enemy_hero, enemy_tower, frame_data, camp):
        if main_hero is None or enemy_tower is None or main_hero.get("hp", 0) <= 0:
            return False
        near_wave_count = self._nearby_lane_count(frame_data, camp, enemy_tower)
        tower_targets_wave = self._tower_targets_lane_unit(frame_data, camp, enemy_tower)
        hero_close = self._distance(main_hero, enemy_tower) <= TOWER_WINDOW_DISTANCE
        enemy_far_or_dead = enemy_hero is None or enemy_hero.get("hp", 0) <= 0 or self._distance(main_hero, enemy_hero) > HERO_THREAT_DISTANCE
        return hero_close and near_wave_count > 0 and (tower_targets_wave or enemy_far_or_dead)

    def _nearby_lane_count(self, frame_data, camp, tower):
        count = 0
        for npc in frame_data["npc_states"]:
            if npc.get("sub_type") == TOWER_SUB_TYPE or npc.get("camp") != camp or npc.get("hp", 0) <= 0:
                continue
            if self._distance(npc, tower) <= TOWER_WINDOW_DISTANCE:
                count += 1
        return count

    def _tower_targets_lane_unit(self, frame_data, camp, tower):
        attack_target = tower.get("attack_target") if tower else None
        if attack_target is None:
            return False
        for npc in frame_data["npc_states"]:
            if npc.get("runtime_id") == attack_target and npc.get("camp") == camp and npc.get("sub_type") != TOWER_SUB_TYPE:
                return True
        return False

    def _hp_rate(self, unit):
        if unit is None:
            return 0.0
        return _safe_div(unit.get("hp", 0), unit.get("max_hp", 0))

    def _level_exp_score(self, hero):
        if hero is None:
            return 0.0
        return hero.get("level", 0) * 100.0 + hero.get("exp", 0)

    def _norm(self, value, scale):
        return value / scale

    def _pos(self, unit):
        location = unit.get("location", {})
        return (location.get("x", 0), location.get("z", 0))

    def _distance(self, unit_a, unit_b):
        if unit_a is None or unit_b is None:
            return float("inf")
        return math.dist(self._pos(unit_a), self._pos(unit_b))
