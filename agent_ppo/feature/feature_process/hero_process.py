#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math


HERO_IDS = [112, 133, 199]
MAX_POSITION = 60000.0
MAX_DISTANCE = 120000.0
MAX_LEVEL = 15.0
MAX_MONEY = 20000.0
MAX_EXP = 2000.0
MAX_ATTACK = 2000.0
MAX_DEFENSE = 2000.0
MAX_MOVE_SPEED = 2000.0
MAX_ATTACK_SPEED = 3000.0
MAX_KDA = 20.0
MAX_COOLDOWN = 60000.0
MAX_SKILL_SLOTS = 4
MAX_DAMAGE = 50000.0
MAX_REVIVE_TIME = 60000.0


def _clip(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def _safe_div(value, denom):
    if denom <= 0:
        return 0.0
    return value / denom


class HeroProcess:
    def __init__(self, camp):
        self.main_camp = camp
        self.transform_camp2_to_camp1 = camp == 2

    def reset(self, camp):
        self.__init__(camp)

    def process_vec_hero(self, frame_state):
        main_hero, enemy_hero = self._get_heroes(frame_state)
        feature = []
        feature.extend(self._main_hero_feature(main_hero))
        feature.extend(self._enemy_hero_feature(main_hero, enemy_hero))
        feature.extend(self._skill_feature(main_hero))
        feature.extend(self._matchup_feature(main_hero, enemy_hero))
        return feature

    def _get_heroes(self, frame_state):
        main_hero, enemy_hero = None, None
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                main_hero = hero
            else:
                enemy_hero = hero
        return main_hero, enemy_hero

    def _main_hero_feature(self, hero):
        if hero is None:
            return [0.0] * 18

        return (
            self._hero_id_one_hot(hero.get("config_id"))
            + [
                self._is_alive(hero),
                self._hp_rate(hero),
                self._ep_rate(hero),
                _clip(hero.get("level", 0) / MAX_LEVEL),
                _clip(hero.get("exp", 0) / MAX_EXP),
                _clip(hero.get("money", 0) / MAX_MONEY),
                _clip(hero.get("money_cnt", 0) / MAX_MONEY),
                _clip(hero.get("phy_atk", 0) / MAX_ATTACK),
                _clip(hero.get("phy_def", 0) / MAX_DEFENSE),
                _clip(hero.get("mov_spd", 0) / MAX_MOVE_SPEED),
                _clip(hero.get("atk_spd", 0) / MAX_ATTACK_SPEED),
            ]
            + self._normal_position(hero.get("location", {}))
            + [
                _clip(hero.get("kill_cnt", 0) / MAX_KDA),
                _clip(hero.get("dead_cnt", 0) / MAX_KDA),
            ]
        )

    def _enemy_hero_feature(self, main_hero, enemy_hero):
        if enemy_hero is None:
            return [0.0] * 13

        return (
            self._hero_id_one_hot(enemy_hero.get("config_id"))
            + [
                1.0,
                self._is_alive(enemy_hero),
                self._hp_rate(enemy_hero),
                _clip(enemy_hero.get("level", 0) / MAX_LEVEL),
                _clip(enemy_hero.get("money_cnt", 0) / MAX_MONEY),
            ]
            + self._relative_position(main_hero, enemy_hero)
            + [
                self._distance(main_hero, enemy_hero),
                _clip(enemy_hero.get("kill_cnt", 0) / MAX_KDA),
                _clip(enemy_hero.get("dead_cnt", 0) / MAX_KDA),
            ]
        )

    def _skill_feature(self, hero):
        if hero is None:
            return [0.0] * (MAX_SKILL_SLOTS * 2)

        slot_states = hero.get("skill_state", {}).get("slot_states", [])[:MAX_SKILL_SLOTS]
        feature = []
        for slot in slot_states:
            cooldown = slot.get("cooldown", 0)
            cooldown_max = slot.get("cooldown_max", 0)
            cooldown_rate = _safe_div(cooldown, cooldown_max)
            feature.extend([1.0 if slot.get("usable", False) else 0.0, _clip(cooldown_rate)])
        while len(feature) < MAX_SKILL_SLOTS * 2:
            feature.extend([0.0, 1.0])
        return feature

    def _matchup_feature(self, main_hero, enemy_hero):
        if main_hero is None or enemy_hero is None:
            return [0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

        return [
            self._advantage(self._hp_rate(main_hero), self._hp_rate(enemy_hero)),
            self._advantage(main_hero.get("level", 0) / MAX_LEVEL, enemy_hero.get("level", 0) / MAX_LEVEL),
            self._advantage(main_hero.get("money_cnt", 0) / MAX_MONEY, enemy_hero.get("money_cnt", 0) / MAX_MONEY),
            self._advantage(
                main_hero.get("total_hurt_to_hero", 0) / MAX_DAMAGE,
                enemy_hero.get("total_hurt_to_hero", 0) / MAX_DAMAGE,
            ),
            self._advantage(
                enemy_hero.get("total_be_hurt_by_hero", 0) / MAX_DAMAGE,
                main_hero.get("total_be_hurt_by_hero", 0) / MAX_DAMAGE,
            ),
            _clip(main_hero.get("revive_time", 0) / MAX_REVIVE_TIME),
            _clip(enemy_hero.get("revive_time", 0) / MAX_REVIVE_TIME),
            1.0 if main_hero.get("hp", 0) < 0.35 * max(main_hero.get("max_hp", 0), 1) else 0.0,
        ]

    def _hero_id_one_hot(self, hero_id):
        return [1.0 if hero_id == value else 0.0 for value in HERO_IDS]

    def _is_alive(self, hero):
        return 1.0 if hero.get("hp", 0) > 0 else 0.0

    def _hp_rate(self, hero):
        return _clip(_safe_div(hero.get("hp", 0), hero.get("max_hp", 0)))

    def _ep_rate(self, hero):
        return _clip(_safe_div(hero.get("ep", 0), hero.get("max_ep", 0)))

    def _normal_position(self, location):
        x = location.get("x", 100000)
        z = location.get("z", 100000)
        if self.transform_camp2_to_camp1 and x != 100000:
            x = -x
        if self.transform_camp2_to_camp1 and z != 100000:
            z = -z
        return [_clip((x + MAX_POSITION) / (2 * MAX_POSITION)), _clip((z + MAX_POSITION) / (2 * MAX_POSITION))]

    def _relative_position(self, main_hero, target_hero):
        if main_hero is None or target_hero is None:
            return [0.5, 0.5]

        main_location = main_hero.get("location", {})
        target_location = target_hero.get("location", {})
        x_diff = target_location.get("x", 0) - main_location.get("x", 0)
        z_diff = target_location.get("z", 0) - main_location.get("z", 0)
        if self.transform_camp2_to_camp1:
            x_diff = -x_diff
            z_diff = -z_diff
        return [_clip((x_diff + 30000.0) / 60000.0), _clip((z_diff + 30000.0) / 60000.0)]

    def _distance(self, main_hero, target_hero):
        if main_hero is None or target_hero is None:
            return 1.0

        main_location = main_hero.get("location", {})
        target_location = target_hero.get("location", {})
        dist = math.dist(
            (main_location.get("x", 0), main_location.get("z", 0)),
            (target_location.get("x", 0), target_location.get("z", 0)),
        )
        return _clip(dist / MAX_DISTANCE)

    def _advantage(self, main_value, enemy_value):
        return _clip((main_value - enemy_value + 1.0) / 2.0)
