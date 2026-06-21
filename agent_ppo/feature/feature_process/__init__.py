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
from agent_ppo.conf.conf import Config


def _clip(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def _safe_div(numerator, denominator, default=0.0):
    if denominator in (0, None):
        return default
    return numerator / denominator


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        feature = self._process_expanded_feature(frame_state)
        assert len(feature) == Config.FEATURE_DIM, "feature dim mismatch: {}/{}".format(len(feature), Config.FEATURE_DIM)
        return feature

    def _process_expanded_feature(self, frame_state):
        main_hero, enemy_hero = self._select_heroes(frame_state)
        main_tower, enemy_tower = self._select_towers(frame_state)

        feature = []
        feature.extend(self._hero_feature(main_hero, enemy_hero, enemy_tower))
        feature.extend(self._hero_feature(enemy_hero, main_hero, main_tower))
        feature.extend(self._tower_feature(main_tower, main_hero))
        feature.extend(self._tower_feature(enemy_tower, main_hero))
        feature.extend(self._pair_feature(main_hero, enemy_hero, main_tower, enemy_tower, frame_state))
        feature.extend(self._skill_feature(main_hero, 6))
        return feature

    def _select_heroes(self, frame_state):
        main_hero, enemy_hero = None, None
        for hero in frame_state.get("hero_states", []):
            if hero.get("camp") == self.camp and main_hero is None:
                main_hero = hero
            elif hero.get("camp") != self.camp and enemy_hero is None:
                enemy_hero = hero
        return main_hero or {}, enemy_hero or {}

    def _select_towers(self, frame_state):
        main_tower, enemy_tower = None, None
        for organ in frame_state.get("npc_states", []):
            if organ.get("sub_type") != 21:
                continue
            if organ.get("camp") == self.camp and main_tower is None:
                main_tower = organ
            elif organ.get("camp") != self.camp and enemy_tower is None:
                enemy_tower = organ
        return main_tower or {}, enemy_tower or {}

    def _hero_feature(self, hero, other_hero, enemy_tower):
        location = hero.get("location", {})
        other_location = other_hero.get("location", {})
        tower_location = enemy_tower.get("location", {})
        hp_rate = self._hp_rate(hero)
        return [
            1.0 if hero.get("hp", 0) > 0 else 0.0,
            hp_rate,
            _clip(hero.get("level", hero.get("lv", 1)) / 15.0),
            _clip(_safe_div(hero.get("exp", 0), hero.get("max_exp", 1))),
            _clip(hero.get("money", hero.get("gold", 0)) / 20000.0),
            self._norm_coord(location.get("x", 0)),
            self._norm_coord(location.get("z", 0)),
            self._norm_distance(location, other_location),
            self._norm_distance(location, tower_location),
        ]

    def _tower_feature(self, tower, main_hero):
        location = tower.get("location", {})
        hero_location = main_hero.get("location", {})
        return [
            1.0 if tower.get("hp", 0) > 0 else 0.0,
            self._hp_rate(tower),
            self._norm_coord(location.get("x", 0)),
            self._norm_coord(location.get("z", 0)),
            self._relative_coord(location.get("x", 0), hero_location.get("x", 0)),
            self._relative_coord(location.get("z", 0), hero_location.get("z", 0)),
        ]

    def _pair_feature(self, main_hero, enemy_hero, main_tower, enemy_tower, frame_state):
        main_location = main_hero.get("location", {})
        enemy_location = enemy_hero.get("location", {})
        main_tower_location = main_tower.get("location", {})
        enemy_tower_location = enemy_tower.get("location", {})
        main_hp_rate = self._hp_rate(main_hero)
        enemy_hp_rate = self._hp_rate(enemy_hero)
        main_level = main_hero.get("level", main_hero.get("lv", 1))
        enemy_level = enemy_hero.get("level", enemy_hero.get("lv", 1))
        return [
            1.0 if enemy_hero else 0.0,
            _clip((main_hp_rate - enemy_hp_rate + 1.0) / 2.0),
            _clip((main_level - enemy_level + 15.0) / 30.0),
            self._norm_distance(main_location, enemy_location),
            self._norm_distance(main_location, enemy_tower_location),
            self._norm_distance(main_location, main_tower_location),
            self._norm_distance(enemy_location, main_tower_location),
            self._lane_progress(main_location, main_tower_location, enemy_tower_location),
            self._lane_progress(enemy_location, enemy_tower_location, main_tower_location),
            1.0 if self._distance(main_location, enemy_tower_location) < 6500 else 0.0,
            1.0 if self._distance(main_location, main_tower_location) < 6500 else 0.0,
            1.0 if main_hero.get("hp", 0) <= 0 else 0.0,
            1.0 if enemy_hero.get("hp", 0) <= 0 else 0.0,
            _clip(frame_state.get("frame_no", 0) / 18000.0),
        ]

    def _skill_feature(self, hero, count):
        skills = hero.get("skill_state") or hero.get("skill_states") or hero.get("skills") or hero.get("skill_list") or []
        if isinstance(skills, dict):
            skills = list(skills.values())

        result = []
        for skill in list(skills)[:count]:
            if not isinstance(skill, dict):
                result.append(0.0)
                continue
            if "usable" in skill:
                result.append(1.0 if skill.get("usable") else 0.0)
                continue
            cooldown = skill.get("cooldown", skill.get("cool_down", skill.get("cd", skill.get("remaining_cooldown", 0))))
            max_cooldown = skill.get("max_cooldown", skill.get("max_cd", cooldown if cooldown else 1))
            result.append(_clip(1.0 - _safe_div(cooldown, max_cooldown)))

        while len(result) < count:
            result.append(0.0)
        return result

    def _hp_rate(self, unit):
        return _clip(_safe_div(unit.get("hp", 0), unit.get("max_hp", 1)))

    def _norm_coord(self, value):
        if self.camp == 2 and value != 100000:
            value = -value
        return _clip((value + 60000.0) / 120000.0)

    def _relative_coord(self, target_value, origin_value):
        if self.camp == 2 and target_value != 100000:
            target_value = -target_value
            origin_value = -origin_value
        return _clip((target_value - origin_value + 30000.0) / 60000.0)

    def _distance(self, pos1, pos2):
        if not pos1 or not pos2:
            return 0.0
        dx = pos1.get("x", 0) - pos2.get("x", 0)
        dz = pos1.get("z", 0) - pos2.get("z", 0)
        return (dx * dx + dz * dz) ** 0.5

    def _norm_distance(self, pos1, pos2):
        return _clip(self._distance(pos1, pos2) / 120000.0)

    def _lane_progress(self, hero_pos, own_tower_pos, enemy_tower_pos):
        full_dist = self._distance(own_tower_pos, enemy_tower_pos)
        if full_dist <= 0:
            return 0.0
        dist_to_enemy = self._distance(hero_pos, enemy_tower_pos)
        return _clip(1.0 - dist_to_enemy / full_dist)
