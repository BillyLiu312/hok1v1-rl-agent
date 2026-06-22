#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math


TOWER_SUB_TYPE = 21
MAX_SOLDIER_COUNT = 8.0
MAX_HP_SUM = 30000.0
MAX_DISTANCE = 120000.0
MAX_POSITION = 60000.0


def _clip(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def _safe_div(value, denom):
    if denom <= 0:
        return 0.0
    return value / denom


class SoldierProcess:
    def __init__(self, camp):
        self.main_camp = camp
        self.transform_camp2_to_camp1 = camp == 2

    def reset(self, camp):
        self.__init__(camp)

    def process_vec_soldier(self, frame_state):
        main_hero = self._get_main_hero(frame_state)
        main_units, enemy_units = self._get_lane_units(frame_state)
        main_summary = self._summarize(main_hero, main_units)
        enemy_summary = self._summarize(main_hero, enemy_units)
        lane_hp_adv = _clip((main_summary["hp_sum"] - enemy_summary["hp_sum"] + MAX_HP_SUM) / (2 * MAX_HP_SUM))

        return [
            _clip(main_summary["count"] / MAX_SOLDIER_COUNT),
            _clip(enemy_summary["count"] / MAX_SOLDIER_COUNT),
            _clip(main_summary["hp_sum"] / MAX_HP_SUM),
            _clip(enemy_summary["hp_sum"] / MAX_HP_SUM),
            main_summary["nearest_dist"],
            enemy_summary["nearest_dist"],
            main_summary["forward_mean"],
            enemy_summary["forward_mean"],
            enemy_summary["lowest_hp_rate"],
            lane_hp_adv,
        ]

    def _get_main_hero(self, frame_state):
        for hero in frame_state["hero_states"]:
            if hero.get("camp") == self.main_camp:
                return hero
        return None

    def _get_lane_units(self, frame_state):
        main_units, enemy_units = [], []
        for npc in frame_state["npc_states"]:
            if not self._is_lane_unit(npc):
                continue
            if npc.get("camp") == self.main_camp:
                main_units.append(npc)
            else:
                enemy_units.append(npc)
        return main_units, enemy_units

    def _is_lane_unit(self, npc):
        return npc.get("sub_type") != TOWER_SUB_TYPE and npc.get("hp", 0) > 0 and npc.get("camp") in (1, 2)

    def _summarize(self, main_hero, units):
        if not units:
            return {
                "count": 0.0,
                "hp_sum": 0.0,
                "nearest_dist": 1.0,
                "forward_mean": 0.5,
                "lowest_hp_rate": 1.0,
            }

        distances = [self._distance(main_hero, unit) for unit in units]
        hp_rates = [_clip(_safe_div(unit.get("hp", 0), unit.get("max_hp", 0))) for unit in units]
        forward_values = [self._forward_position(unit) for unit in units]

        return {
            "count": float(len(units)),
            "hp_sum": float(sum(unit.get("hp", 0) for unit in units)),
            "nearest_dist": min(distances) if distances else 1.0,
            "forward_mean": sum(forward_values) / len(forward_values),
            "lowest_hp_rate": min(hp_rates) if hp_rates else 1.0,
        }

    def _distance(self, main_hero, unit):
        if main_hero is None:
            return 1.0

        main_location = main_hero.get("location", {})
        unit_location = unit.get("location", {})
        dist = math.dist(
            (main_location.get("x", 0), main_location.get("z", 0)),
            (unit_location.get("x", 0), unit_location.get("z", 0)),
        )
        return _clip(dist / MAX_DISTANCE)

    def _forward_position(self, unit):
        location = unit.get("location", {})
        x = location.get("x", 0)
        z = location.get("z", 0)
        if self.transform_camp2_to_camp1:
            x = -x
            z = -z
        return _clip(((x + z) / 2.0 + MAX_POSITION) / (2 * MAX_POSITION))
