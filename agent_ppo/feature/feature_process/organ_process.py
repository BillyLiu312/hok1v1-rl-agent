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
MAX_POSITION = 60000.0
MAX_DISTANCE = 120000.0


def _clip(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def _safe_div(value, denom):
    if denom <= 0:
        return 0.0
    return value / denom


class OrganProcess:
    def __init__(self, camp):
        self.main_camp = camp
        self.transform_camp2_to_camp1 = camp == 2

    def reset(self, camp):
        self.__init__(camp)

    def process_vec_organ(self, frame_state):
        main_hero = self._get_main_hero(frame_state)
        main_tower, enemy_tower = self._get_towers(frame_state)
        return self._tower_feature(main_hero, main_tower) + self._tower_feature(main_hero, enemy_tower)

    def _get_main_hero(self, frame_state):
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                return hero
        return None

    def _get_towers(self, frame_state):
        main_tower, enemy_tower = None, None
        for npc in frame_state["npc_states"]:
            if npc.get("sub_type") != TOWER_SUB_TYPE:
                continue
            if npc.get("camp") == self.main_camp:
                main_tower = npc
            else:
                enemy_tower = npc
        return main_tower, enemy_tower

    def _tower_feature(self, main_hero, tower):
        if tower is None:
            return [0.0] * 7

        return [
            1.0 if tower.get("hp", 0) > 0 else 0.0,
            _clip(_safe_div(tower.get("hp", 0), tower.get("max_hp", 0))),
        ] + self._normal_position(tower.get("location", {})) + self._relative_position(main_hero, tower) + [
            self._distance(main_hero, tower)
        ]

    def _normal_position(self, location):
        x = location.get("x", 100000)
        z = location.get("z", 100000)
        if self.transform_camp2_to_camp1 and x != 100000:
            x = -x
        if self.transform_camp2_to_camp1 and z != 100000:
            z = -z
        return [_clip((x + MAX_POSITION) / (2 * MAX_POSITION)), _clip((z + MAX_POSITION) / (2 * MAX_POSITION))]

    def _relative_position(self, main_hero, tower):
        if main_hero is None or tower is None:
            return [0.5, 0.5]

        main_location = main_hero.get("location", {})
        tower_location = tower.get("location", {})
        x_diff = tower_location.get("x", 0) - main_location.get("x", 0)
        z_diff = tower_location.get("z", 0) - main_location.get("z", 0)
        if self.transform_camp2_to_camp1:
            x_diff = -x_diff
            z_diff = -z_diff
        return [_clip((x_diff + 30000.0) / 60000.0), _clip((z_diff + 30000.0) / 60000.0)]

    def _distance(self, main_hero, tower):
        if main_hero is None or tower is None:
            return 1.0

        main_location = main_hero.get("location", {})
        tower_location = tower.get("location", {})
        dist = math.dist(
            (main_location.get("x", 0), main_location.get("z", 0)),
            (tower_location.get("x", 0), tower_location.get("z", 0)),
        )
        return _clip(dist / MAX_DISTANCE)
