#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


SUMMONER_SKILL_MAP = {
    80102: "治疗",
    80109: "疾跑",
    80104: "惩击",
    80108: "终结",
    80110: "狂暴",
    80105: "干扰",
    80103: "晕眩",
    80107: "净化",
    80121: "弱化",
    80115: "闪现",
}

DEFAULT_SUMMONER_SKILL_BY_HERO = {
    112: 80115,  # Luban No.7: Flash offsets the lack of mobility.
    133: 80110,  # Di Renjie: Frenzy improves lane pressure and tower damage.
    199: 80110,  # Arli: mobility is built in, so prefer sustained damage.
}

MATCHUP_SUMMONER_SKILL_OVERRIDES = {
    (199, 133): 80107,  # Purify helps Arli against Di Renjie's stun.
}


def select_summoner_skill(hero_id, opponent_hero_id=None):
    return MATCHUP_SUMMONER_SKILL_OVERRIDES.get(
        (hero_id, opponent_hero_id),
        DEFAULT_SUMMONER_SKILL_BY_HERO.get(hero_id, 80115),
    )
