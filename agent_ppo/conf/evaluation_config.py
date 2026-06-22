#!/usr/bin/env python3
"""
Helpers for preserving externally supplied evaluation configuration.
"""


PRESET_SKILL_KEYS = ("select_skill", "summoner_skill_id", "skill_id")


def camp_has_preset_skills(usr_conf, camp_key):
    entries = usr_conf.get("lineups", {}).get(camp_key, [])
    if not entries:
        return False
    for entry in entries:
        if not any(key in entry for key in PRESET_SKILL_KEYS):
            return False
    return True
