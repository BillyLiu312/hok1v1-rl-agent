#!/usr/bin/env python3

import unittest

from agent_ppo.conf.evaluation_config import camp_has_preset_skills


class EvaluationConfigTest(unittest.TestCase):
    def test_detects_select_skill_presets(self):
        usr_conf = {"lineups": {"blue_camp": [{"hero_id": 112, "select_skill": 80115}]}}
        self.assertTrue(camp_has_preset_skills(usr_conf, "blue_camp"))

    def test_detects_legacy_summoner_skill_presets(self):
        usr_conf = {"lineups": {"blue_camp": [{"hero_id": 112, "summoner_skill_id": 80115}]}}
        self.assertTrue(camp_has_preset_skills(usr_conf, "blue_camp"))

    def test_requires_every_entry_to_have_skill(self):
        usr_conf = {
            "lineups": {
                "blue_camp": [
                    {"hero_id": 112, "select_skill": 80115},
                    {"hero_id": 133},
                ]
            }
        }
        self.assertFalse(camp_has_preset_skills(usr_conf, "blue_camp"))

    def test_missing_camp_is_not_preset(self):
        self.assertFalse(camp_has_preset_skills({}, "blue_camp"))


if __name__ == "__main__":
    unittest.main()
