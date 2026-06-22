#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from utils.summoner_skill_policy_patch import build_patch_rows, read_rows, write_markdown, write_python_patch


class SummonerSkillPolicyPatchTest(unittest.TestCase):
    def test_build_patch_rows_filters_reviewable_policy_updates(self):
        rows = [
            {
                "checkpoint_step": "15000",
                "matchup": "199_vs_133",
                "recommended_skill": "80110",
                "recommended_skill_name": "狂暴",
                "recommended_win_rate": "0.7",
                "recommended_avg_death": "3",
                "recommended_avg_enemy_tower_hp": "1000",
                "recommended_episodes": "20",
                "current_policy_skill": "80107",
                "current_policy_skill_name": "净化",
                "current_policy_win_rate": "0.55",
                "current_policy_avg_death": "2",
                "current_policy_avg_enemy_tower_hp": "2000",
                "recommendation_score": "65",
                "needs_policy_update": "True",
            },
            {
                "checkpoint_step": "15000",
                "matchup": "112_vs_112",
                "recommended_skill": "80115",
                "recommended_episodes": "20",
                "current_policy_skill": "80115",
                "needs_policy_update": "False",
            },
        ]

        patch_rows = build_patch_rows(rows, min_episodes=10, min_win_delta=0.05)

        self.assertEqual(len(patch_rows), 1)
        self.assertEqual(patch_rows[0]["hero_id"], 199)
        self.assertEqual(patch_rows[0]["opponent_hero_id"], 133)
        self.assertEqual(patch_rows[0]["recommended_skill"], 80110)
        self.assertEqual(patch_rows[0]["current_policy_avg_death"], "2")
        self.assertAlmostEqual(patch_rows[0]["win_rate_delta"], 0.15)

    def test_write_policy_patch_outputs_reviewable_dict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = [
                {
                    "checkpoint_step": "15000",
                    "hero_id": 199,
                    "opponent_hero_id": 133,
                    "recommended_skill": 80110,
                    "recommended_skill_name": "狂暴",
                    "current_policy_skill": 80107,
                    "current_policy_skill_name": "净化",
                    "recommended_win_rate": 0.7,
                    "current_policy_win_rate": 0.55,
                    "win_rate_delta": 0.15,
                    "recommended_avg_death": "3",
                    "current_policy_avg_death": "2",
                    "recommended_avg_enemy_tower_hp": "1000",
                    "current_policy_avg_enemy_tower_hp": "2000",
                    "recommended_episodes": "20",
                    "recommendation_score": "65",
                }
            ]
            py_path = root / "patch.py"
            md_path = root / "patch.md"

            write_python_patch(rows, py_path)
            write_markdown(rows, md_path)

            self.assertIn("MATCHUP_SUMMONER_SKILL_OVERRIDES_CANDIDATE", py_path.read_text(encoding="utf-8"))
            self.assertIn("(199, 133): 80110", py_path.read_text(encoding="utf-8"))
            self.assertIn("win_delta=0.15", py_path.read_text(encoding="utf-8"))
            self.assertIn("recommended_skill_name", md_path.read_text(encoding="utf-8"))
            self.assertIn("current_policy_avg_death", md_path.read_text(encoding="utf-8"))

    def test_read_rows_handles_missing_file(self):
        self.assertEqual(read_rows(Path("/tmp/definitely-missing-summoner-policy.csv")), [])


if __name__ == "__main__":
    unittest.main()
