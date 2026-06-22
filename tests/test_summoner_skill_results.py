#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from utils.summoner_skill_results import collect_rows, write_csv, write_markdown


def make_event(skill_id, win):
    return {
        "stream": "episode_end",
        "payload": {
            "monitor_agent_index": 0,
            "monitor_hero_id": 199,
            "opponent_hero_id": 133,
            "is_eval": True,
            "opponent_agent": "common_ai",
            "checkpoint": {"actual_train_global_step": 15000},
            "usr_conf": {
                "lineups": {
                    "blue_camp": [{"hero_id": 199, "select_skill": skill_id}],
                    "red_camp": [{"hero_id": 133, "select_skill": 80110}],
                }
            },
            "frame_no": 12000,
            "reward_sum": [10.0, -10.0],
            "agents": [
                {
                    "win": win,
                    "hero": {"config_id": 199, "kill_cnt": 2, "dead_cnt": 1, "money_cnt": 5000},
                    "enemy_hero": {"config_id": 133},
                    "tower": {"hp": 8000},
                    "enemy_tower": {"hp": 0},
                },
                {},
            ],
        },
    }


class SummonerSkillResultsTest(unittest.TestCase):
    def test_collect_summoner_skill_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            record_dir = Path(temp_dir)
            events = [make_event("80107", 1), make_event(80107, 0), make_event("80110", 1)]
            (record_dir / "episode_end-unit-1.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            rows = collect_rows(record_dir)
            self.assertEqual(len(rows), 2)

            purify = [row for row in rows if row["monitor_skill"] == 80107][0]
            self.assertEqual(purify["matchup"], "199_vs_133")
            self.assertEqual(purify["monitor_skill_name"], "净化")
            self.assertTrue(purify["is_current_policy_skill"])
            self.assertEqual(purify["episodes"], 2)
            self.assertEqual(purify["win_rate"], 0.5)

            frenzy = [row for row in rows if row["monitor_skill"] == 80110][0]
            self.assertFalse(frenzy["is_current_policy_skill"])
            self.assertEqual(frenzy["win_rate"], 1.0)

            csv_path = record_dir / "skills.csv"
            md_path = record_dir / "skills.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path, "Skill Results")
            self.assertIn("monitor_skill_name", csv_path.read_text(encoding="utf-8"))
            self.assertIn("199_vs_133", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
