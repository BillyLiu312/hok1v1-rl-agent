#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from utils.opponent_curriculum_summary import collect_rows, write_csv, write_markdown


class OpponentCurriculumSummaryTest(unittest.TestCase):
    def test_collects_actual_opponent_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            record_dir = Path(temp_dir)
            base_event = {
                "stream": "episode_end",
                "payload": {
                    "monitor_agent_index": 0,
                    "monitor_hero_id": 199,
                    "opponent_hero_id": 133,
                    "configured_opponent_agent": "curriculum",
                    "opponent_agent": "100",
                    "opponent_source": "historical",
                    "is_eval": False,
                    "checkpoint": {"actual_train_global_step": 15000},
                    "frame_no": 12000,
                    "reward_sum": [10.0, -10.0],
                    "agents": [
                        {
                            "win": 1,
                            "hero": {"config_id": 199, "dead_cnt": 1},
                            "enemy_hero": {"config_id": 133},
                            "tower": {"hp": 8000},
                            "enemy_tower": {"hp": 0},
                        },
                        {},
                    ],
                },
            }
            second_event = json.loads(json.dumps(base_event))
            second_event["payload"]["opponent_agent"] = "common_ai"
            second_event["payload"]["opponent_source"] = "common_ai"
            second_event["payload"]["agents"][0]["win"] = 0
            second_event["payload"]["agents"][0]["hero"]["dead_cnt"] = 3
            second_event["payload"]["agents"][0]["enemy_tower"]["hp"] = 3000
            (record_dir / "episode_end-unit-1.jsonl").write_text(
                json.dumps(base_event) + "\n" + json.dumps(second_event) + "\n",
                encoding="utf-8",
            )

            rows = collect_rows(record_dir)

            self.assertEqual(len(rows), 2)
            historical = next(row for row in rows if row["opponent_source"] == "historical")
            common_ai = next(row for row in rows if row["opponent_source"] == "common_ai")
            self.assertEqual(historical["configured_opponent_agent"], "curriculum")
            self.assertEqual(historical["opponent_agent"], "100")
            self.assertEqual(historical["episodes"], 1)
            self.assertEqual(historical["win_rate"], 1.0)
            self.assertEqual(common_ai["win_rate"], 0.0)
            self.assertEqual(common_ai["avg_death"], 3.0)

            csv_path = record_dir / "curriculum.csv"
            md_path = record_dir / "curriculum.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path, "Curriculum")
            self.assertIn("opponent_source", csv_path.read_text(encoding="utf-8"))
            self.assertIn("historical", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
