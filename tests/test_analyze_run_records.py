#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from utils.analyze_run_records import collect_rows, write_csv, write_markdown


class AnalyzeRunRecordsTest(unittest.TestCase):
    def test_collect_matchup_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            record_dir = Path(temp_dir)
            event = {
                "stream": "episode_end",
                "payload": {
                    "monitor_agent_index": 0,
                    "monitor_hero_id": 199,
                    "opponent_hero_id": 133,
                    "is_eval": True,
                    "opponent_agent": "common_ai",
                    "frame_no": 12000,
                    "reward_sum": [10.0, -10.0],
                    "agents": [
                        {
                            "win": 1,
                            "hero": {"config_id": 199, "kill_cnt": 2, "dead_cnt": 1, "money_cnt": 5000},
                            "enemy_hero": {"config_id": 133},
                            "tower": {"hp": 8000},
                            "enemy_tower": {"hp": 0},
                        },
                        {},
                    ],
                },
            }
            (record_dir / "episode_end-unit-1.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

            rows = collect_rows(record_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["matchup"], "199_vs_133")
            self.assertEqual(rows[0]["win_rate"], 1.0)
            self.assertEqual(rows[0]["avg_enemy_tower_hp"], 0)

            csv_path = record_dir / "summary.csv"
            md_path = record_dir / "summary.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path, "Run Summary")
            self.assertIn("199_vs_133", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
