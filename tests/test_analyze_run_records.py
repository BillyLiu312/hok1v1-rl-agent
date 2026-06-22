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
                    "checkpoint": {"actual_train_global_step": 15000},
                    "evaluation": {"eval_id": 7, "checkpoint_step": 15000, "repeat_index": 1},
                    "frame_no": 12000,
                    "reward_sum": [10.0, -10.0],
                    "reward_detail": [
                        {
                            "enemy_tower_hp_down": 0.3,
                            "self_tower_hp_down": -0.1,
                            "push_window_tower_damage": 0.2,
                            "unsafe_dive": -1.0,
                            "push_window_active": 12.0,
                            "unsafe_dive_active": 3.0,
                            "win_result": 1.0,
                            "timeout_tower_gap": 0.0,
                        },
                        {},
                    ],
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
            event_2 = json.loads(json.dumps(event))
            event_2["payload"]["agents"][0]["hero"]["dead_cnt"] = 3
            event_2["payload"]["evaluation"]["eval_id"] = 8
            event_2["payload"]["evaluation"]["repeat_index"] = 2
            event_2["payload"]["reward_detail"][0]["enemy_tower_hp_down"] = 0.1
            event_2["payload"]["reward_detail"][0]["push_window_tower_damage"] = 0.1
            event_2["payload"]["reward_detail"][0]["unsafe_dive_active"] = 9.0
            (record_dir / "episode_end-unit-1.jsonl").write_text(
                json.dumps(event) + "\n" + json.dumps(event_2) + "\n",
                encoding="utf-8",
            )

            rows = collect_rows(record_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["checkpoint_step"], 15000)
            self.assertEqual(rows[0]["eval_ids"], "7,8")
            self.assertEqual(rows[0]["evaluation_checkpoint_step"], 15000)
            self.assertEqual(rows[0]["repeat_indices"], "1,2")
            self.assertEqual(rows[0]["matchup"], "199_vs_133")
            self.assertEqual(rows[0]["win_rate"], 1.0)
            self.assertEqual(rows[0]["avg_enemy_tower_hp"], 0)
            self.assertAlmostEqual(rows[0]["avg_push_window_tower_damage"], 0.15)
            self.assertEqual(rows[0]["avg_unsafe_dive"], -1.0)
            self.assertEqual(rows[0]["avg_push_window_active_frames"], 12.0)
            self.assertEqual(rows[0]["avg_unsafe_dive_active_frames"], 6.0)
            self.assertAlmostEqual(rows[0]["push_window_tower_damage_share"], 0.75)
            self.assertAlmostEqual(rows[0]["unsafe_dive_death_corr"], 1.0)

            csv_path = record_dir / "summary.csv"
            md_path = record_dir / "summary.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path, "Run Summary")
            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("199_vs_133", markdown)
            self.assertIn("eval_ids", markdown)
            self.assertIn("avg_push_window_tower_damage", markdown)
            self.assertIn("push_window_tower_damage_share", markdown)
            self.assertIn("unsafe_dive_death_corr", markdown)
            self.assertIn("avg_unsafe_dive_active_frames", markdown)

    def test_collect_handles_missing_reward_sum_for_monitor_side(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            record_dir = Path(temp_dir)
            event = {
                "stream": "episode_end",
                "payload": {
                    "monitor_agent_index": 1,
                    "monitor_hero_id": 133,
                    "opponent_hero_id": 199,
                    "is_eval": True,
                    "opponent_agent": "common_ai",
                    "checkpoint": {"actual_train_global_step": 15000},
                    "frame_no": 10000,
                    "reward_sum": [10.0],
                    "agents": [{}, {"win": 0, "hero": {"config_id": 133}, "enemy_hero": {"config_id": 199}}],
                },
            }
            (record_dir / "episode_end-unit-1.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

            rows = collect_rows(record_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["matchup"], "133_vs_199")
            self.assertEqual(rows[0]["avg_reward_sum"], "")


if __name__ == "__main__":
    unittest.main()
