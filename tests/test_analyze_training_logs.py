#!/usr/bin/env python3

import csv
import tempfile
import unittest
from pathlib import Path

from utils.analyze_training_logs import collect_rows, write_csv, write_markdown


class AnalyzeTrainingLogsTest(unittest.TestCase):
    def test_collect_rows_and_write_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            (log_dir / "step-100.md").write_text(
                """# Training Record

## Sampling

- target_step: 100
- actual_train_global_step: 98

## Basic Metrics

- episode_cnt: 12
- sample_receive_cnt: 345

## Environment Metrics: Common AI

- win_rate: 0.5
- self_tower_hp: 6000
- enemy_tower_hp: 3000
- frame: 12000
- kill: 1
- death: 2
- money_per_frame: 0.4
- hurt_to_hero: 1.2
- hurt_by_hero: 0.7

## Environment Metrics: Selfplay

- win_rate: 0.25
- hurt_to_hero: 0.4
- hurt_by_hero: 0.6

## Algorithm Metrics

- reward: 3.5
- total_loss: 0.1

## Reward Detail Metrics

- enemy_tower_hp_down: 0.3
- self_tower_hp_down: -0.1
- push_window_tower_damage: 0.2
- unsafe_dive: -1.0
- unsafe_dive_severity: -1.5
- push_window_active: 12
- unsafe_dive_active: 3
- win_result: 1
- timeout_tower_gap: 0
""",
                encoding="utf-8",
            )

            rows = collect_rows(log_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], 100)
            self.assertEqual(rows[0]["actual_train_global_step"], 98)
            self.assertEqual(rows[0]["common_ai_win_rate"], 0.5)
            self.assertEqual(rows[0]["common_ai_hurt_to_hero"], 1.2)
            self.assertEqual(rows[0]["common_ai_hurt_by_hero"], 0.7)
            self.assertEqual(rows[0]["selfplay_hurt_to_hero"], 0.4)
            self.assertEqual(rows[0]["selfplay_hurt_by_hero"], 0.6)
            self.assertEqual(rows[0]["reward_push_window_tower_damage"], 0.2)
            self.assertEqual(rows[0]["reward_unsafe_dive_severity"], -1.5)
            self.assertEqual(rows[0]["reward_unsafe_dive_active"], 3)
            self.assertEqual(rows[0]["reward_win_result"], 1)

            csv_path = log_dir / "summary.csv"
            md_path = log_dir / "summary.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path, "Unit Summary")

            with csv_path.open(encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(csv_rows[0]["step"], "100")
            self.assertIn("common_ai_hurt_to_hero", csv_rows[0])
            self.assertIn("reward_push_window_tower_damage", csv_rows[0])
            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("best_common_ai_win_rate: 0.5", markdown)
            self.assertIn("common_ai_hurt_to_hero", markdown)
            self.assertIn("reward_push_window_tower_damage", markdown)


if __name__ == "__main__":
    unittest.main()
