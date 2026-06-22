#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from utils.v1_2_baseline import build_baseline, write_json, write_markdown


class V12BaselineTest(unittest.TestCase):
    def test_build_baseline_selects_v1_1_reference_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            (log_dir / "step-15000.md").write_text(
                """# Training Record

## Sampling

- target_step: 15000

## Environment Metrics: Common AI

- win_rate: 0.84
- enemy_tower_hp: 1400.59
- death: 2.72
""",
                encoding="utf-8",
            )
            (log_dir / "step-17057.md").write_text(
                """# Training Record

## Sampling

- target_step: 17057

## Environment Metrics: Common AI

- win_rate: 0.81
- enemy_tower_hp: 1600
- death: 3.09
""",
                encoding="utf-8",
            )

            baseline = build_baseline(log_dir)

            self.assertEqual(baseline["status"], "PASS")
            self.assertEqual(baseline["rows"], 2)
            self.assertEqual(baseline["best_win_step"], 15000)
            self.assertEqual(baseline["best_win_rate"], 0.84)
            self.assertEqual(baseline["best_tower_step"], 15000)
            self.assertEqual(baseline["best_enemy_tower_hp"], 1400.59)
            self.assertEqual(baseline["late_step"], 17057)
            self.assertEqual(baseline["late_death"], 3.09)

            json_path = log_dir / "baseline.json"
            md_path = log_dir / "baseline.md"
            write_json(baseline, json_path)
            write_markdown(baseline, md_path)
            self.assertIn("best_win_rate", json_path.read_text(encoding="utf-8"))
            self.assertIn("# v1.1 Baseline For v1.2", md_path.read_text(encoding="utf-8"))

    def test_missing_common_ai_rows_is_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            (log_dir / "step-1000.md").write_text("# Training Record\n", encoding="utf-8")

            baseline = build_baseline(log_dir)

            self.assertEqual(baseline["status"], "MISSING")
            self.assertIn("reason", baseline)


if __name__ == "__main__":
    unittest.main()
