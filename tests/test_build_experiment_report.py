#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from utils.build_experiment_report import build_report


class BuildExperimentReportTest(unittest.TestCase):
    def test_build_report_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_dir = root / "logs"
            output_dir = root / "report"
            log_dir.mkdir()
            (log_dir / "step-15000.md").write_text(
                """# Training Record

## Sampling

- target_step: 15000
- actual_train_global_step: 15039

## Environment Metrics: Common AI

- win_rate: 0.84
- enemy_tower_hp: 1400.59
- self_tower_hp: 7745.34
- death: 2.72
""",
                encoding="utf-8",
            )

            artifacts = build_report(
                log_dir=log_dir,
                output_dir=output_dir,
                checkpoints=[15000],
                heroes=[112],
                repeats=1,
            )

            for path in artifacts.values():
                self.assertTrue(path.exists(), path)
            manifest = artifacts["manifest"].read_text(encoding="utf-8")
            self.assertIn("recommended_checkpoint: 15000", manifest)
            self.assertIn("evaluation_rows: 2", manifest)
            self.assertIn("checkpoint_ranking_csv", manifest)
            self.assertIn("v1.2_candidate_gate_csv", manifest)

    def test_build_report_includes_run_record_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_dir = root / "logs"
            record_dir = root / "records"
            output_dir = root / "report"
            log_dir.mkdir()
            record_dir.mkdir()
            (log_dir / "step-15000.md").write_text(
                """# Training Record

## Sampling

- target_step: 15000
- actual_train_global_step: 15039

## Environment Metrics: Common AI

- win_rate: 0.84
- enemy_tower_hp: 1400.59
- self_tower_hp: 7745.34
- death: 2.72
""",
                encoding="utf-8",
            )
            event = {
                "stream": "episode_end",
                "payload": {
                    "monitor_agent_index": 0,
                    "monitor_hero_id": 199,
                    "opponent_hero_id": 133,
                    "is_eval": True,
                    "opponent_agent": "17057",
                    "checkpoint": {"actual_train_global_step": 15000},
                    "usr_conf": {
                        "lineups": {
                            "blue_camp": [{"hero_id": 199, "select_skill": "80107"}],
                            "red_camp": [{"hero_id": 133, "select_skill": 80110}],
                        }
                    },
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

            artifacts = build_report(
                log_dir=log_dir,
                record_dir=record_dir,
                output_dir=output_dir,
                checkpoints=[15000],
                heroes=[199],
                repeats=1,
            )

            self.assertTrue(artifacts["matchup_summary_csv"].exists())
            self.assertTrue(artifacts["checkpoint_matrix_csv"].exists())
            self.assertTrue(artifacts["summoner_skill_results_csv"].exists())
            self.assertTrue(artifacts["v1.2_candidate_gate_csv"].exists())
            self.assertIn(
                "push_window_tower_damage_share",
                artifacts["matchup_summary_csv"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "matchup_avg_unsafe_dive_death_corr",
                artifacts["checkpoint_ranking_csv"].read_text(encoding="utf-8"),
            )
            manifest = artifacts["manifest"].read_text(encoding="utf-8")
            self.assertIn("checkpoint_matrix_csv", manifest)
            self.assertIn("summoner_skill_results_csv", manifest)


if __name__ == "__main__":
    unittest.main()
