#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from utils.build_experiment_report import build_report, filter_rows_for_checkpoint, manifest_value


class BuildExperimentReportTest(unittest.TestCase):
    def test_manifest_value_preserves_zero(self):
        self.assertEqual(manifest_value(0), 0)
        self.assertEqual(manifest_value(0.0), 0.0)
        self.assertEqual(manifest_value(None), "")

    def test_filter_rows_for_checkpoint_matches_exact_step(self):
        rows = [
            {"checkpoint_step": 15000, "matchup": "199_vs_133"},
            {"checkpoint_step": 15039, "matchup": "133_vs_199"},
            {"checkpoint_step": "17057", "matchup": "112_vs_112"},
        ]
        candidate = {"checkpoint_step": "15000", "actual_train_global_step": 15039}
        self.assertEqual(filter_rows_for_checkpoint(rows, candidate), [rows[0], rows[1]])
        self.assertEqual(filter_rows_for_checkpoint(rows, None), rows)

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
            launch_manifest = root / "launch_manifest.json"
            launch_manifest.write_text(
                json.dumps(
                    {
                        "stage": "v1.2-a",
                        "run_id": "unit-launch",
                        "git_commit": "abc123",
                        "preflight_status": "PASS",
                        "sync_package_sha256": "f" * 64,
                        "env": {
                            "HOK_TRAINING_RECORD_DIR": "logs/run_records/unit",
                            "HOK_TRAINING_RUN_ID": "unit-train",
                            "HOK_REWARD_PROFILE": "v1.2",
                            "HOK_REWARD_WEIGHT_OVERRIDES": "death:5",
                            "HOK_OPPONENT_SCHEDULE": "common_ai:4,historical:4,selfplay:2",
                        },
                    }
                ),
                encoding="utf-8",
            )

            artifacts = build_report(
                log_dir=log_dir,
                output_dir=output_dir,
                launch_manifest=launch_manifest,
                checkpoints=[15000],
                heroes=[112],
                repeats=1,
            )

            for path in artifacts.values():
                self.assertTrue(path.exists(), path)
            manifest = artifacts["manifest"].read_text(encoding="utf-8")
            self.assertIn("recommended_checkpoint: 15000", manifest)
            self.assertIn("evaluation_rows: 2", manifest)
            self.assertIn("evaluation_skill_pairs: 1", manifest)
            self.assertIn("candidate_gate_status: FAIL", manifest)
            self.assertIn("launch_run_id: unit-launch", manifest)
            self.assertIn("launch_git_commit: abc123", manifest)
            self.assertIn("launch_sync_package_sha256: " + "f" * 64, manifest)
            self.assertIn("launch_training_record_dir: logs/run_records/unit", manifest)
            self.assertIn("launch_training_run_id: unit-train", manifest)
            self.assertIn("launch_reward_profile: v1.2", manifest)
            self.assertIn("launch_reward_weight_overrides: death:5", manifest)
            self.assertIn("launch_opponent_schedule: common_ai:4,historical:4,selfplay:2", manifest)
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
                    "checkpoint": {"actual_train_global_step": 15039},
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
            other_checkpoint_event = json.loads(json.dumps(event))
            other_checkpoint_event["payload"]["monitor_hero_id"] = 112
            other_checkpoint_event["payload"]["opponent_hero_id"] = 112
            other_checkpoint_event["payload"]["checkpoint"] = {"actual_train_global_step": 17057}
            other_checkpoint_event["payload"]["agents"][0]["hero"]["config_id"] = 112
            other_checkpoint_event["payload"]["agents"][0]["enemy_hero"]["config_id"] = 112
            (record_dir / "episode_end-unit-1.jsonl").write_text(
                json.dumps(event) + "\n" + json.dumps(other_checkpoint_event) + "\n",
                encoding="utf-8",
            )

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
            self.assertIn("recommended_matchup_rows", manifest)
            self.assertIn("recommended_push_window_tower_damage_share", manifest)
            candidate_gate = artifacts["v1.2_candidate_gate_csv"].read_text(encoding="utf-8")
            self.assertIn("raw_matchup_rows,1", candidate_gate)

    def test_build_report_can_expand_skill_grid_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_dir = root / "logs"
            output_dir = root / "report"
            log_dir.mkdir()
            (log_dir / "step-15000.md").write_text(
                """# Training Record

## Sampling

- target_step: 15000

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
                heroes=[199],
                repeats=1,
                include_skill_grid=True,
                skills=[80107, 80110],
            )

            manifest = artifacts["manifest"].read_text(encoding="utf-8")
            self.assertIn("evaluation_rows: 8", manifest)
            self.assertIn("evaluation_skill_pairs: 4", manifest)
            self.assertIn("80107", artifacts["evaluation_matrix_csv"].read_text(encoding="utf-8"))
            self.assertIn("80110", artifacts["summoner_skill_grid_csv"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
