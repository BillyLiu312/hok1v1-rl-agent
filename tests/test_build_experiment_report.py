#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from utils.build_experiment_report import (
    DEFAULT_OUTPUT_DIR,
    build_report,
    filter_fixed_eval_rows_for_checkpoint,
    filter_rows_for_checkpoint,
    manifest_value,
    resolve_experiment_metadata,
)


class BuildExperimentReportTest(unittest.TestCase):
    def test_default_output_dir_matches_v1_2_report_convention(self):
        self.assertEqual(DEFAULT_OUTPUT_DIR.as_posix(), "logs/v1.2/report-v1.2")

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

    def test_filter_fixed_eval_rows_for_checkpoint_matches_ranking_scope(self):
        rows = [
            {"checkpoint_step": 15000, "matchup": "199_vs_133", "is_eval": True, "opponent_agent": "common_ai"},
            {"checkpoint_step": 15039, "matchup": "133_vs_199", "is_eval": True, "opponent_agent": "17057"},
            {"checkpoint_step": 15000, "matchup": "112_vs_112", "is_eval": False, "opponent_agent": "common_ai"},
        ]
        candidate = {"checkpoint_step": "15000", "actual_train_global_step": 15039}

        self.assertEqual(filter_fixed_eval_rows_for_checkpoint(rows, candidate), [rows[0]])

    def test_resolve_experiment_metadata_selects_named_ablation(self):
        plan = {
            "stage": "v1.2-a",
            "story": {"research_question": "question", "main_hypothesis": "main"},
            "ablations": [
                {"name": "v1.2", "reward_profile": "v1.2", "hypothesis": "full"},
                {"name": "no_window_reward", "reward_profile": "no_window_reward", "hypothesis": "window"},
            ],
        }

        metadata = resolve_experiment_metadata(plan, "no_window_reward")

        self.assertEqual(metadata["plan_stage"], "v1.2-a")
        self.assertEqual(metadata["experiment_name"], "no_window_reward")
        self.assertEqual(metadata["experiment_reward_profile"], "no_window_reward")
        self.assertEqual(metadata["experiment_hypothesis"], "window")
        self.assertEqual(metadata["success_metrics"], [])

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
- hurt_to_hero: 1.88
- hurt_by_hero: 1.21
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
            experiment_plan = root / "experiment_plan.json"
            experiment_plan.write_text(
                json.dumps(
                    {
                        "stage": "v1.2-a",
                        "story": {
                            "research_question": "How can the agent learn stable tower-taking wins?",
                            "main_hypothesis": "Push-window modeling improves tower pressure without extra deaths.",
                        },
                        "ablations": [
                            {
                                "name": "v1.2",
                                "reward_profile": "v1.2",
                                "hypothesis": "Full reward should stabilize the v1.1 peak.",
                                "report_dir": "logs/v1.2/report-v1.2",
                            }
                        ],
                        "success_metrics": [
                            {"metric": "avg_win_rate", "target": "> 0.84"},
                            {"metric": "avg_death", "target": "< 3.09"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            baseline_json = root / "baseline_v1.1.json"
            baseline_json.write_text(
                json.dumps(
                    {
                        "best_win_rate": 0.85,
                        "best_win_enemy_tower_hp": 1200,
                        "late_death": 2.5,
                        "best_win_hero_damage_balance": 0.67,
                        "source_log_dir": "logs/v1.1",
                    }
                ),
                encoding="utf-8",
            )

            artifacts = build_report(
                log_dir=log_dir,
                output_dir=output_dir,
                launch_manifest=launch_manifest,
                experiment_plan=experiment_plan,
                experiment_name="v1.2",
                baseline_json=baseline_json,
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
            self.assertIn("experiment_plan_stage: v1.2-a", manifest)
            self.assertIn("experiment_name: v1.2", manifest)
            self.assertIn("experiment_hypothesis: Full reward should stabilize the v1.1 peak.", manifest)
            self.assertIn("experiment_main_hypothesis: Push-window modeling improves tower pressure without extra deaths.", manifest)
            self.assertIn("experiment_success_metric_count: 2", manifest)
            self.assertIn("experiment_success_metrics: avg_win_rate,avg_death", manifest)
            self.assertIn("baseline_source: logs/v1.1", manifest)
            self.assertIn("baseline_best_win_rate: 0.85", manifest)
            self.assertIn("baseline_best_enemy_tower_hp: 1200.0", manifest)
            self.assertIn("baseline_late_death: 2.5", manifest)
            self.assertIn("baseline_best_hero_damage_balance: 0.67", manifest)
            self.assertIn("checkpoint_ranking_csv", manifest)
            self.assertIn("v1.2_candidate_gate_csv", manifest)
            self.assertIn("baseline_json", manifest)
            self.assertIn("evaluation_toml_metadata_csv", manifest)
            self.assertIn("evaluation_toml_metadata_jsonl", manifest)
            gate_md = artifacts["v1.2_candidate_gate_md"].read_text(encoding="utf-8")
            self.assertIn("> 0.85", gate_md)
            self.assertIn("< 1200.0", gate_md)

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
                    "opponent_agent": "common_ai",
                    "opponent_source": "common_ai",
                    "configured_opponent_agent": "curriculum",
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
                            "hero": {
                                "config_id": 199,
                                "kill_cnt": 2,
                                "dead_cnt": 1,
                                "money_cnt": 5000,
                                "total_hurt_to_hero": 12000,
                                "total_be_hurt_by_hero": 6000,
                            },
                            "enemy_hero": {"config_id": 133},
                            "tower": {"hp": 8000},
                            "enemy_tower": {"hp": 0},
                        },
                        {},
                    ],
                },
            }
            event_2 = json.loads(json.dumps(event))
            event_2["payload"]["frame_no"] = 20000
            event_2["payload"]["agents"][0]["hero"]["dead_cnt"] = 3
            event_2["payload"]["agents"][0]["hero"]["total_hurt_to_hero"] = 10000
            event_2["payload"]["agents"][0]["hero"]["total_be_hurt_by_hero"] = 20000
            event_2["payload"]["agents"][0]["tower"]["hp"] = 500
            other_checkpoint_event = json.loads(json.dumps(event))
            other_checkpoint_event["payload"]["monitor_hero_id"] = 112
            other_checkpoint_event["payload"]["opponent_hero_id"] = 112
            other_checkpoint_event["payload"]["checkpoint"] = {"actual_train_global_step": 17057}
            other_checkpoint_event["payload"]["agents"][0]["hero"]["config_id"] = 112
            other_checkpoint_event["payload"]["agents"][0]["enemy_hero"]["config_id"] = 112
            (record_dir / "episode_end-unit-1.jsonl").write_text(
                json.dumps(event) + "\n" + json.dumps(event_2) + "\n" + json.dumps(other_checkpoint_event) + "\n",
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
            self.assertTrue(artifacts["opponent_curriculum_summary_csv"].exists())
            self.assertTrue(artifacts["checkpoint_matrix_csv"].exists())
            self.assertTrue(artifacts["summoner_skill_results_csv"].exists())
            self.assertTrue(artifacts["summoner_skill_policy_patch_py"].exists())
            self.assertTrue(artifacts["summoner_skill_policy_patch_md"].exists())
            self.assertTrue(artifacts["summoner_skill_policy_gate_csv"].exists())
            self.assertTrue(artifacts["v1.2_candidate_gate_csv"].exists())
            self.assertIn(
                "push_window_tower_damage_share",
                artifacts["matchup_summary_csv"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "opponent_source",
                artifacts["opponent_curriculum_summary_csv"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "matchup_avg_unsafe_dive_death_corr",
                artifacts["checkpoint_ranking_csv"].read_text(encoding="utf-8"),
            )
            manifest = artifacts["manifest"].read_text(encoding="utf-8")
            self.assertIn("checkpoint_matrix_csv", manifest)
            self.assertIn("opponent_curriculum_summary_csv", manifest)
            self.assertIn("summoner_skill_results_csv", manifest)
            self.assertIn("summoner_skill_policy_gate_csv", manifest)
            self.assertIn("summoner_skill_policy_patch_py", manifest)
            self.assertIn("recommended_matchup_rows", manifest)
            self.assertIn("recommended_hurt_to_hero", manifest)
            self.assertIn("recommended_hurt_by_hero", manifest)
            self.assertIn("recommended_hero_damage_balance", manifest)
            self.assertIn("recommended_push_window_tower_damage_share", manifest)
            self.assertIn("recommended_death_p90", manifest)
            self.assertIn("recommended_self_tower_hp_p10", manifest)
            self.assertIn("recommended_timeout_rate", manifest)
            self.assertIn("recommended_unsafe_dive_severity", manifest)
            candidate_gate = artifacts["v1.2_candidate_gate_csv"].read_text(encoding="utf-8")
            self.assertIn("raw_matchup_rows,1", candidate_gate)
            self.assertIn("death_tail_risk", candidate_gate)

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
