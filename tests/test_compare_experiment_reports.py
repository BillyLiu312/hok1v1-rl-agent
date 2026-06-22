#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from utils.compare_experiment_reports import collect_rows, read_manifest_summary, write_csv, write_markdown


def make_report(root: Path, name: str, profile: str, win_rate: float, gate_status: str):
    report_dir = root / name
    report_dir.mkdir()
    (report_dir / "run_metadata_summary.csv").write_text(
        "\n".join(
            [
                "run_id,reward_profile,reward_weight_overrides,opponent_schedule",
                f"{name},{profile},,common_ai:4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "checkpoint_ranking.csv").write_text(
        "\n".join(
            [
                "checkpoint_step,score,matchup_groups,matchup_rows,matchup_avg_win_rate,matchup_min_win_rate,matchup_avg_death,matchup_avg_enemy_tower_hp,reward_push_window_tower_damage,reward_unsafe_dive,reward_win_result,matchup_avg_push_window_active_frames,matchup_avg_unsafe_dive_active_frames,matchup_avg_push_window_tower_damage_share,matchup_avg_unsafe_dive_death_corr",
                f"15000,100,9,18,{win_rate},0.7,2.1,900,0.4,-0.5,1,12,1,0.75,0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "v1.2_candidate_gate.csv").write_text(
        "\n".join(
            [
                "status,gate,observed,threshold,detail",
                f"{gate_status},common_ai_win_rate,{win_rate},>0.84,test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# v1.2 Experiment Evidence Package",
                "",
                "- evaluation_rows: 360",
                "- evaluation_matchups: 9",
                "- evaluation_skill_pairs: 4",
                f"- candidate_gate_status: {gate_status}",
                "- candidate_gate_fail: 0",
                "- candidate_gate_missing: 0",
                "- launch_stage: v1.2-a",
                f"- launch_run_id: {name}-run",
                "- launch_preflight_status: PASS",
                "- launch_git_commit: abc123",
                f"- experiment_name: {name}",
                f"- experiment_hypothesis: hypothesis for {profile}",
                "- experiment_success_metric_count: 7",
                "- experiment_success_metrics: avg_win_rate,avg_death",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report_dir


class CompareExperimentReportsTest(unittest.TestCase):
    def test_collects_report_comparison_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_a = make_report(root, "v1.2", "v1.2", 0.88, "PASS")
            report_b = make_report(root, "no_window", "no_window_reward", 0.8, "WARN")

            rows = collect_rows([report_a, report_b])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["reward_profile"], "v1.2")
            self.assertEqual(rows[0]["experiment_name"], "v1.2")
            self.assertEqual(rows[0]["experiment_hypothesis"], "hypothesis for v1.2")
            self.assertEqual(rows[0]["success_metric_count"], "7")
            self.assertEqual(rows[0]["success_metrics"], "avg_win_rate,avg_death")
            self.assertEqual(rows[0]["baseline_experiment"], "v1.2")
            self.assertEqual(rows[0]["avg_win_rate_delta_vs_baseline"], 0.0)
            self.assertEqual(rows[0]["launch_run_id"], "v1.2-run")
            self.assertEqual(rows[0]["launch_preflight_status"], "PASS")
            self.assertEqual(rows[0]["evaluation_matchups"], "9")
            self.assertEqual(rows[0]["gate_status"], "PASS")
            self.assertEqual(rows[0]["matchup_rows"], "18")
            self.assertEqual(rows[0]["reward_push_window_tower_damage"], 0.4)
            self.assertEqual(rows[0]["reward_win_result"], 1.0)
            self.assertEqual(rows[0]["avg_push_window_tower_damage_share"], 0.75)
            self.assertEqual(rows[0]["avg_unsafe_dive_death_corr"], 0.1)
            self.assertEqual(rows[1]["reward_profile"], "no_window_reward")
            self.assertEqual(rows[1]["gate_status"], "WARN")
            self.assertEqual(rows[1]["baseline_experiment"], "v1.2")
            self.assertAlmostEqual(rows[1]["avg_win_rate_delta_vs_baseline"], -0.08)
            self.assertEqual(rows[1]["avg_death_delta_vs_baseline"], 0.0)
            self.assertEqual(rows[1]["avg_enemy_tower_hp_delta_vs_baseline"], 0.0)

            csv_path = root / "comparison.csv"
            md_path = root / "comparison.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path)
            self.assertIn("launch_run_id", csv_path.read_text(encoding="utf-8"))
            self.assertIn("experiment_hypothesis", csv_path.read_text(encoding="utf-8"))
            self.assertIn("success_metric_count", csv_path.read_text(encoding="utf-8"))
            self.assertIn("avg_win_rate_delta_vs_baseline", csv_path.read_text(encoding="utf-8"))
            self.assertIn("reward_profile", csv_path.read_text(encoding="utf-8"))
            self.assertIn("v1.2-run", md_path.read_text(encoding="utf-8"))
            self.assertIn("no_window_reward", md_path.read_text(encoding="utf-8"))
            self.assertIn("hypothesis for v1.2", md_path.read_text(encoding="utf-8"))
            self.assertIn("-0.08", md_path.read_text(encoding="utf-8"))

    def test_read_manifest_summary_parses_top_level_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "manifest.md"
            manifest.write_text(
                "\n".join(
                    [
                        "# Manifest",
                        "- launch_run_id: unit",
                        "- checkpoint_ranking_csv: `/tmp/ranking.csv`",
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(read_manifest_summary(manifest), {"launch_run_id": "unit"})

    def test_collects_launch_env_when_metadata_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_dir = root / "manifest_only"
            report_dir.mkdir()
            (report_dir / "checkpoint_ranking.csv").write_text(
                "\n".join(
                    [
                        "checkpoint_step,score,matchup_groups,matchup_rows,matchup_avg_win_rate",
                        "17057,101,9,18,0.9",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (report_dir / "manifest.md").write_text(
                "\n".join(
                    [
                        "# v1.2 Experiment Evidence Package",
                        "",
                        "- launch_reward_profile: v1.2",
                        "- experiment_reward_profile: no_window_reward",
                        "- experiment_name: no_window_reward",
                        "- experiment_hypothesis: No window reward ablation.",
                        "- experiment_success_metric_count: 7",
                        "- experiment_success_metrics: avg_win_rate,avg_death",
                        "- launch_reward_weight_overrides: death:5",
                        "- launch_opponent_schedule: common_ai:4,historical:4,selfplay:2",
                        "- candidate_gate_status: PASS",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            rows = collect_rows([report_dir])

            self.assertEqual(rows[0]["reward_profile"], "v1.2")
            self.assertEqual(rows[0]["reward_weight_overrides"], "death:5")
            self.assertEqual(rows[0]["opponent_schedule"], "common_ai:4,historical:4,selfplay:2")
            self.assertEqual(rows[0]["experiment_name"], "no_window_reward")
            self.assertEqual(rows[0]["experiment_hypothesis"], "No window reward ablation.")
            self.assertEqual(rows[0]["success_metric_count"], "7")
            self.assertEqual(rows[0]["baseline_experiment"], "no_window_reward")
            self.assertEqual(rows[0]["avg_win_rate_delta_vs_baseline"], 0.0)
            self.assertEqual(rows[0]["gate_status"], "PASS")


if __name__ == "__main__":
    unittest.main()
