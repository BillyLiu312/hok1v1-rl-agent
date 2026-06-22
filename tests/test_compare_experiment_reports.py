#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from utils.compare_experiment_reports import collect_rows, write_csv, write_markdown


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
                "checkpoint_step,score,matchup_groups,matchup_avg_win_rate,matchup_min_win_rate,matchup_avg_death,matchup_avg_enemy_tower_hp,matchup_avg_push_window_active_frames,matchup_avg_unsafe_dive_active_frames,matchup_avg_push_window_tower_damage_share,matchup_avg_unsafe_dive_death_corr",
                f"15000,100,9,{win_rate},0.7,2.1,900,12,1,0.75,0.1",
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
            self.assertEqual(rows[0]["gate_status"], "PASS")
            self.assertEqual(rows[0]["avg_push_window_tower_damage_share"], 0.75)
            self.assertEqual(rows[0]["avg_unsafe_dive_death_corr"], 0.1)
            self.assertEqual(rows[1]["reward_profile"], "no_window_reward")
            self.assertEqual(rows[1]["gate_status"], "WARN")

            csv_path = root / "comparison.csv"
            md_path = root / "comparison.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path)
            self.assertIn("reward_profile", csv_path.read_text(encoding="utf-8"))
            self.assertIn("no_window_reward", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
