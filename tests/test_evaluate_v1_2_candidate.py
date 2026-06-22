#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from utils.evaluate_v1_2_candidate import (
    evaluate_candidate,
    find_candidate,
    overall_status,
    read_csv_rows,
    write_csv,
    write_markdown,
)


class EvaluateV12CandidateTest(unittest.TestCase):
    def test_candidate_passes_documented_gates(self):
        candidate = {
            "checkpoint_step": 20000,
            "matchup_avg_win_rate": 0.88,
            "matchup_min_win_rate": 0.75,
            "matchup_avg_death": 2.0,
            "matchup_avg_enemy_tower_hp": 900,
            "matchup_groups": 9,
            "matchup_avg_push_window_tower_damage_share": 0.45,
            "matchup_avg_unsafe_dive_death_corr": 0.1,
        }
        matchup_rows = [
            {"matchup": f"{hero}_vs_{opponent}", "episodes": 20}
            for hero in (112, 133, 199)
            for opponent in (112, 133, 199)
        ]

        rows = evaluate_candidate(candidate, matchup_rows=matchup_rows)
        self.assertEqual(overall_status(rows), "PASS")
        self.assertTrue(all(row["status"] == "PASS" for row in rows))

    def test_candidate_fails_missing_matchup_evidence(self):
        candidate = {
            "checkpoint_step": 15000,
            "common_ai_win_rate": 0.8,
            "common_ai_death": 3.5,
            "common_ai_enemy_tower_hp": 2000,
        }

        rows = evaluate_candidate(candidate)
        statuses = {row["gate"]: row["status"] for row in rows}
        self.assertEqual(statuses["common_ai_win_rate"], "FAIL")
        self.assertEqual(statuses["matchup_coverage"], "MISSING")
        self.assertEqual(statuses["avg_death"], "FAIL")
        self.assertEqual(statuses["push_window_evidence"], "MISSING")
        self.assertEqual(statuses["unsafe_dive_risk"], "WARN")
        self.assertEqual(overall_status(rows), "FAIL")

    def test_candidate_warns_on_unsafe_dive_correlation(self):
        candidate = {
            "checkpoint_step": 20000,
            "matchup_avg_win_rate": 0.88,
            "matchup_min_win_rate": 0.75,
            "matchup_avg_death": 2.0,
            "matchup_avg_enemy_tower_hp": 900,
            "matchup_groups": 9,
            "matchup_avg_push_window_tower_damage_share": 0.08,
            "matchup_avg_unsafe_dive_death_corr": 0.6,
        }

        rows = evaluate_candidate(candidate)
        statuses = {row["gate"]: row["status"] for row in rows}
        self.assertEqual(statuses["push_window_evidence"], "WARN")
        self.assertEqual(statuses["unsafe_dive_risk"], "WARN")

    def test_candidate_warns_when_matchup_episode_count_is_too_low(self):
        candidate = {
            "checkpoint_step": 20000,
            "matchup_avg_win_rate": 0.88,
            "matchup_min_win_rate": 0.75,
            "matchup_avg_death": 2.0,
            "matchup_avg_enemy_tower_hp": 900,
            "matchup_groups": 9,
            "matchup_avg_push_window_tower_damage_share": 0.45,
            "matchup_avg_unsafe_dive_death_corr": 0.1,
        }
        matchup_rows = [
            {"matchup": f"{hero}_vs_{opponent}", "episodes": 20}
            for hero in (112, 133, 199)
            for opponent in (112, 133, 199)
        ]
        matchup_rows[0]["episodes"] = 3

        rows = evaluate_candidate(candidate, matchup_rows=matchup_rows)
        statuses = {row["gate"]: row["status"] for row in rows}

        self.assertEqual(statuses["raw_matchup_rows"], "PASS")
        self.assertEqual(statuses["matchup_min_episodes"], "WARN")
        self.assertEqual(overall_status(rows), "WARN")

    def test_read_write_and_find_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ranking_csv = root / "ranking.csv"
            ranking_csv.write_text(
                "\n".join(
                    [
                        "checkpoint_step,matchup_avg_win_rate",
                        "100,0.7",
                        "200,0.9",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = read_csv_rows(ranking_csv)
            self.assertEqual(find_candidate(rows, 200)["checkpoint_step"], "200")

            gate_rows = evaluate_candidate(find_candidate(rows, 200))
            csv_path = root / "gate.csv"
            md_path = root / "gate.md"
            write_csv(gate_rows, csv_path)
            write_markdown(gate_rows, md_path, "Gate", find_candidate(rows, 200))
            self.assertIn("checkpoint_selection", csv_path.read_text(encoding="utf-8"))
            self.assertIn("overall_status", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
