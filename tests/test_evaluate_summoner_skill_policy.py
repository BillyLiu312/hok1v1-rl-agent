#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from utils.evaluate_summoner_skill_policy import (
    evaluate_recommendations,
    overall_status,
    read_rows,
    write_csv,
    write_markdown,
)


class EvaluateSummonerSkillPolicyTest(unittest.TestCase):
    def test_policy_gate_passes_strong_recommendation(self):
        rows = [
            {
                "matchup": "199_vs_133",
                "recommended_skill": 80110,
                "recommended_win_rate": 0.72,
                "recommended_avg_death": 2.4,
                "recommended_avg_enemy_tower_hp": 900,
                "recommended_episodes": 20,
                "current_policy_skill": 80107,
                "current_policy_win_rate": 0.6,
                "current_policy_avg_death": 2.2,
                "current_policy_avg_enemy_tower_hp": 1000,
                "needs_policy_update": True,
            }
        ]

        gates = evaluate_recommendations(rows)
        statuses = {row["gate"]: row["status"] for row in gates}

        self.assertEqual(overall_status(gates), "PASS")
        self.assertEqual(statuses["policy_update_candidates"], "PASS")
        self.assertEqual(statuses["current_policy_coverage"], "PASS")
        self.assertEqual(statuses["policy_win_delta"], "PASS")
        self.assertEqual(statuses["death_regression"], "PASS")

    def test_policy_gate_warns_on_weak_evidence(self):
        rows = [
            {
                "matchup": "199_vs_133",
                "recommended_skill": 80110,
                "recommended_win_rate": 0.62,
                "recommended_avg_death": 3.0,
                "recommended_avg_enemy_tower_hp": 1800,
                "recommended_episodes": 3,
                "current_policy_skill": 80107,
                "current_policy_win_rate": 0.6,
                "current_policy_avg_death": 2.2,
                "current_policy_avg_enemy_tower_hp": 1000,
                "needs_policy_update": True,
            }
        ]

        gates = evaluate_recommendations(rows)
        statuses = {row["gate"]: row["status"] for row in gates}

        self.assertEqual(overall_status(gates), "WARN")
        self.assertEqual(statuses["recommended_episodes"], "WARN")
        self.assertEqual(statuses["policy_win_delta"], "WARN")
        self.assertEqual(statuses["death_regression"], "WARN")
        self.assertEqual(statuses["tower_pressure_regression"], "WARN")

    def test_policy_gate_missing_without_rows(self):
        gates = evaluate_recommendations([])
        self.assertEqual(overall_status(gates), "FAIL")
        self.assertEqual(gates[0]["gate"], "recommendations")

    def test_read_write_outputs(self):
        rows = [
            {
                "status": "PASS",
                "gate": "policy_win_delta",
                "observed": 0.1,
                "threshold": ">= 0.05",
                "detail": "ok",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "gate.csv"
            md_path = root / "gate.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path)

            self.assertEqual(read_rows(csv_path)[0]["gate"], "policy_win_delta")
            self.assertIn("overall_status: PASS", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
