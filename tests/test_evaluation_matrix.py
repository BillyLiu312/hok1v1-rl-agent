#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from utils.evaluation_matrix import build_rows, summarize_rows, write_csv, write_markdown


class EvaluationMatrixTest(unittest.TestCase):
    def test_build_rows_covers_checkpoints_matchups_and_sides(self):
        rows = build_rows(checkpoints=[15000], hero_ids=[112, 133], repeats=2)
        self.assertEqual(len(rows), 16)

        summary = summarize_rows(rows)
        self.assertEqual(summary["checkpoints"], 1)
        self.assertEqual(summary["matchups"], 4)
        self.assertEqual(summary["monitor_sides"], 2)
        self.assertEqual({row["monitor_side"] for row in rows}, {0, 1})
        self.assertIn("112_vs_133", {row["matchup"] for row in rows})
        self.assertIn("133_vs_112", {row["matchup"] for row in rows})

    def test_skill_grid_expands_candidate_skills(self):
        rows = build_rows(
            checkpoints=[15000],
            hero_ids=[199],
            repeats=1,
            include_skill_grid=True,
            candidate_skills=[80107, 80110],
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["blue_select_skill"] for row in rows}, {80107, 80110})
        self.assertEqual({row["red_select_skill"] for row in rows}, {80107, 80110})

    def test_write_outputs(self):
        rows = build_rows(checkpoints=[15000], hero_ids=[112], repeats=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "matrix.csv"
            md_path = Path(temp_dir) / "matrix.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path)

            self.assertIn("checkpoint_step", csv_path.read_text(encoding="utf-8"))
            self.assertIn("112_vs_112", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
