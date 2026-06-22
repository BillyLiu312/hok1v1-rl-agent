#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from utils.summoner_skill_grid import build_rows, write_csv, write_markdown


class SummonerSkillGridTest(unittest.TestCase):
    def test_build_rows_marks_current_policy(self):
        rows = build_rows(hero_ids=[199], candidate_skills=[80107, 80110])
        self.assertEqual(len(rows), 2)
        current_rows = [row for row in rows if row["is_current_policy"]]
        self.assertEqual(len(current_rows), 1)
        self.assertEqual(current_rows[0]["skill_id"], 80110)

    def test_write_outputs(self):
        rows = build_rows(hero_ids=[199], candidate_skills=[80110])
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "grid.csv"
            md_path = Path(temp_dir) / "grid.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path)
            self.assertIn("80110", csv_path.read_text(encoding="utf-8"))
            self.assertIn("199", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
