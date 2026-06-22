#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from utils.v1_2_experiment_plan import build_manifest, write_json, write_markdown


class V12ExperimentPlanTest(unittest.TestCase):
    def test_build_manifest_defines_story_and_ablations(self):
        manifest = build_manifest(stage="v1.2-a", checkpoints=[15000, 17057], heroes=[112, 133], repeats=3, skills=[80107, 80110])

        self.assertIn("research_question", manifest["story"])
        self.assertEqual(manifest["matrix"]["matchups"], 4)
        self.assertEqual(manifest["matrix"]["skill_pairs"], 4)
        self.assertEqual([group["reward_profile"] for group in manifest["ablations"]], ["v1.2", "no_window_reward", "no_terminal_reward", "death_only_risk"])
        self.assertEqual(manifest["ablations"][0]["env"]["HOK_TRAINING_RUN_ID"], "v1.2-a-v1.2")
        self.assertIn("--skill-grid", manifest["ablations"][0]["report_command"])
        self.assertIn("logs/v1.2/report-no-terminal", manifest["comparison_command"])

    def test_write_outputs_include_hypotheses_and_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = build_manifest(stage="unit", checkpoints=[1], heroes=[112], repeats=1, skills=[80107])
            json_path = root / "plan.json"
            md_path = root / "plan.md"

            write_json(manifest, json_path)
            write_markdown(manifest, md_path)

            json_text = json_path.read_text(encoding="utf-8")
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("main_hypothesis", json_text)
            self.assertIn("## Ablations", md_text)
            self.assertIn("no_window_reward", md_text)
            self.assertIn("python3 utils/compare_experiment_reports.py", md_text)


if __name__ == "__main__":
    unittest.main()
