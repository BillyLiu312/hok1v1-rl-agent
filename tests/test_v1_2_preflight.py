#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from utils.v1_2_preflight import (
    REQUIRED_TOOLS,
    check_experiment_plan,
    check_launch_manifest_commands,
    check_train_env_conf,
    collect_rows,
    overall_status,
    parse_simple_toml_value,
    write_csv,
    write_markdown,
)


class V12PreflightTest(unittest.TestCase):
    def test_parse_simple_toml_value(self):
        text = '[episode]\nopponent_agent = "common_ai"\neval_opponent_type = "common_ai"\n'
        self.assertEqual(parse_simple_toml_value(text, "opponent_agent"), "common_ai")
        self.assertEqual(parse_simple_toml_value(text, "eval_opponent_type"), "common_ai")

    def test_train_env_conf_checks_fail_on_selfplay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train_env_conf.toml"
            path.write_text(
                '[episode]\nopponent_agent = "selfplay"\neval_opponent_type = "common_ai"\n',
                encoding="utf-8",
            )
            rows = check_train_env_conf(path)
            statuses = {row["check"]: row["status"] for row in rows}
            self.assertEqual(statuses["v1.2_a_opponent_agent"], "FAIL")
            self.assertEqual(statuses["eval_opponent_type"], "PASS")

    def test_collect_rows_passes_current_repo(self):
        rows = collect_rows()
        self.assertEqual(overall_status(rows), "PASS")
        self.assertGreater(len(rows), 10)

    def test_experiment_plan_checks_cover_research_protocol(self):
        rows = check_experiment_plan()
        statuses = {row["check"]: row["status"] for row in rows}
        self.assertEqual(statuses["experiment_plan_ablations"], "PASS")
        self.assertEqual(statuses["experiment_plan_reward_profiles"], "PASS")
        self.assertEqual(statuses["experiment_plan_matchups"], "PASS")
        self.assertEqual(statuses["experiment_plan_skill_pairs"], "PASS")
        self.assertEqual(statuses["experiment_plan_success_metrics"], "PASS")
        self.assertEqual(statuses["experiment_plan_report_bindings"], "PASS")

    def test_launch_manifest_commands_bind_experiment_plan(self):
        rows = check_launch_manifest_commands(Path.cwd())
        statuses = {row["check"]: row["status"] for row in rows}
        self.assertEqual(statuses["launch_manifest_experiment_plan_command"], "PASS")
        self.assertEqual(statuses["launch_manifest_report_binding"], "PASS")
        self.assertEqual(statuses["launch_manifest_report_record_dir"], "PASS")
        self.assertEqual(statuses["launch_manifest_report_output_dir"], "PASS")

    def test_required_tools_cover_v1_2_evidence_chain(self):
        expected = {
            "utils/build_experiment_report.py",
            "utils/checkpoint_matrix.py",
            "utils/compare_experiment_reports.py",
            "utils/run_metadata_summary.py",
            "utils/summoner_skill_grid.py",
            "utils/summoner_skill_results.py",
            "utils/v1_2_launch_manifest.py",
        }
        self.assertTrue(expected.issubset(set(REQUIRED_TOOLS)))

    def test_write_outputs(self):
        rows = collect_rows()
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "preflight.csv"
            md_path = Path(temp_dir) / "preflight.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path)
            self.assertIn("v1.2_a_opponent_agent", csv_path.read_text(encoding="utf-8"))
            self.assertIn("overall_status", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
