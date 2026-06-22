#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from utils.v1_2_launch_manifest import ablation_for_profile, build_commands, build_manifest, write_json, write_markdown


class V12LaunchManifestTest(unittest.TestCase):
    def test_build_commands_bind_experiment_plan(self):
        commands = build_commands("v1.2-a", "v1.2")
        self.assertIn("utils/v1_2_baseline.py", commands["baseline"])
        self.assertIn("utils/v1_2_experiment_plan.py --stage v1.2-a", commands["experiment_plan"])
        self.assertIn("--record-dir logs/run_records/v1.2-a", commands["report"])
        self.assertIn("--output-dir logs/v1.2/report-v1.2", commands["report"])
        self.assertIn("--baseline-json logs/v1.2/baseline_v1.1.json", commands["report"])
        self.assertIn("--experiment-plan logs/v1.2/experiment_plan.json", commands["report"])
        self.assertIn("--experiment-name v1.2", commands["report"])

    def test_ablation_profile_commands_use_ablation_dirs(self):
        self.assertEqual(ablation_for_profile("no_window_reward")["name"], "no_window_reward")
        commands = build_commands("v1.2-a", "no_window_reward")
        self.assertIn("--record-dir logs/run_records/v1.2-no-window", commands["report"])
        self.assertIn("--output-dir logs/v1.2/report-no-window", commands["report"])
        self.assertIn("--experiment-name no_window_reward", commands["report"])

    def test_build_manifest_records_sync_package_and_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sync_package = Path(temp_dir) / "sync_package.txt"
            sync_package.write_text("package", encoding="utf-8")

            manifest = build_manifest(sync_package, run_id="unit-run", stage="v1.2-a", reward_profile="v1.2")
            self.assertEqual(manifest["run_id"], "unit-run")
            self.assertTrue(manifest["sync_package_exists"])
            self.assertEqual(manifest["sync_package_bytes"], len("package"))
            self.assertEqual(manifest["env"]["HOK_TRAINING_RUN_ID"], "unit-run")
            self.assertEqual(manifest["env"]["HOK_REWARD_PROFILE"], "v1.2")
            self.assertIn("utils/v1_2_experiment_plan.py --stage v1.2-a", manifest["commands"]["experiment_plan"])
            self.assertIn("utils/v1_2_baseline.py", manifest["commands"]["baseline"])
            self.assertIn("--launch-manifest logs/v1.2/launch_manifest.json", manifest["commands"]["report"])
            self.assertIn("--baseline-json logs/v1.2/baseline_v1.1.json", manifest["commands"]["report"])
            self.assertIn("--experiment-plan logs/v1.2/experiment_plan.json", manifest["commands"]["report"])
            self.assertIn("--experiment-name v1.2", manifest["commands"]["report"])

            json_path = Path(temp_dir) / "launch.json"
            md_path = Path(temp_dir) / "launch.md"
            write_json(manifest, json_path)
            write_markdown(manifest, md_path)
            self.assertIn("sync_package_sha256", json_path.read_text(encoding="utf-8"))
            self.assertIn("unit-run", md_path.read_text(encoding="utf-8"))
            self.assertIn("baseline", md_path.read_text(encoding="utf-8"))
            self.assertIn("experiment_plan", md_path.read_text(encoding="utf-8"))

    def test_v1_2_b_sets_curriculum_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sync_package = Path(temp_dir) / "sync_package.txt"
            sync_package.write_text("package", encoding="utf-8")
            manifest = build_manifest(sync_package, run_id="unit-b", stage="v1.2-b", reward_profile="v1.2")
            self.assertEqual(manifest["env"]["HOK_TRAINING_RECORD_DIR"], "logs/run_records/v1.2-b")
            self.assertEqual(manifest["env"]["HOK_OPPONENT_SCHEDULE"], "common_ai:4,historical:4,selfplay:2")
            self.assertIn("--stage v1.2-b", manifest["commands"]["experiment_plan"])
            self.assertIn("--record-dir logs/run_records/v1.2-b", manifest["commands"]["report"])
            self.assertIn("--output-dir logs/v1.2/report-v1.2-b", manifest["commands"]["report"])

    def test_ablation_manifest_uses_profile_record_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sync_package = Path(temp_dir) / "sync_package.txt"
            sync_package.write_text("package", encoding="utf-8")
            manifest = build_manifest(sync_package, run_id="unit-no-window", stage="v1.2-a", reward_profile="no_window_reward")
            self.assertEqual(manifest["env"]["HOK_REWARD_PROFILE"], "no_window_reward")
            self.assertEqual(manifest["env"]["HOK_TRAINING_RECORD_DIR"], "logs/run_records/v1.2-no-window")
            self.assertIn("--record-dir logs/run_records/v1.2-no-window", manifest["commands"]["report"])
            self.assertIn("--output-dir logs/v1.2/report-no-window", manifest["commands"]["report"])
            self.assertIn("--experiment-name no_window_reward", manifest["commands"]["report"])

    def test_build_manifest_records_explicit_schedule_and_weight_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sync_package = Path(temp_dir) / "sync_package.txt"
            sync_package.write_text("package", encoding="utf-8")
            manifest = build_manifest(
                sync_package,
                run_id="unit-overrides",
                stage="v1.2-a",
                reward_profile="v1.2",
                reward_weight_overrides="death:5,push_window_tower_damage:0",
                opponent_schedule="common_ai:1,selfplay:1",
            )

            self.assertEqual(manifest["env"]["HOK_REWARD_WEIGHT_OVERRIDES"], "death:5,push_window_tower_damage:0")
            self.assertEqual(manifest["env"]["HOK_OPPONENT_SCHEDULE"], "common_ai:1,selfplay:1")

            md_path = Path(temp_dir) / "launch.md"
            write_markdown(manifest, md_path)
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("HOK_REWARD_WEIGHT_OVERRIDES", md_text)
            self.assertIn("common_ai:1,selfplay:1", md_text)


if __name__ == "__main__":
    unittest.main()
