#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from utils.v1_2_launch_manifest import build_manifest, write_json, write_markdown


class V12LaunchManifestTest(unittest.TestCase):
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
            self.assertIn("--launch-manifest logs/v1.2/launch_manifest.json", manifest["commands"]["report"])

            json_path = Path(temp_dir) / "launch.json"
            md_path = Path(temp_dir) / "launch.md"
            write_json(manifest, json_path)
            write_markdown(manifest, md_path)
            self.assertIn("sync_package_sha256", json_path.read_text(encoding="utf-8"))
            self.assertIn("unit-run", md_path.read_text(encoding="utf-8"))

    def test_v1_2_b_sets_curriculum_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sync_package = Path(temp_dir) / "sync_package.txt"
            sync_package.write_text("package", encoding="utf-8")
            manifest = build_manifest(sync_package, run_id="unit-b", stage="v1.2-b", reward_profile="v1.2")
            self.assertEqual(manifest["env"]["HOK_TRAINING_RECORD_DIR"], "logs/run_records/v1.2-b")
            self.assertEqual(manifest["env"]["HOK_OPPONENT_SCHEDULE"], "common_ai:4,historical:4,selfplay:2")


if __name__ == "__main__":
    unittest.main()
