#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from utils.run_metadata_summary import collect_rows, write_csv, write_markdown


class RunMetadataSummaryTest(unittest.TestCase):
    def test_collect_config_snapshot_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            record_dir = Path(temp_dir)
            event = {
                "time": "2026-06-22 12:00:00 +0800",
                "run_id": "v1.2-a-001",
                "pid": 123,
                "stream": "config",
                "payload": {
                    "name": "ppo_training_start",
                    "extra": {
                        "reward_profile": "v1.2",
                        "reward_weight_overrides": "",
                        "opponent_schedule": "common_ai:4,historical:4,selfplay:2",
                        "workflow": "agent_ppo/workflow/train_workflow.py",
                    },
                    "files": [
                        {
                            "path": "agent_ppo/conf/train_env_conf.toml",
                            "exists": True,
                            "sha256": "a" * 64,
                        },
                        {
                            "path": "agent_ppo/conf/conf.py",
                            "exists": True,
                            "sha256": "b" * 64,
                        },
                        {
                            "path": "kaiwu.json",
                            "exists": False,
                        },
                    ],
                },
            }
            (record_dir / "config-unit-1.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

            rows = collect_rows(record_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "v1.2-a-001")
            self.assertEqual(rows[0]["reward_profile"], "v1.2")
            self.assertEqual(rows[0]["train_env_conf_sha"], "a" * 12)
            self.assertEqual(rows[0]["conf_py_sha"], "b" * 12)
            self.assertEqual(rows[0]["missing_files"], "kaiwu.json")

            csv_path = record_dir / "metadata.csv"
            md_path = record_dir / "metadata.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path, "Metadata")
            self.assertIn("reward_profile", csv_path.read_text(encoding="utf-8"))
            self.assertIn("v1.2-a-001", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
