#!/usr/bin/env python3

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.training_recorder import TrainingRecorder


class TrainingRecorderTest(unittest.TestCase):
    def test_record_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TrainingRecorder(record_dir=temp_dir, run_id="unit-test")
            recorder.record("episode_end", {"episode_cnt": 1, "value": 2})

            files = list(Path(temp_dir).glob("episode_end-unit-test-*.jsonl"))
            self.assertEqual(len(files), 1)

            line = files[0].read_text(encoding="utf-8").strip()
            event = json.loads(line)
            self.assertEqual(event["run_id"], "unit-test")
            self.assertEqual(event["stream"], "episode_end")
            self.assertEqual(event["payload"]["episode_cnt"], 1)
            self.assertEqual(event["payload"]["value"], 2)

    def test_record_can_be_disabled_by_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"HOK_TRAINING_RECORDER": "0"}):
                recorder = TrainingRecorder(record_dir=temp_dir, run_id="disabled")
                recorder.record("episode_end", {"episode_cnt": 1})

            self.assertEqual(list(Path(temp_dir).glob("*.jsonl")), [])


if __name__ == "__main__":
    unittest.main()
