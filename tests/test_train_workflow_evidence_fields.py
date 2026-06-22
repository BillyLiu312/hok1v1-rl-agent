#!/usr/bin/env python3

import unittest
from pathlib import Path


class TrainWorkflowEvidenceFieldsTest(unittest.TestCase):
    def test_episode_record_promotes_evaluation_metadata(self):
        workflow_text = Path("agent_ppo/workflow/train_workflow.py").read_text(encoding="utf-8")

        self.assertIn('"evaluation": self._extract_evaluation_metadata(usr_conf)', workflow_text)
        self.assertIn("def _extract_evaluation_metadata", workflow_text)
        self.assertIn('usr_conf.get("evaluation")', workflow_text)


if __name__ == "__main__":
    unittest.main()
