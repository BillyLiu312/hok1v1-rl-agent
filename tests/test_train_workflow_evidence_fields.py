#!/usr/bin/env python3

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TrainWorkflowEvidenceFieldsTest(unittest.TestCase):
    def test_agent_bootstrap_exposes_project_root_utils(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "agent_ppo").symlink_to(root / "agent_ppo", target_is_directory=True)
            code = (
                "from agent_ppo.bootstrap import ensure_project_root_on_path;"
                "ensure_project_root_on_path();"
                "import utils.training_recorder;"
                "print('ok')"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=temp_path,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_episode_record_promotes_evaluation_metadata(self):
        workflow_text = Path("agent_ppo/workflow/train_workflow.py").read_text(encoding="utf-8")

        self.assertIn('"evaluation": self._extract_evaluation_metadata(usr_conf)', workflow_text)
        self.assertIn("def _extract_evaluation_metadata", workflow_text)
        self.assertIn('usr_conf.get("evaluation")', workflow_text)


if __name__ == "__main__":
    unittest.main()
