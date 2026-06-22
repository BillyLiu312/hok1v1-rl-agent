#!/usr/bin/env python3

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TrainWorkflowEvidenceFieldsTest(unittest.TestCase):
    def test_agent_training_recorder_imports_from_symlinked_agent_package(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "agent_ppo").symlink_to(root / "agent_ppo", target_is_directory=True)
            code = (
                "from agent_ppo.bootstrap import ensure_project_root_on_path;"
                "ensure_project_root_on_path();"
                "import agent_ppo.conf.training_recorder;"
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

    def test_agent_bootstrap_prefers_logical_project_root_for_symlinked_agent(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logical_root = temp_path / "project"
            real_root = temp_path / "real_agent_root"
            logical_root.mkdir()
            (logical_root / "utils").mkdir()
            (real_root / "agent_ppo").mkdir(parents=True)
            (real_root / "agent_ppo" / "__init__.py").write_text("", encoding="utf-8")
            (real_root / "agent_ppo" / "bootstrap.py").write_text(
                (root / "agent_ppo" / "bootstrap.py").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (logical_root / "utils" / "__init__.py").write_text("", encoding="utf-8")
            (logical_root / "utils" / "training_recorder.py").write_text("VALUE = 'logical-utils'\n", encoding="utf-8")
            (logical_root / "agent_ppo").symlink_to(real_root / "agent_ppo", target_is_directory=True)

            code = (
                "from agent_ppo.bootstrap import ensure_project_root_on_path;"
                "root = ensure_project_root_on_path();"
                "import utils.training_recorder as recorder;"
                "print(root);"
                "print(recorder.VALUE)"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=logical_root,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(logical_root), result.stdout)
        self.assertIn("logical-utils", result.stdout)

    def test_episode_record_promotes_evaluation_metadata(self):
        workflow_text = Path("agent_ppo/workflow/train_workflow.py").read_text(encoding="utf-8")

        self.assertIn('"evaluation": self._extract_evaluation_metadata(usr_conf)', workflow_text)
        self.assertIn("def _extract_evaluation_metadata", workflow_text)
        self.assertIn('usr_conf.get("evaluation")', workflow_text)

    def test_training_hot_path_does_not_import_project_level_utils(self):
        workflow_text = Path("agent_ppo/workflow/train_workflow.py").read_text(encoding="utf-8")
        algorithm_text = Path("agent_ppo/algorithm/algorithm.py").read_text(encoding="utf-8")

        self.assertNotIn("from utils.training_recorder import TrainingRecorder", workflow_text)
        self.assertNotIn("from utils.training_recorder import TrainingRecorder", algorithm_text)
        self.assertIn("from agent_ppo.conf.training_recorder import TrainingRecorder", workflow_text)
        self.assertIn("from agent_ppo.conf.training_recorder import TrainingRecorder", algorithm_text)


if __name__ == "__main__":
    unittest.main()
