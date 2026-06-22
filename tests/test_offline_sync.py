#!/usr/bin/env python3

import unittest
import tempfile
from pathlib import Path

from utils.offline_sync import (
    V1_2_REQUIRED_FILES,
    iter_files,
    preset_include_patterns,
    repo_root,
    v1_2_readiness,
)


class OfflineSyncTest(unittest.TestCase):
    def test_v1_2_preset_contains_required_files(self):
        root = repo_root()
        readiness = v1_2_readiness(root, preset_include_patterns("v1.2"), [])
        self.assertTrue(readiness["ready"], readiness["missing"])
        self.assertGreater(readiness["included_count"], readiness["required_count"])

    def test_v1_2_required_files_exist(self):
        root = repo_root()
        missing = [path for path in V1_2_REQUIRED_FILES if not (root / Path(path)).exists()]
        self.assertEqual(missing, [])

    def test_iter_files_follows_included_symlink_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target_dir = root / "mounted_agent_ppo"
            (target_dir / "conf").mkdir(parents=True)
            (target_dir / "conf" / "conf.py").write_text("# remote-mounted agent code\n", encoding="utf-8")
            (root / "agent_ppo").symlink_to(target_dir, target_is_directory=True)

            files = [
                path.relative_to(root).as_posix()
                for path in iter_files(root, ["agent_ppo/conf/**"], [])
            ]

            self.assertEqual(files, ["agent_ppo/conf/conf.py"])


if __name__ == "__main__":
    unittest.main()
