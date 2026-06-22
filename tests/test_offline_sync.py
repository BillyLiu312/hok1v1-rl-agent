#!/usr/bin/env python3

import unittest
from pathlib import Path

from utils.offline_sync import (
    V1_2_REQUIRED_FILES,
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


if __name__ == "__main__":
    unittest.main()
