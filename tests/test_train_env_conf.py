#!/usr/bin/env python3

import unittest
from pathlib import Path


class TrainEnvConfTest(unittest.TestCase):
    def test_v1_2_a_defaults_to_common_ai(self):
        config_text = Path("agent_ppo/conf/train_env_conf.toml").read_text(encoding="utf-8")
        self.assertIn('opponent_agent = "common_ai"', config_text)
        self.assertIn('eval_opponent_type = "common_ai"', config_text)
        self.assertIn("v1.2-a", config_text)


if __name__ == "__main__":
    unittest.main()
