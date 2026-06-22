#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_ppo.conf.opponent_schedule import (
    apply_opponent_agent,
    load_model_pool,
    load_opponent_schedule,
    parse_schedule,
    select_curriculum_opponent,
)


class FixedRng:
    def choices(self, candidates, weights, k):
        return [candidates[0]]


class OpponentScheduleTest(unittest.TestCase):
    def test_select_curriculum_uses_historical_pool(self):
        opponent = select_curriculum_opponent(
            model_pool=["100", "200"],
            schedule={"historical": 1},
            rng=FixedRng(),
        )
        self.assertEqual(opponent, "100")

    def test_select_curriculum_falls_back_to_selfplay_without_candidates(self):
        opponent = select_curriculum_opponent(model_pool=[], schedule={"historical": 1}, rng=FixedRng())
        self.assertEqual(opponent, "selfplay")

    def test_apply_opponent_agent_updates_usr_conf(self):
        usr_conf = {"episode": {"opponent_agent": "selfplay"}}
        apply_opponent_agent(usr_conf, "common_ai")
        self.assertEqual(usr_conf["episode"]["opponent_agent"], "common_ai")

    def test_load_model_pool_reads_kaiwu_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "kaiwu.json"
            path.write_text('{"model_pool": [123, "456"]}', encoding="utf-8")
            self.assertEqual(load_model_pool(path), ["123", "456"])

    def test_parse_schedule_accepts_weight_string(self):
        self.assertEqual(
            parse_schedule("common_ai:4,historical:4,selfplay:2"),
            {"common_ai": 4.0, "historical": 4.0, "selfplay": 2.0},
        )

    def test_parse_schedule_ignores_invalid_items(self):
        self.assertEqual(parse_schedule("common_ai:4,bad,historical:x,selfplay:2"), {"common_ai": 4.0, "selfplay": 2.0})

    def test_load_opponent_schedule_reads_env(self):
        with patch.dict("os.environ", {"UNIT_OPPONENT_SCHEDULE": "common_ai:1,selfplay:3"}):
            self.assertEqual(load_opponent_schedule("UNIT_OPPONENT_SCHEDULE"), {"common_ai": 1.0, "selfplay": 3.0})


if __name__ == "__main__":
    unittest.main()
