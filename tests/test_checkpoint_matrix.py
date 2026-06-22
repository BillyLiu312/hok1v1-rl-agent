#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from utils.checkpoint_matrix import build_checkpoint_matrix, collect_games, write_csv, write_markdown


def make_event(checkpoint, opponent, win, frame_no=10000):
    return {
        "stream": "episode_end",
        "payload": {
            "monitor_agent_index": 0,
            "monitor_hero_id": 199,
            "opponent_hero_id": 133,
            "is_eval": True,
            "opponent_agent": str(opponent),
            "checkpoint": {"actual_train_global_step": checkpoint},
            "frame_no": frame_no,
            "reward_sum": [10.0, -10.0],
            "agents": [
                {
                    "win": win,
                    "hero": {"config_id": 199, "kill_cnt": 2, "dead_cnt": 1, "money_cnt": 5000},
                    "enemy_hero": {"config_id": 133},
                    "tower": {"hp": 8000},
                    "enemy_tower": {"hp": 0},
                },
                {},
            ],
        },
    }


class CheckpointMatrixTest(unittest.TestCase):
    def test_build_checkpoint_matrix_and_elo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            record_dir = Path(temp_dir)
            events = [
                make_event(15000, 17057, 1),
                make_event(15000, 17057, 1),
                make_event(17057, 15000, 0),
            ]
            (record_dir / "episode_end-unit-1.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            games = collect_games(record_dir)
            self.assertEqual(len(games), 3)

            matrix_rows, elo_rows = build_checkpoint_matrix(record_dir)
            self.assertEqual(len(matrix_rows), 2)
            row_15000 = [row for row in matrix_rows if row["checkpoint"] == "15000"][0]
            self.assertEqual(row_15000["opponent"], "17057")
            self.assertEqual(row_15000["games"], 2)
            self.assertEqual(row_15000["win_rate"], 1.0)
            self.assertEqual(elo_rows[0]["player"], "15000")

            matrix_csv = record_dir / "matrix.csv"
            md_path = record_dir / "matrix.md"
            write_csv(
                matrix_rows,
                matrix_csv,
                ["checkpoint", "opponent", "games", "win_rate", "avg_frame", "avg_death", "avg_enemy_tower_hp"],
            )
            write_markdown(matrix_rows, elo_rows, md_path, "Unit Matrix")
            self.assertIn("15000", matrix_csv.read_text(encoding="utf-8"))
            self.assertIn("Elo Ranking", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
