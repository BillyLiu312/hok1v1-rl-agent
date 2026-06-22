#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from utils.select_checkpoint import attach_matchup_metrics, collect_candidates, rank_candidates, write_csv, write_markdown


class SelectCheckpointTest(unittest.TestCase):
    def test_ranks_checkpoint_from_training_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            (log_dir / "step-15000.md").write_text(
                """# Training Record

## Sampling

- target_step: 15000

## Environment Metrics: Common AI

- win_rate: 0.84
- enemy_tower_hp: 1400.59
- self_tower_hp: 7745.34
- death: 2.72
""",
                encoding="utf-8",
            )
            (log_dir / "step-17057.md").write_text(
                """# Training Record

## Sampling

- target_step: 17057

## Environment Metrics: Common AI

- win_rate: 0.75
- enemy_tower_hp: 1236.17
- self_tower_hp: 7204.92
- death: 3.09
""",
                encoding="utf-8",
            )

            rows = rank_candidates(collect_candidates(log_dir=log_dir))
            self.assertEqual(rows[0]["checkpoint_step"], 15000)

            md_path = log_dir / "ranking.md"
            write_markdown(rows, md_path, "Unit Ranking")
            self.assertIn("recommended_checkpoint: 15000", md_path.read_text(encoding="utf-8"))

    def test_matchup_metrics_can_override_step_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            training_csv = log_dir / "summary.csv"
            training_csv.write_text(
                "\n".join(
                    [
                        "source_file,step,common_ai_win_rate,common_ai_enemy_tower_hp,common_ai_self_tower_hp,common_ai_death,reward_push_window_tower_damage,reward_unsafe_dive,reward_win_result,reward_timeout_tower_gap",
                        "a,100,0.8,1000,7000,2,0.1,-1,0,0",
                        "b,200,0.7,900,7000,2,0.4,-0.5,1,0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            matchup_csv = log_dir / "matchups.csv"
            matchup_csv.write_text(
                "\n".join(
                    [
                        "checkpoint_step,eval_ids,evaluation_checkpoint_step,repeat_indices,matchup,is_eval,opponent_agent,episodes,win_rate,avg_frame,avg_self_tower_hp,avg_enemy_tower_hp,avg_kill,avg_death,avg_money_cnt,avg_reward_sum,avg_push_window_tower_damage,avg_unsafe_dive,avg_push_window_active_frames,avg_unsafe_dive_active_frames,push_window_tower_damage_share,unsafe_dive_death_corr",
                        "100,1,100,1,112_vs_112,True,common_ai,20,0.5,10000,6000,3000,1,4,5000,1,0.1,-2,3,20,0.2,0.8",
                        "200,2,200,1,112_vs_112,True,common_ai,20,0.9,9000,8000,1000,2,1,6000,5,0.4,-0.5,12,2,0.75,0.1",
                        "200,3,200,2,112_vs_112,True,17057,20,0.8,9500,7000,1200,2,2,5800,4,0.2,-0.7,10,4,0.5,0.2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            candidates = collect_candidates(training_csv=training_csv)
            attach_matchup_metrics(candidates, matchup_csv)
            rows = rank_candidates(candidates)
            self.assertEqual(rows[0]["checkpoint_step"], 200)
            self.assertEqual(rows[0]["matchup_groups"], 1)
            self.assertEqual(rows[0]["matchup_rows"], 1)
            self.assertTrue(rows[0]["matchup_filter_eval_only"])
            self.assertEqual(rows[0]["matchup_filter_opponent_agent"], "common_ai")
            self.assertEqual(rows[0]["matchup_eval_ids"], "2")
            self.assertEqual(rows[0]["matchup_repeat_indices"], "1")
            self.assertEqual(rows[0]["matchup_evaluation_checkpoint_step"], "200")
            self.assertEqual(rows[0]["reward_push_window_tower_damage"], 0.4)
            self.assertEqual(rows[0]["reward_win_result"], 1)
            self.assertEqual(rows[0]["matchup_min_win_rate"], 0.9)
            self.assertAlmostEqual(rows[0]["matchup_avg_push_window_tower_damage"], 0.4)
            self.assertEqual(rows[0]["matchup_avg_push_window_active_frames"], 12)
            self.assertEqual(rows[0]["matchup_avg_unsafe_dive_active_frames"], 2)
            self.assertEqual(rows[0]["matchup_avg_push_window_tower_damage_share"], 0.75)
            self.assertAlmostEqual(rows[0]["matchup_avg_unsafe_dive_death_corr"], 0.1)

            csv_path = log_dir / "ranking.csv"
            md_path = log_dir / "ranking.md"
            write_csv(rows, csv_path)
            write_markdown(rows, md_path, "Ranking")
            self.assertIn("reward_push_window_tower_damage", csv_path.read_text(encoding="utf-8"))
            self.assertIn("matchup_eval_ids", csv_path.read_text(encoding="utf-8"))
            self.assertIn("reward_win_result", md_path.read_text(encoding="utf-8"))
            self.assertIn("matchup_repeat_indices", md_path.read_text(encoding="utf-8"))

            attach_matchup_metrics(candidates, matchup_csv, eval_only=False, opponent_agent=None)
            unfiltered_rows = rank_candidates(candidates)
            self.assertEqual(unfiltered_rows[0]["matchup_rows"], 2)
            self.assertIsNone(unfiltered_rows[0]["matchup_filter_opponent_agent"])
            self.assertEqual(unfiltered_rows[0]["matchup_eval_ids"], "2,3")

    def test_matchup_actual_step_maps_to_target_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            training_csv = log_dir / "summary.csv"
            training_csv.write_text(
                "\n".join(
                    [
                        "source_file,step,actual_train_global_step,common_ai_win_rate",
                        "a,15000,15039,0.84",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            matchup_csv = log_dir / "matchups.csv"
            matchup_csv.write_text(
                "\n".join(
                    [
                        "checkpoint_step,matchup,is_eval,opponent_agent,episodes,win_rate,avg_frame,avg_self_tower_hp,avg_enemy_tower_hp,avg_kill,avg_death,avg_money_cnt,avg_reward_sum,avg_push_window_tower_damage,avg_unsafe_dive,avg_push_window_active_frames,avg_unsafe_dive_active_frames",
                        "15039,112_vs_112,True,common_ai,20,0.9,9000,8000,1000,2,1,6000,5,0.3,-0.2,8,1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            candidates = collect_candidates(training_csv=training_csv)
            attach_matchup_metrics(candidates, matchup_csv)
            rows = rank_candidates(candidates)
            self.assertEqual(rows[0]["checkpoint_step"], 15000)
            self.assertEqual(rows[0]["actual_train_global_step"], 15039)
            self.assertEqual(rows[0]["matchup_avg_win_rate"], 0.9)


if __name__ == "__main__":
    unittest.main()
