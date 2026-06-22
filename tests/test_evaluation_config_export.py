#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from utils.evaluation_config_export import build_usr_conf, export_configs, render_toml
from utils.evaluation_matrix import build_rows, write_csv


class EvaluationConfigExportTest(unittest.TestCase):
    def test_build_usr_conf_from_matrix_row(self):
        row = build_rows(checkpoints=[15000], hero_ids=[112], repeats=1)[0]
        usr_conf = build_usr_conf(row)

        self.assertEqual(usr_conf["monitor"]["monitor_side"], 0)
        self.assertFalse(usr_conf["monitor"]["auto_switch_monitor_side"])
        self.assertEqual(usr_conf["episode"]["opponent_agent"], "common_ai")
        self.assertEqual(usr_conf["lineups"]["blue_camp"][0]["hero_id"], 112)
        self.assertEqual(usr_conf["lineups"]["blue_camp"][0]["select_skill"], 80115)
        self.assertEqual(usr_conf["evaluation"]["eval_id"], row["eval_id"])
        self.assertEqual(usr_conf["evaluation"]["repeat_index"], row["repeat_index"])

    def test_render_toml_contains_eval_config(self):
        row = build_rows(checkpoints=[15000], hero_ids=[112], repeats=1)[0]
        toml_text = render_toml(row)

        self.assertIn("monitor_side = 0", toml_text)
        self.assertIn('opponent_agent = "common_ai"', toml_text)
        self.assertIn("select_skill = 80115", toml_text)
        self.assertNotIn("[evaluation]", toml_text)
        self.assertNotIn("checkpoint_step = 15000", toml_text)

    def test_export_configs_writes_jsonl_toml_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            matrix_csv = root / "matrix.csv"
            output_dir = root / "configs"
            rows = build_rows(checkpoints=[15000], hero_ids=[112], repeats=2)
            write_csv(rows, matrix_csv)

            artifacts = export_configs(matrix_csv, output_dir, toml_limit=1)

            self.assertEqual(artifacts["rows"], 4)
            self.assertEqual(artifacts["toml_count"], 1)
            self.assertEqual(artifacts["toml_metadata_rows"], 1)
            jsonl_lines = artifacts["jsonl"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(jsonl_lines), 4)
            first_event = json.loads(jsonl_lines[0])
            self.assertEqual(first_event["evaluation"]["checkpoint_step"], 15000)
            self.assertEqual(first_event["evaluation"]["blue_select_skill"], 80115)
            self.assertEqual(first_event["evaluation"]["red_select_skill"], 80115)
            self.assertEqual(first_event["usr_conf"]["evaluation"], first_event["evaluation"])
            metadata_csv = artifacts["toml_metadata_csv"].read_text(encoding="utf-8")
            metadata_jsonl = artifacts["toml_metadata_jsonl"].read_text(encoding="utf-8").splitlines()
            self.assertIn("eval_id", metadata_csv)
            self.assertIn("usr_conf_json", metadata_csv)
            self.assertEqual(len(metadata_jsonl), 1)
            metadata_event = json.loads(metadata_jsonl[0])
            self.assertEqual(metadata_event["eval_id"], first_event["eval_id"])
            self.assertEqual(json.loads(metadata_event["usr_conf_json"])["evaluation"], first_event["evaluation"])
            manifest = artifacts["manifest"].read_text(encoding="utf-8")
            self.assertIn("toml_files: 1", manifest)
            self.assertIn("skill_pairs: 1", manifest)
            self.assertIn("toml_metadata_csv: toml_metadata.csv", manifest)
            self.assertIn("toml_metadata_jsonl: toml_metadata.jsonl", manifest)

    def test_export_configs_preserves_skill_grid_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            matrix_csv = root / "matrix.csv"
            output_dir = root / "configs"
            rows = build_rows(
                checkpoints=[15000],
                hero_ids=[199],
                repeats=1,
                include_skill_grid=True,
                candidate_skills=[80107, 80110],
            )
            write_csv(rows, matrix_csv)

            artifacts = export_configs(matrix_csv, output_dir, toml_limit=1)

            self.assertEqual(artifacts["toml_metadata_rows"], 1)
            jsonl_lines = artifacts["jsonl"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(jsonl_lines), 8)
            events = [json.loads(line) for line in jsonl_lines]
            skill_pairs = {
                (event["evaluation"]["blue_select_skill"], event["evaluation"]["red_select_skill"])
                for event in events
            }
            self.assertEqual(skill_pairs, {(80107, 80107), (80107, 80110), (80110, 80107), (80110, 80110)})
            manifest = artifacts["manifest"].read_text(encoding="utf-8")
            self.assertIn("skill_pairs: 4", manifest)


if __name__ == "__main__":
    unittest.main()
