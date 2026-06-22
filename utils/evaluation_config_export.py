#!/usr/bin/env python3
"""
Export evaluation matrix rows as usr_conf JSONL and TOML config snippets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_matrix(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_int(value):
    return int(float(value))


def build_usr_conf(row: dict) -> dict:
    usr_conf = {
        "monitor": {
            "monitor_side": to_int(row["monitor_side"]),
            "auto_switch_monitor_side": False,
        },
        "episode": {
            "opponent_agent": str(row["opponent_agent"]),
            "eval_opponent_type": str(row["opponent_agent"]),
            "eval_interval": 1,
        },
        "lineups": {
            "blue_camp": [
                {
                    "hero_id": to_int(row["blue_hero_id"]),
                    "select_skill": to_int(row["blue_select_skill"]),
                }
            ],
            "red_camp": [
                {
                    "hero_id": to_int(row["red_hero_id"]),
                    "select_skill": to_int(row["red_select_skill"]),
                }
            ],
        },
        "evaluation": build_eval_metadata(row),
    }
    return usr_conf


def build_eval_metadata(row: dict) -> dict:
    return {
        "eval_id": to_int(row["eval_id"]),
        "checkpoint_step": to_int(row["checkpoint_step"]),
        "opponent_agent": str(row["opponent_agent"]),
        "matchup": row["matchup"],
        "repeat_index": to_int(row["repeat_index"]),
        "monitor_side": to_int(row["monitor_side"]),
        "monitor_hero_id": to_int(row["monitor_hero_id"]),
        "opponent_hero_id": to_int(row["opponent_hero_id"]),
        "blue_hero_id": to_int(row["blue_hero_id"]),
        "red_hero_id": to_int(row["red_hero_id"]),
        "blue_select_skill": to_int(row["blue_select_skill"]),
        "red_select_skill": to_int(row["red_select_skill"]),
    }


def toml_quote(value) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def render_toml(row: dict) -> str:
    monitor_side = to_int(row["monitor_side"])
    blue_hero_id = to_int(row["blue_hero_id"])
    red_hero_id = to_int(row["red_hero_id"])
    blue_skill = to_int(row["blue_select_skill"])
    red_skill = to_int(row["red_select_skill"])
    opponent_agent = str(row["opponent_agent"])
    return "\n".join(
        [
            "[monitor]",
            f"monitor_side = {monitor_side}",
            "auto_switch_monitor_side = false",
            "",
            "[episode]",
            f"opponent_agent = {toml_quote(opponent_agent)}",
            "eval_interval = 1",
            f"eval_opponent_type = {toml_quote(opponent_agent)}",
            "",
            "[[lineups.blue_camp]]",
            f"hero_id = {blue_hero_id}",
            f"select_skill = {blue_skill}",
            "",
            "[[lineups.red_camp]]",
            f"hero_id = {red_hero_id}",
            f"select_skill = {red_skill}",
            "",
        ]
    )


def write_jsonl(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            event = {
                "eval_id": to_int(row["eval_id"]),
                "checkpoint_step": to_int(row["checkpoint_step"]),
                "matchup": row["matchup"],
                "repeat_index": to_int(row["repeat_index"]),
                "evaluation": build_eval_metadata(row),
                "usr_conf": build_usr_conf(row),
            }
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_toml_files(rows: list[dict], output_dir: Path, limit: int | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = rows[:limit] if limit else rows
    paths = []
    for row in selected_rows:
        eval_id = to_int(row["eval_id"])
        checkpoint = to_int(row["checkpoint_step"])
        repeat_index = to_int(row["repeat_index"])
        matchup = row["matchup"]
        path = output_dir / f"eval-{eval_id:04d}-ckpt-{checkpoint}-{matchup}-r{repeat_index}.toml"
        path.write_text(render_toml(row), encoding="utf-8")
        paths.append(path)
    return paths


def write_toml_metadata(rows: list[dict], toml_paths: list[Path], output_csv: Path, output_jsonl: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_rows = rows[: len(toml_paths)]
    metadata_rows = []
    for row, toml_path in zip(selected_rows, toml_paths):
        metadata = build_eval_metadata(row)
        metadata_rows.append(
            {
                "toml_path": toml_path.as_posix(),
                **metadata,
                "usr_conf_json": json.dumps(build_usr_conf(row), ensure_ascii=False, sort_keys=True),
            }
        )

    fieldnames = [
        "toml_path",
        "eval_id",
        "checkpoint_step",
        "opponent_agent",
        "matchup",
        "repeat_index",
        "monitor_side",
        "monitor_hero_id",
        "opponent_hero_id",
        "blue_hero_id",
        "red_hero_id",
        "blue_select_skill",
        "red_select_skill",
        "usr_conf_json",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for item in metadata_rows:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return metadata_rows


def write_manifest(rows: list[dict], output_path: Path, toml_paths: list[Path]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted({to_int(row["checkpoint_step"]) for row in rows})
    matchups = sorted({row["matchup"] for row in rows})
    skill_pairs = sorted({(to_int(row["blue_select_skill"]), to_int(row["red_select_skill"])) for row in rows})
    opponent_agents = sorted({str(row["opponent_agent"]) for row in rows})
    lines = [
        "# Evaluation Config Export",
        "",
        f"- rows: {len(rows)}",
        f"- checkpoints: {', '.join(str(value) for value in checkpoints)}",
        f"- matchups: {len(matchups)}",
        f"- opponent_agents: {', '.join(opponent_agents)}",
        f"- skill_pairs: {len(skill_pairs)}",
        f"- toml_files: {len(toml_paths)}",
        "- toml_metadata_csv: toml_metadata.csv",
        "- toml_metadata_jsonl: toml_metadata.jsonl",
        "- note: TOML files contain only platform train_env_conf fields; evaluation metadata is kept in eval_usr_conf.jsonl, toml_metadata.* and filenames.",
        "",
        "## TOML Preview",
        "",
    ]
    for path in toml_paths[:20]:
        lines.append(f"- `{path.as_posix()}`")
    if len(toml_paths) > 20:
        lines.append(f"- ... {len(toml_paths) - 20} more")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_configs(matrix_csv: Path, output_dir: Path, toml_limit: int | None = None):
    rows = read_matrix(matrix_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "eval_usr_conf.jsonl"
    toml_dir = output_dir / "toml"
    manifest_path = output_dir / "manifest.md"
    toml_metadata_csv = output_dir / "toml_metadata.csv"
    toml_metadata_jsonl = output_dir / "toml_metadata.jsonl"
    write_jsonl(rows, jsonl_path)
    toml_paths = write_toml_files(rows, toml_dir, limit=toml_limit)
    toml_metadata_rows = write_toml_metadata(rows, toml_paths, toml_metadata_csv, toml_metadata_jsonl)
    write_manifest(rows, manifest_path, toml_paths)
    return {
        "jsonl": jsonl_path,
        "toml_dir": toml_dir,
        "toml_metadata_csv": toml_metadata_csv,
        "toml_metadata_jsonl": toml_metadata_jsonl,
        "toml_metadata_rows": len(toml_metadata_rows),
        "manifest": manifest_path,
        "toml_count": len(toml_paths),
        "rows": len(rows),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Export evaluation matrix rows into usr_conf JSONL/TOML artifacts.")
    parser.add_argument("matrix_csv", type=Path, help="CSV produced by utils/evaluation_matrix.py")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/evaluation_configs"), help="Output directory")
    parser.add_argument("--toml-limit", type=int, default=None, help="Only write the first N TOML files")
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts = export_configs(args.matrix_csv, args.output_dir, toml_limit=args.toml_limit)
    print(
        f"exported {artifacts['rows']} usr_conf rows to {artifacts['jsonl']} "
        f"and {artifacts['toml_count']} TOML files under {artifacts['toml_dir']}"
    )


if __name__ == "__main__":
    main()
