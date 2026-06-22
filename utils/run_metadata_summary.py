#!/usr/bin/env python3
"""
Summarize TrainingRecorder config snapshots for reproducible v1.2 reports.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def iter_config_events(record_dir: Path):
    for path in sorted(record_dir.glob("config-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def short_sha(value):
    return value[:12] if value else ""


def summarize_event(event: dict) -> dict:
    payload = event.get("payload", {})
    extra = payload.get("extra", {})
    files = payload.get("files", [])
    file_hashes = {
        item.get("path"): short_sha(item.get("sha256"))
        for item in files
        if item.get("path") and item.get("sha256")
    }
    missing_files = [item.get("path") for item in files if item.get("exists") is False]

    return {
        "time": event.get("time"),
        "run_id": event.get("run_id"),
        "pid": event.get("pid"),
        "name": payload.get("name"),
        "reward_profile": extra.get("reward_profile", ""),
        "reward_weight_overrides": extra.get("reward_weight_overrides", ""),
        "opponent_schedule": extra.get("opponent_schedule", ""),
        "model_pool_count": extra.get("model_pool_count", ""),
        "model_pool": ",".join(str(item) for item in extra.get("model_pool", [])),
        "workflow": extra.get("workflow", ""),
        "train_env_conf_sha": file_hashes.get("agent_ppo/conf/train_env_conf.toml", ""),
        "conf_py_sha": file_hashes.get("agent_ppo/conf/conf.py", ""),
        "kaiwu_json_sha": file_hashes.get("kaiwu.json", ""),
        "file_count": len(files),
        "missing_files": ",".join(path for path in missing_files if path),
    }


def collect_rows(record_dir: Path) -> list[dict]:
    return [summarize_event(event) for event in iter_config_events(record_dir)]


def write_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time",
        "run_id",
        "pid",
        "name",
        "reward_profile",
        "reward_weight_overrides",
        "opponent_schedule",
        "model_pool_count",
        "model_pool",
        "workflow",
        "train_env_conf_sha",
        "conf_py_sha",
        "kaiwu_json_sha",
        "file_count",
        "missing_files",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "" if value is None else str(value)


def write_markdown(rows: list[dict], output_path: Path, title: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id",
        "name",
        "reward_profile",
        "reward_weight_overrides",
        "opponent_schedule",
        "model_pool_count",
        "train_env_conf_sha",
        "conf_py_sha",
        "missing_files",
    ]
    lines = [f"# {title}", "", f"- config_snapshots: {len(rows)}", ""]
    lines.extend(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize TrainingRecorder config snapshots.")
    parser.add_argument("record_dir", type=Path, help="Directory containing config-*.jsonl files")
    parser.add_argument("--csv", type=Path, default=None, help="CSV output path")
    parser.add_argument("--md", type=Path, default=None, help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_rows(args.record_dir)
    csv_path = args.csv or args.record_dir / "run_metadata_summary.csv"
    md_path = args.md or args.record_dir / "run_metadata_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, f"Run Metadata Summary: {args.record_dir.as_posix()}")
    print(f"wrote {len(rows)} run metadata rows to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
