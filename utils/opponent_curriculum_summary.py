#!/usr/bin/env python3
"""
Summarize actual opponent curriculum sampling from TrainingRecorder episode logs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from utils.analyze_run_records import avg, iter_events, summarize_episode


def collect_rows(record_dir: Path) -> list[dict]:
    groups = defaultdict(list)
    for event in iter_events(record_dir, "episode_end"):
        payload = event.get("payload", {})
        episode = summarize_episode(payload)
        key = (
            episode.get("checkpoint_step"),
            episode.get("is_eval"),
            episode.get("configured_opponent_agent"),
            episode.get("opponent_source"),
            episode.get("opponent_agent"),
        )
        groups[key].append(episode)

    rows = []
    for (checkpoint_step, is_eval, configured_opponent_agent, opponent_source, opponent_agent), items in sorted(
        groups.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        rows.append(
            {
                "checkpoint_step": checkpoint_step,
                "is_eval": is_eval,
                "configured_opponent_agent": configured_opponent_agent,
                "opponent_source": opponent_source,
                "opponent_agent": opponent_agent,
                "episodes": len(items),
                "win_rate": avg([item["win"] for item in items]),
                "avg_death": avg([item["death"] for item in items]),
                "avg_enemy_tower_hp": avg([item["enemy_tower_hp"] for item in items]),
                "avg_self_tower_hp": avg([item["self_tower_hp"] for item in items]),
                "avg_frame": avg([item["frame_no"] for item in items]),
            }
        )
    return rows


def fmt(value):
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint_step",
        "is_eval",
        "configured_opponent_agent",
        "opponent_source",
        "opponent_agent",
        "episodes",
        "win_rate",
        "avg_death",
        "avg_enemy_tower_hp",
        "avg_self_tower_hp",
        "avg_frame",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output_path: Path, title: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "checkpoint_step",
        "is_eval",
        "configured_opponent_agent",
        "opponent_source",
        "opponent_agent",
        "episodes",
        "win_rate",
        "avg_death",
        "avg_enemy_tower_hp",
        "avg_self_tower_hp",
        "avg_frame",
    ]
    lines = [f"# {title}", "", f"- groups: {len(rows)}", ""]
    lines.extend(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize curriculum opponent sampling from episode_end JSONL.")
    parser.add_argument("record_dir", type=Path, help="Directory containing episode_end-*.jsonl files")
    parser.add_argument("--csv", type=Path, default=None, help="CSV output path")
    parser.add_argument("--md", type=Path, default=None, help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_rows(args.record_dir)
    csv_path = args.csv or args.record_dir / "opponent_curriculum_summary.csv"
    md_path = args.md or args.record_dir / "opponent_curriculum_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, f"Opponent Curriculum Summary: {args.record_dir.as_posix()}")
    print(f"wrote {len(rows)} opponent curriculum groups to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
