#!/usr/bin/env python3
"""
Compare multiple v1.2 experiment report directories.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value, default=None):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value):
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def gate_status(rows: list[dict]) -> str:
    statuses = {row.get("status") for row in rows if row.get("status")}
    if "FAIL" in statuses or "MISSING" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    if "PASS" in statuses:
        return "PASS"
    return ""


def count_status(rows: list[dict], status: str) -> int:
    return sum(1 for row in rows if row.get("status") == status)


def first_row(rows: list[dict]) -> dict:
    return rows[0] if rows else {}


def report_label(report_dir: Path) -> str:
    return report_dir.name or report_dir.as_posix()


def summarize_report(report_dir: Path) -> dict:
    ranking = first_row(read_csv_rows(report_dir / "checkpoint_ranking.csv"))
    gate_rows = read_csv_rows(report_dir / "v1.2_candidate_gate.csv")
    metadata = first_row(read_csv_rows(report_dir / "run_metadata_summary.csv"))

    return {
        "report": report_label(report_dir),
        "path": report_dir.as_posix(),
        "reward_profile": metadata.get("reward_profile", ""),
        "reward_weight_overrides": metadata.get("reward_weight_overrides", ""),
        "opponent_schedule": metadata.get("opponent_schedule", ""),
        "recommended_checkpoint": ranking.get("checkpoint_step", ""),
        "score": to_float(ranking.get("score"), ""),
        "gate_status": gate_status(gate_rows),
        "gate_fail": count_status(gate_rows, "FAIL"),
        "gate_missing": count_status(gate_rows, "MISSING"),
        "matchup_groups": ranking.get("matchup_groups", ""),
        "avg_win_rate": to_float(ranking.get("matchup_avg_win_rate") or ranking.get("common_ai_win_rate"), ""),
        "min_win_rate": to_float(ranking.get("matchup_min_win_rate"), ""),
        "avg_death": to_float(ranking.get("matchup_avg_death") or ranking.get("common_ai_death"), ""),
        "avg_enemy_tower_hp": to_float(ranking.get("matchup_avg_enemy_tower_hp") or ranking.get("common_ai_enemy_tower_hp"), ""),
        "avg_push_window_active_frames": to_float(ranking.get("matchup_avg_push_window_active_frames"), ""),
        "avg_unsafe_dive_active_frames": to_float(ranking.get("matchup_avg_unsafe_dive_active_frames"), ""),
    }


def collect_rows(report_dirs: list[Path]) -> list[dict]:
    return [summarize_report(report_dir) for report_dir in report_dirs]


def write_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "report",
        "path",
        "reward_profile",
        "reward_weight_overrides",
        "opponent_schedule",
        "recommended_checkpoint",
        "score",
        "gate_status",
        "gate_fail",
        "gate_missing",
        "matchup_groups",
        "avg_win_rate",
        "min_win_rate",
        "avg_death",
        "avg_enemy_tower_hp",
        "avg_push_window_active_frames",
        "avg_unsafe_dive_active_frames",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output_path: Path, title="v1.2 Experiment Comparison"):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "report",
        "reward_profile",
        "recommended_checkpoint",
        "gate_status",
        "avg_win_rate",
        "min_win_rate",
        "avg_death",
        "avg_enemy_tower_hp",
        "avg_push_window_active_frames",
        "avg_unsafe_dive_active_frames",
    ]
    lines = [f"# {title}", "", f"- reports: {len(rows)}", ""]
    lines.extend(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple v1.2 experiment report directories.")
    parser.add_argument("report_dirs", nargs="+", type=Path, help="Report directories from utils/build_experiment_report.py")
    parser.add_argument("--csv", type=Path, default=Path("logs/v1.2/report_comparison.csv"), help="CSV output path")
    parser.add_argument("--md", type=Path, default=Path("logs/v1.2/report_comparison.md"), help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_rows(args.report_dirs)
    write_csv(rows, args.csv)
    write_markdown(rows, args.md)
    print(f"wrote comparison for {len(rows)} reports to {args.csv} and {args.md}")


if __name__ == "__main__":
    main()
