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


def read_manifest_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    summary = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("- ") or ": " not in line:
            continue
        key, value = line[2:].split(": ", 1)
        if key.endswith("_csv") or key.endswith("_md") or key in ("path",):
            continue
        summary[key] = value.strip("`")
    return summary


def report_label(report_dir: Path) -> str:
    return report_dir.name or report_dir.as_posix()


def first_non_empty(*values):
    for value in values:
        if value not in ("", None):
            return value
    return ""


def summarize_report(report_dir: Path) -> dict:
    ranking = first_row(read_csv_rows(report_dir / "checkpoint_ranking.csv"))
    gate_rows = read_csv_rows(report_dir / "v1.2_candidate_gate.csv")
    metadata = first_row(read_csv_rows(report_dir / "run_metadata_summary.csv"))
    manifest = read_manifest_summary(report_dir / "manifest.md")

    return {
        "report": report_label(report_dir),
        "path": report_dir.as_posix(),
        "launch_stage": manifest.get("launch_stage", ""),
        "launch_run_id": manifest.get("launch_run_id", ""),
        "launch_preflight_status": manifest.get("launch_preflight_status", ""),
        "launch_git_commit": manifest.get("launch_git_commit", ""),
        "experiment_plan_stage": manifest.get("experiment_plan_stage", ""),
        "experiment_name": manifest.get("experiment_name", ""),
        "experiment_hypothesis": manifest.get("experiment_hypothesis", ""),
        "success_metric_count": manifest.get("experiment_success_metric_count", ""),
        "success_metrics": manifest.get("experiment_success_metrics", ""),
        "reward_profile": first_non_empty(
            metadata.get("reward_profile"),
            manifest.get("launch_reward_profile"),
            manifest.get("experiment_reward_profile"),
        ),
        "reward_weight_overrides": first_non_empty(
            metadata.get("reward_weight_overrides"),
            manifest.get("launch_reward_weight_overrides"),
        ),
        "opponent_schedule": first_non_empty(metadata.get("opponent_schedule"), manifest.get("launch_opponent_schedule")),
        "evaluation_rows": manifest.get("evaluation_rows", ""),
        "evaluation_matchups": manifest.get("evaluation_matchups", ""),
        "evaluation_skill_pairs": manifest.get("evaluation_skill_pairs", ""),
        "recommended_checkpoint": ranking.get("checkpoint_step", ""),
        "score": to_float(ranking.get("score"), ""),
        "gate_status": manifest.get("candidate_gate_status") or gate_status(gate_rows),
        "gate_fail": manifest.get("candidate_gate_fail") or count_status(gate_rows, "FAIL"),
        "gate_missing": manifest.get("candidate_gate_missing") or count_status(gate_rows, "MISSING"),
        "matchup_groups": ranking.get("matchup_groups", ""),
        "matchup_rows": ranking.get("matchup_rows", ""),
        "avg_win_rate": to_float(ranking.get("matchup_avg_win_rate") or ranking.get("common_ai_win_rate"), ""),
        "min_win_rate": to_float(ranking.get("matchup_min_win_rate"), ""),
        "avg_death": to_float(ranking.get("matchup_avg_death") or ranking.get("common_ai_death"), ""),
        "avg_enemy_tower_hp": to_float(ranking.get("matchup_avg_enemy_tower_hp") or ranking.get("common_ai_enemy_tower_hp"), ""),
        "reward_push_window_tower_damage": to_float(ranking.get("reward_push_window_tower_damage"), ""),
        "reward_unsafe_dive": to_float(ranking.get("reward_unsafe_dive"), ""),
        "reward_win_result": to_float(ranking.get("reward_win_result"), ""),
        "avg_push_window_active_frames": to_float(ranking.get("matchup_avg_push_window_active_frames"), ""),
        "avg_unsafe_dive_active_frames": to_float(ranking.get("matchup_avg_unsafe_dive_active_frames"), ""),
        "avg_push_window_tower_damage_share": to_float(ranking.get("matchup_avg_push_window_tower_damage_share"), ""),
        "avg_unsafe_dive_death_corr": to_float(ranking.get("matchup_avg_unsafe_dive_death_corr"), ""),
    }


DELTA_METRICS = [
    "avg_win_rate",
    "min_win_rate",
    "avg_death",
    "avg_enemy_tower_hp",
    "avg_push_window_tower_damage_share",
    "avg_unsafe_dive_death_corr",
]


def find_baseline_row(rows: list[dict]) -> dict:
    for row in rows:
        if row.get("experiment_name") == "v1.2":
            return row
    for row in rows:
        if row.get("reward_profile") == "v1.2":
            return row
    return first_row(rows)


def attach_baseline_deltas(rows: list[dict]) -> list[dict]:
    baseline = find_baseline_row(rows)
    baseline_label = first_non_empty(baseline.get("experiment_name"), baseline.get("report"))
    for row in rows:
        row["baseline_experiment"] = baseline_label
        for metric in DELTA_METRICS:
            row[f"{metric}_delta_vs_baseline"] = metric_delta(row.get(metric), baseline.get(metric))
    return rows


def metric_delta(value, baseline_value):
    if value in ("", None) or baseline_value in ("", None):
        return ""
    return value - baseline_value


def collect_rows(report_dirs: list[Path]) -> list[dict]:
    return attach_baseline_deltas([summarize_report(report_dir) for report_dir in report_dirs])


def write_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "report",
        "path",
        "launch_stage",
        "launch_run_id",
        "launch_preflight_status",
        "launch_git_commit",
        "experiment_plan_stage",
        "experiment_name",
        "experiment_hypothesis",
        "success_metric_count",
        "success_metrics",
        "baseline_experiment",
        "reward_profile",
        "reward_weight_overrides",
        "opponent_schedule",
        "evaluation_rows",
        "evaluation_matchups",
        "evaluation_skill_pairs",
        "recommended_checkpoint",
        "score",
        "gate_status",
        "gate_fail",
        "gate_missing",
        "matchup_groups",
        "matchup_rows",
        "avg_win_rate",
        "avg_win_rate_delta_vs_baseline",
        "min_win_rate",
        "min_win_rate_delta_vs_baseline",
        "avg_death",
        "avg_death_delta_vs_baseline",
        "avg_enemy_tower_hp",
        "avg_enemy_tower_hp_delta_vs_baseline",
        "reward_push_window_tower_damage",
        "reward_unsafe_dive",
        "reward_win_result",
        "avg_push_window_active_frames",
        "avg_unsafe_dive_active_frames",
        "avg_push_window_tower_damage_share",
        "avg_push_window_tower_damage_share_delta_vs_baseline",
        "avg_unsafe_dive_death_corr",
        "avg_unsafe_dive_death_corr_delta_vs_baseline",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output_path: Path, title="v1.2 Experiment Comparison"):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "report",
        "experiment_name",
        "launch_run_id",
        "launch_preflight_status",
        "reward_profile",
        "experiment_hypothesis",
        "success_metric_count",
        "baseline_experiment",
        "recommended_checkpoint",
        "gate_status",
        "evaluation_matchups",
        "avg_win_rate",
        "avg_win_rate_delta_vs_baseline",
        "min_win_rate",
        "avg_death",
        "avg_death_delta_vs_baseline",
        "avg_enemy_tower_hp",
        "avg_enemy_tower_hp_delta_vs_baseline",
        "reward_push_window_tower_damage",
        "reward_unsafe_dive",
        "reward_win_result",
        "avg_push_window_active_frames",
        "avg_unsafe_dive_active_frames",
        "avg_push_window_tower_damage_share",
        "avg_push_window_tower_damage_share_delta_vs_baseline",
        "avg_unsafe_dive_death_corr",
        "avg_unsafe_dive_death_corr_delta_vs_baseline",
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
