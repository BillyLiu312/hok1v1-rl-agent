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
        "candidate_gate_matchup_filter": manifest.get("candidate_gate_matchup_filter", ""),
        "matchup_groups": ranking.get("matchup_groups", ""),
        "matchup_rows": ranking.get("matchup_rows", ""),
        "matchup_filter_eval_only": ranking.get("matchup_filter_eval_only", ""),
        "matchup_filter_opponent_agent": ranking.get("matchup_filter_opponent_agent", ""),
        "matchup_eval_ids": ranking.get("matchup_eval_ids", ""),
        "matchup_repeat_indices": ranking.get("matchup_repeat_indices", ""),
        "avg_win_rate": to_float(ranking.get("matchup_avg_win_rate") or ranking.get("common_ai_win_rate"), ""),
        "min_win_rate": to_float(ranking.get("matchup_min_win_rate"), ""),
        "avg_death": to_float(ranking.get("matchup_avg_death") or ranking.get("common_ai_death"), ""),
        "max_death_p90": to_float(ranking.get("matchup_max_death_p90"), ""),
        "min_self_tower_hp_p10": to_float(ranking.get("matchup_min_self_tower_hp_p10"), ""),
        "avg_timeout_rate": to_float(ranking.get("matchup_avg_timeout_rate"), ""),
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
    "max_death_p90",
    "min_self_tower_hp_p10",
    "avg_timeout_rate",
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
        row["ablation_interpretation"] = interpret_ablation(row, baseline_label)
        row["research_story_verdict"] = research_story_verdict(row, baseline_label)
    return rows


def metric_delta(value, baseline_value):
    if value in ("", None) or baseline_value in ("", None):
        return ""
    return value - baseline_value


def interpret_ablation(row: dict, baseline_label: str) -> str:
    if first_non_empty(row.get("experiment_name"), row.get("report")) == baseline_label:
        return "baseline"
    win_delta = row.get("avg_win_rate_delta_vs_baseline")
    death_delta = row.get("avg_death_delta_vs_baseline")
    tower_delta = row.get("avg_enemy_tower_hp_delta_vs_baseline")
    push_share_delta = row.get("avg_push_window_tower_damage_share_delta_vs_baseline")
    observed = [value for value in (win_delta, death_delta, tower_delta, push_share_delta) if value not in ("", None)]
    if not observed:
        return "inconclusive"

    worse = 0
    better = 0
    if win_delta not in ("", None):
        worse += win_delta < 0
        better += win_delta > 0
    if death_delta not in ("", None):
        worse += death_delta > 0
        better += death_delta < 0
    if tower_delta not in ("", None):
        worse += tower_delta > 0
        better += tower_delta < 0
    if push_share_delta not in ("", None):
        worse += push_share_delta < 0
        better += push_share_delta > 0

    if worse and not better:
        return "supports_baseline"
    if better and not worse:
        return "ablation_improves"
    return "mixed"


def is_better_delta(delta, higher_is_better=True):
    if delta in ("", None):
        return None
    return delta > 0 if higher_is_better else delta < 0


def is_worse_delta(delta, higher_is_better=True):
    if delta in ("", None):
        return None
    return delta < 0 if higher_is_better else delta > 0


def any_true(values):
    return any(value is True for value in values)


def all_observed(values):
    return any(value is not None for value in values)


def research_story_verdict(row: dict, baseline_label: str) -> str:
    name = first_non_empty(row.get("experiment_name"), row.get("report"))
    profile = row.get("reward_profile", "")
    if name == baseline_label:
        return "baseline_reference"

    win_better = is_better_delta(row.get("avg_win_rate_delta_vs_baseline"), higher_is_better=True)
    win_worse = is_worse_delta(row.get("avg_win_rate_delta_vs_baseline"), higher_is_better=True)
    min_win_worse = is_worse_delta(row.get("min_win_rate_delta_vs_baseline"), higher_is_better=True)
    death_better = is_better_delta(row.get("avg_death_delta_vs_baseline"), higher_is_better=False)
    death_worse = is_worse_delta(row.get("avg_death_delta_vs_baseline"), higher_is_better=False)
    death_tail_better = is_better_delta(row.get("max_death_p90_delta_vs_baseline"), higher_is_better=False)
    death_tail_worse = is_worse_delta(row.get("max_death_p90_delta_vs_baseline"), higher_is_better=False)
    self_tower_tail_better = is_better_delta(row.get("min_self_tower_hp_p10_delta_vs_baseline"), higher_is_better=True)
    self_tower_tail_worse = is_worse_delta(row.get("min_self_tower_hp_p10_delta_vs_baseline"), higher_is_better=True)
    timeout_better = is_better_delta(row.get("avg_timeout_rate_delta_vs_baseline"), higher_is_better=False)
    timeout_worse = is_worse_delta(row.get("avg_timeout_rate_delta_vs_baseline"), higher_is_better=False)
    tower_better = is_better_delta(row.get("avg_enemy_tower_hp_delta_vs_baseline"), higher_is_better=False)
    tower_worse = is_worse_delta(row.get("avg_enemy_tower_hp_delta_vs_baseline"), higher_is_better=False)
    push_share_better = is_better_delta(row.get("avg_push_window_tower_damage_share_delta_vs_baseline"), higher_is_better=True)
    push_share_worse = is_worse_delta(row.get("avg_push_window_tower_damage_share_delta_vs_baseline"), higher_is_better=True)
    unsafe_corr_better = is_better_delta(row.get("avg_unsafe_dive_death_corr_delta_vs_baseline"), higher_is_better=False)
    unsafe_corr_worse = is_worse_delta(row.get("avg_unsafe_dive_death_corr_delta_vs_baseline"), higher_is_better=False)

    observed = [
        win_better,
        win_worse,
        min_win_worse,
        death_better,
        death_worse,
        death_tail_better,
        death_tail_worse,
        self_tower_tail_better,
        self_tower_tail_worse,
        timeout_better,
        timeout_worse,
        tower_better,
        tower_worse,
        push_share_better,
        push_share_worse,
        unsafe_corr_better,
        unsafe_corr_worse,
    ]
    if not all_observed(observed):
        return "insufficient_evidence"

    if profile == "no_window_reward" or "window" in name:
        if any_true([win_worse, min_win_worse, death_worse, tower_worse, push_share_worse]):
            return "supports_push_window_modeling"
        if any_true([win_better, death_better, tower_better, push_share_better]) and not any_true([win_worse, death_worse, tower_worse]):
            return "challenges_push_window_modeling"
        return "mixed_push_window_evidence"

    if profile == "no_terminal_reward" or "terminal" in name:
        if any_true([win_worse, min_win_worse, tower_worse]):
            return "supports_terminal_alignment"
        if any_true([win_better, tower_better]) and not any_true([win_worse, tower_worse]):
            return "challenges_terminal_alignment"
        return "mixed_terminal_evidence"

    if profile == "death_only_risk" or "death" in name or "risk" in name:
        risk_better = any_true([death_better, death_tail_better, self_tower_tail_better, timeout_better])
        risk_worse = any_true([death_worse, death_tail_worse, self_tower_tail_worse, timeout_worse])
        if risk_better and any_true([win_worse, min_win_worse, tower_worse, push_share_worse]):
            return "death_risk_reduces_deaths_but_hurts_objective"
        if risk_better and any_true([win_better, tower_better]) and not any_true([win_worse, tower_worse]):
            return "death_risk_improves_stability"
        if risk_worse:
            return "challenges_death_only_risk"
        return "mixed_risk_evidence"

    if any_true([win_worse, death_worse, tower_worse, push_share_worse, unsafe_corr_worse]):
        return "supports_full_v1_2_recipe"
    if any_true([win_better, death_better, tower_better, push_share_better, unsafe_corr_better]):
        return "ablation_may_improve_recipe"
    return "mixed_research_evidence"


def collect_rows(report_dirs: list[Path]) -> list[dict]:
    return attach_baseline_deltas([summarize_report(report_dir) for report_dir in report_dirs])


def interpretation_counts(rows: list[dict]) -> dict:
    counts = {}
    for row in rows:
        key = row.get("ablation_interpretation") or "inconclusive"
        counts[key] = counts.get(key, 0) + 1
    return counts


def interpretation_summary_lines(rows: list[dict]) -> list[str]:
    counts = interpretation_counts(rows)
    lines = [
        "## Interpretation Summary",
        "",
        f"- baseline: {counts.get('baseline', 0)}",
        f"- supports_baseline: {counts.get('supports_baseline', 0)}",
        f"- ablation_improves: {counts.get('ablation_improves', 0)}",
        f"- mixed: {counts.get('mixed', 0)}",
        f"- inconclusive: {counts.get('inconclusive', 0)}",
        "",
    ]
    for row in rows:
        if row.get("ablation_interpretation") == "baseline":
            continue
        name = first_non_empty(row.get("experiment_name"), row.get("report"))
        lines.append(
            f"- {name}: {row.get('ablation_interpretation', '')} "
            f"[{row.get('research_story_verdict', '')}] "
            f"(win_delta={fmt(row.get('avg_win_rate_delta_vs_baseline'))}, "
            f"death_delta={fmt(row.get('avg_death_delta_vs_baseline'))}, "
            f"tower_hp_delta={fmt(row.get('avg_enemy_tower_hp_delta_vs_baseline'))})"
        )
    lines.append("")
    return lines


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
        "ablation_interpretation",
        "research_story_verdict",
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
        "candidate_gate_matchup_filter",
        "matchup_groups",
        "matchup_rows",
        "matchup_filter_eval_only",
        "matchup_filter_opponent_agent",
        "matchup_eval_ids",
        "matchup_repeat_indices",
        "avg_win_rate",
        "avg_win_rate_delta_vs_baseline",
        "min_win_rate",
        "min_win_rate_delta_vs_baseline",
        "avg_death",
        "avg_death_delta_vs_baseline",
        "max_death_p90",
        "max_death_p90_delta_vs_baseline",
        "min_self_tower_hp_p10",
        "min_self_tower_hp_p10_delta_vs_baseline",
        "avg_timeout_rate",
        "avg_timeout_rate_delta_vs_baseline",
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
        "ablation_interpretation",
        "research_story_verdict",
        "recommended_checkpoint",
        "gate_status",
        "evaluation_matchups",
        "candidate_gate_matchup_filter",
        "matchup_filter_eval_only",
        "matchup_filter_opponent_agent",
        "avg_win_rate",
        "avg_win_rate_delta_vs_baseline",
        "min_win_rate",
        "avg_death",
        "avg_death_delta_vs_baseline",
        "max_death_p90",
        "max_death_p90_delta_vs_baseline",
        "min_self_tower_hp_p10",
        "min_self_tower_hp_p10_delta_vs_baseline",
        "avg_timeout_rate",
        "avg_timeout_rate_delta_vs_baseline",
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
    lines.extend(interpretation_summary_lines(rows))
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
