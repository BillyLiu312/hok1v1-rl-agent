#!/usr/bin/env python3
"""
Rank candidate checkpoints from training summaries and matchup evaluations.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.analyze_training_logs import collect_rows as collect_training_rows


def to_float(value, default=None):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=None):
    number = to_float(value, default=None)
    if number is None:
        return default
    return int(number)


def fmt(value):
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def is_truthy(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def collect_unique_tokens(rows: list[dict], key: str) -> str:
    values = []
    seen = set()
    for row in rows:
        raw_value = row.get(key)
        if raw_value in ("", None):
            continue
        for token in str(raw_value).split(","):
            token = token.strip()
            if token and token not in seen:
                seen.add(token)
                values.append(token)
    return ",".join(values)


def first_non_empty(rows: list[dict], key: str):
    for row in rows:
        value = row.get(key)
        if value not in ("", None):
            return value
    return ""


def normalize_training_row(row: dict) -> dict:
    step = to_int(row.get("step"))
    actual_step = to_int(row.get("actual_train_global_step"))
    return {
        "checkpoint_step": step,
        "actual_train_global_step": actual_step,
        "source": row.get("source_file", ""),
        "common_ai_win_rate": to_float(row.get("common_ai_win_rate")),
        "common_ai_enemy_tower_hp": to_float(row.get("common_ai_enemy_tower_hp")),
        "common_ai_self_tower_hp": to_float(row.get("common_ai_self_tower_hp")),
        "common_ai_death": to_float(row.get("common_ai_death")),
        "common_ai_frame": to_float(row.get("common_ai_frame")),
        "selfplay_win_rate": to_float(row.get("selfplay_win_rate")),
        "reward": to_float(row.get("reward")),
        "reward_push_window_tower_damage": to_float(row.get("reward_push_window_tower_damage")),
        "reward_unsafe_dive": to_float(row.get("reward_unsafe_dive")),
        "reward_win_result": to_float(row.get("reward_win_result")),
        "reward_timeout_tower_gap": to_float(row.get("reward_timeout_tower_gap")),
        "total_loss": to_float(row.get("total_loss")),
    }


def collect_candidates(log_dir: Path | None = None, training_csv: Path | None = None) -> dict[int, dict]:
    if training_csv:
        raw_rows = read_csv_rows(training_csv)
    elif log_dir:
        raw_rows = collect_training_rows(log_dir)
    else:
        raw_rows = []

    candidates = {}
    for raw_row in raw_rows:
        row = normalize_training_row(raw_row)
        step = row["checkpoint_step"]
        if step is not None:
            candidates[step] = row
    return candidates


def attach_matchup_metrics(
    candidates: dict[int, dict],
    matchup_csv: Path | None,
    eval_only=True,
    opponent_agent="common_ai",
):
    if not matchup_csv:
        return

    step_aliases = {}
    for step, candidate in candidates.items():
        step_aliases[step] = step
        actual_step = candidate.get("actual_train_global_step")
        if actual_step is not None:
            step_aliases[int(actual_step)] = step

    grouped = defaultdict(list)
    for row in read_csv_rows(matchup_csv):
        if eval_only and not is_truthy(row.get("is_eval")):
            continue
        if opponent_agent and str(row.get("opponent_agent")) != str(opponent_agent):
            continue
        raw_step = to_int(row.get("checkpoint_step"))
        if raw_step is None:
            continue
        step = step_aliases.get(raw_step, raw_step)
        grouped[step].append(row)

    for step, rows in grouped.items():
        candidate = candidates.setdefault(step, {"checkpoint_step": step, "source": ""})
        unique_matchups = {row.get("matchup") for row in rows if row.get("matchup")}
        win_rates = [to_float(row.get("win_rate")) for row in rows]
        win_rates = [value for value in win_rates if value is not None]
        deaths = [to_float(row.get("avg_death")) for row in rows]
        deaths = [value for value in deaths if value is not None]
        death_p90 = [to_float(row.get("death_p90")) for row in rows]
        death_p90 = [value for value in death_p90 if value is not None]
        enemy_tower_hp = [to_float(row.get("avg_enemy_tower_hp")) for row in rows]
        enemy_tower_hp = [value for value in enemy_tower_hp if value is not None]
        self_tower_hp = [to_float(row.get("avg_self_tower_hp")) for row in rows]
        self_tower_hp = [value for value in self_tower_hp if value is not None]
        self_tower_hp_p10 = [to_float(row.get("self_tower_hp_p10")) for row in rows]
        self_tower_hp_p10 = [value for value in self_tower_hp_p10 if value is not None]
        frame_p90 = [to_float(row.get("frame_p90")) for row in rows]
        frame_p90 = [value for value in frame_p90 if value is not None]
        timeout_rate = [to_float(row.get("timeout_rate")) for row in rows]
        timeout_rate = [value for value in timeout_rate if value is not None]
        push_window_tower_damage = [to_float(row.get("avg_push_window_tower_damage")) for row in rows]
        push_window_tower_damage = [value for value in push_window_tower_damage if value is not None]
        unsafe_dive = [to_float(row.get("avg_unsafe_dive")) for row in rows]
        unsafe_dive = [value for value in unsafe_dive if value is not None]
        unsafe_dive_severity = [to_float(row.get("avg_unsafe_dive_severity")) for row in rows]
        unsafe_dive_severity = [value for value in unsafe_dive_severity if value is not None]
        push_window_active_frames = [to_float(row.get("avg_push_window_active_frames")) for row in rows]
        push_window_active_frames = [value for value in push_window_active_frames if value is not None]
        unsafe_dive_active_frames = [to_float(row.get("avg_unsafe_dive_active_frames")) for row in rows]
        unsafe_dive_active_frames = [value for value in unsafe_dive_active_frames if value is not None]
        push_window_tower_damage_share = [to_float(row.get("push_window_tower_damage_share")) for row in rows]
        push_window_tower_damage_share = [value for value in push_window_tower_damage_share if value is not None]
        unsafe_dive_death_corr = [to_float(row.get("unsafe_dive_death_corr")) for row in rows]
        unsafe_dive_death_corr = [value for value in unsafe_dive_death_corr if value is not None]

        candidate.update(
            {
                "matchup_groups": len(unique_matchups),
                "matchup_rows": len(rows),
                "matchup_filter_eval_only": eval_only,
                "matchup_filter_opponent_agent": opponent_agent,
                "matchup_eval_ids": collect_unique_tokens(rows, "eval_ids"),
                "matchup_repeat_indices": collect_unique_tokens(rows, "repeat_indices"),
                "matchup_evaluation_checkpoint_step": first_non_empty(rows, "evaluation_checkpoint_step"),
                "matchup_avg_win_rate": avg(win_rates),
                "matchup_min_win_rate": min(win_rates) if win_rates else None,
                "matchup_avg_death": avg(deaths),
                "matchup_avg_death_p90": avg(death_p90),
                "matchup_max_death_p90": max(death_p90) if death_p90 else None,
                "matchup_avg_enemy_tower_hp": avg(enemy_tower_hp),
                "matchup_avg_self_tower_hp": avg(self_tower_hp),
                "matchup_min_self_tower_hp_p10": min(self_tower_hp_p10) if self_tower_hp_p10 else None,
                "matchup_avg_frame_p90": avg(frame_p90),
                "matchup_avg_timeout_rate": avg(timeout_rate),
                "matchup_max_timeout_rate": max(timeout_rate) if timeout_rate else None,
                "matchup_avg_push_window_tower_damage": avg(push_window_tower_damage),
                "matchup_avg_unsafe_dive": avg(unsafe_dive),
                "matchup_avg_unsafe_dive_severity": avg(unsafe_dive_severity),
                "matchup_avg_push_window_active_frames": avg(push_window_active_frames),
                "matchup_avg_unsafe_dive_active_frames": avg(unsafe_dive_active_frames),
                "matchup_avg_push_window_tower_damage_share": avg(push_window_tower_damage_share),
                "matchup_avg_unsafe_dive_death_corr": avg(unsafe_dive_death_corr),
            }
        )


def avg(values):
    return sum(values) / len(values) if values else None


def metric(candidate: dict, primary_key: str, fallback_key: str | None = None):
    value = candidate.get(primary_key)
    if value is None and fallback_key:
        value = candidate.get(fallback_key)
    return value


def compute_score(candidate: dict) -> float:
    win_rate = metric(candidate, "matchup_avg_win_rate", "common_ai_win_rate") or 0.0
    min_win_rate = candidate.get("matchup_min_win_rate")
    death = metric(candidate, "matchup_avg_death", "common_ai_death")
    enemy_tower_hp = metric(candidate, "matchup_avg_enemy_tower_hp", "common_ai_enemy_tower_hp")
    self_tower_hp = metric(candidate, "matchup_avg_self_tower_hp", "common_ai_self_tower_hp")

    score = win_rate * 100.0
    if min_win_rate is not None:
        score += min_win_rate * 25.0
    if death is not None:
        score -= death * 2.0
    if enemy_tower_hp is not None:
        score -= enemy_tower_hp / 1000.0
    if self_tower_hp is not None:
        score += self_tower_hp / 5000.0
    unsafe_dive_active_frames = candidate.get("matchup_avg_unsafe_dive_active_frames")
    if unsafe_dive_active_frames is not None:
        score -= unsafe_dive_active_frames * 0.05
    unsafe_dive_severity = candidate.get("matchup_avg_unsafe_dive_severity")
    if unsafe_dive_severity is not None:
        score -= unsafe_dive_severity * 0.5
    death_p90 = candidate.get("matchup_max_death_p90")
    if death_p90 is not None:
        score -= death_p90 * 1.5
    self_tower_hp_p10 = candidate.get("matchup_min_self_tower_hp_p10")
    if self_tower_hp_p10 is not None:
        score += self_tower_hp_p10 / 6000.0
    timeout_rate = candidate.get("matchup_avg_timeout_rate")
    if timeout_rate is not None:
        score -= timeout_rate * 5.0
    push_window_tower_damage_share = candidate.get("matchup_avg_push_window_tower_damage_share")
    if push_window_tower_damage_share is not None:
        score += push_window_tower_damage_share * 2.0
    unsafe_dive_death_corr = candidate.get("matchup_avg_unsafe_dive_death_corr")
    if unsafe_dive_death_corr is not None and unsafe_dive_death_corr > 0:
        score -= unsafe_dive_death_corr * 2.0
    score += min(candidate.get("matchup_groups") or 0, 9) * 0.5
    return score


def rank_candidates(candidates: dict[int, dict]) -> list[dict]:
    rows = []
    for candidate in candidates.values():
        row = dict(candidate)
        row["score"] = compute_score(row)
        rows.append(row)
    return sorted(rows, key=lambda row: (row["score"], row.get("checkpoint_step") or 0), reverse=True)


def recommendation_reason(row: dict) -> str:
    parts = []
    win_rate = metric(row, "matchup_avg_win_rate", "common_ai_win_rate")
    if win_rate is not None:
        parts.append(f"win_rate={fmt(win_rate)}")
    min_win_rate = row.get("matchup_min_win_rate")
    if min_win_rate is not None:
        parts.append(f"min_matchup_win_rate={fmt(min_win_rate)}")
    death = metric(row, "matchup_avg_death", "common_ai_death")
    if death is not None:
        parts.append(f"death={fmt(death)}")
    death_p90 = row.get("matchup_max_death_p90")
    if death_p90 is not None:
        parts.append(f"max_death_p90={fmt(death_p90)}")
    enemy_tower_hp = metric(row, "matchup_avg_enemy_tower_hp", "common_ai_enemy_tower_hp")
    if enemy_tower_hp is not None:
        parts.append(f"enemy_tower_hp={fmt(enemy_tower_hp)}")
    groups = row.get("matchup_groups")
    if groups:
        parts.append(f"matchup_groups={groups}")
    timeout_rate = row.get("matchup_avg_timeout_rate")
    if timeout_rate is not None:
        parts.append(f"timeout_rate={fmt(timeout_rate)}")
    unsafe_dive_active_frames = row.get("matchup_avg_unsafe_dive_active_frames")
    if unsafe_dive_active_frames is not None:
        parts.append(f"unsafe_dive_active_frames={fmt(unsafe_dive_active_frames)}")
    push_window_tower_damage_share = row.get("matchup_avg_push_window_tower_damage_share")
    if push_window_tower_damage_share is not None:
        parts.append(f"push_window_tower_damage_share={fmt(push_window_tower_damage_share)}")
    return ", ".join(parts)


def write_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint_step",
        "actual_train_global_step",
        "score",
        "common_ai_win_rate",
        "common_ai_enemy_tower_hp",
        "common_ai_self_tower_hp",
        "common_ai_death",
        "common_ai_frame",
        "selfplay_win_rate",
        "reward",
        "reward_push_window_tower_damage",
        "reward_unsafe_dive",
        "reward_win_result",
        "reward_timeout_tower_gap",
        "matchup_groups",
        "matchup_rows",
        "matchup_filter_eval_only",
        "matchup_filter_opponent_agent",
        "matchup_eval_ids",
        "matchup_repeat_indices",
        "matchup_evaluation_checkpoint_step",
        "matchup_avg_win_rate",
        "matchup_min_win_rate",
        "matchup_avg_death",
        "matchup_avg_death_p90",
        "matchup_max_death_p90",
        "matchup_avg_enemy_tower_hp",
        "matchup_avg_self_tower_hp",
        "matchup_min_self_tower_hp_p10",
        "matchup_avg_frame_p90",
        "matchup_avg_timeout_rate",
        "matchup_max_timeout_rate",
        "matchup_avg_push_window_tower_damage",
        "matchup_avg_unsafe_dive",
        "matchup_avg_unsafe_dive_severity",
        "matchup_avg_push_window_active_frames",
        "matchup_avg_unsafe_dive_active_frames",
        "matchup_avg_push_window_tower_damage_share",
        "matchup_avg_unsafe_dive_death_corr",
        "source",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output_path: Path, title: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "checkpoint_step",
        "score",
        "common_ai_win_rate",
        "common_ai_enemy_tower_hp",
        "common_ai_death",
        "reward_push_window_tower_damage",
        "reward_unsafe_dive",
        "reward_win_result",
        "matchup_groups",
        "matchup_rows",
        "matchup_eval_ids",
        "matchup_repeat_indices",
        "matchup_avg_win_rate",
        "matchup_min_win_rate",
        "matchup_avg_death",
        "matchup_max_death_p90",
        "matchup_min_self_tower_hp_p10",
        "matchup_avg_timeout_rate",
        "matchup_avg_unsafe_dive_severity",
        "matchup_avg_push_window_active_frames",
        "matchup_avg_unsafe_dive_active_frames",
        "matchup_avg_push_window_tower_damage_share",
        "matchup_avg_unsafe_dive_death_corr",
    ]
    lines = [f"# {title}", "", f"- candidates: {len(rows)}"]
    if rows:
        best = rows[0]
        lines.extend(
            [
                f"- recommended_checkpoint: {best.get('checkpoint_step')}",
                f"- reason: {recommendation_reason(best)}",
            ]
        )
    lines.extend(["", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Rank v1.2 checkpoint candidates from logs and matchup summaries.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log-dir", type=Path, help="Directory containing step-*.md training records")
    source.add_argument("--training-csv", type=Path, help="CSV produced by utils/analyze_training_logs.py")
    parser.add_argument("--matchup-csv", type=Path, default=None, help="CSV produced by utils/analyze_run_records.py")
    parser.add_argument("--matchup-opponent-agent", default="common_ai", help="Only use matchup rows from this opponent agent; empty string disables filtering")
    parser.add_argument("--include-training-matchups", action="store_true", help="Include non-eval matchup rows in checkpoint ranking")
    parser.add_argument("--csv", type=Path, default=Path("logs/checkpoint_ranking.csv"), help="CSV output path")
    parser.add_argument("--md", type=Path, default=Path("logs/checkpoint_ranking.md"), help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = collect_candidates(log_dir=args.log_dir, training_csv=args.training_csv)
    attach_matchup_metrics(
        candidates,
        args.matchup_csv,
        eval_only=not args.include_training_matchups,
        opponent_agent=args.matchup_opponent_agent or None,
    )
    rows = rank_candidates(candidates)
    write_csv(rows, args.csv)
    write_markdown(rows, args.md, "Checkpoint Ranking")
    if rows:
        print(f"recommended checkpoint {rows[0]['checkpoint_step']} -> {args.csv} and {args.md}")
    else:
        print(f"no candidates found -> {args.csv} and {args.md}")


if __name__ == "__main__":
    main()
