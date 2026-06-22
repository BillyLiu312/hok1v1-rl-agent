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


def attach_matchup_metrics(candidates: dict[int, dict], matchup_csv: Path | None):
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
        raw_step = to_int(row.get("checkpoint_step"))
        if raw_step is None:
            continue
        step = step_aliases.get(raw_step, raw_step)
        grouped[step].append(row)

    for step, rows in grouped.items():
        candidate = candidates.setdefault(step, {"checkpoint_step": step, "source": ""})
        win_rates = [to_float(row.get("win_rate")) for row in rows]
        win_rates = [value for value in win_rates if value is not None]
        deaths = [to_float(row.get("avg_death")) for row in rows]
        deaths = [value for value in deaths if value is not None]
        enemy_tower_hp = [to_float(row.get("avg_enemy_tower_hp")) for row in rows]
        enemy_tower_hp = [value for value in enemy_tower_hp if value is not None]
        self_tower_hp = [to_float(row.get("avg_self_tower_hp")) for row in rows]
        self_tower_hp = [value for value in self_tower_hp if value is not None]
        push_window_tower_damage = [to_float(row.get("avg_push_window_tower_damage")) for row in rows]
        push_window_tower_damage = [value for value in push_window_tower_damage if value is not None]
        unsafe_dive = [to_float(row.get("avg_unsafe_dive")) for row in rows]
        unsafe_dive = [value for value in unsafe_dive if value is not None]

        candidate.update(
            {
                "matchup_groups": len(rows),
                "matchup_avg_win_rate": avg(win_rates),
                "matchup_min_win_rate": min(win_rates) if win_rates else None,
                "matchup_avg_death": avg(deaths),
                "matchup_avg_enemy_tower_hp": avg(enemy_tower_hp),
                "matchup_avg_self_tower_hp": avg(self_tower_hp),
                "matchup_avg_push_window_tower_damage": avg(push_window_tower_damage),
                "matchup_avg_unsafe_dive": avg(unsafe_dive),
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
    enemy_tower_hp = metric(row, "matchup_avg_enemy_tower_hp", "common_ai_enemy_tower_hp")
    if enemy_tower_hp is not None:
        parts.append(f"enemy_tower_hp={fmt(enemy_tower_hp)}")
    groups = row.get("matchup_groups")
    if groups:
        parts.append(f"matchup_groups={groups}")
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
        "matchup_groups",
        "matchup_avg_win_rate",
        "matchup_min_win_rate",
        "matchup_avg_death",
        "matchup_avg_enemy_tower_hp",
        "matchup_avg_self_tower_hp",
        "matchup_avg_push_window_tower_damage",
        "matchup_avg_unsafe_dive",
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
        "matchup_groups",
        "matchup_avg_win_rate",
        "matchup_min_win_rate",
        "matchup_avg_death",
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
    parser.add_argument("--csv", type=Path, default=Path("logs/checkpoint_ranking.csv"), help="CSV output path")
    parser.add_argument("--md", type=Path, default=Path("logs/checkpoint_ranking.md"), help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = collect_candidates(log_dir=args.log_dir, training_csv=args.training_csv)
    attach_matchup_metrics(candidates, args.matchup_csv)
    rows = rank_candidates(candidates)
    write_csv(rows, args.csv)
    write_markdown(rows, args.md, "Checkpoint Ranking")
    if rows:
        print(f"recommended checkpoint {rows[0]['checkpoint_step']} -> {args.csv} and {args.md}")
    else:
        print(f"no candidates found -> {args.csv} and {args.md}")


if __name__ == "__main__":
    main()
