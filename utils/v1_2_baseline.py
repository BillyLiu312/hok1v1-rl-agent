#!/usr/bin/env python3
"""
Build v1.2 acceptance baselines from v1.1 training logs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.analyze_training_logs import collect_rows


DEFAULT_LOG_DIR = Path("logs/v1.1")
DEFAULT_JSON = Path("logs/v1.2/baseline_v1.1.json")
DEFAULT_MD = Path("logs/v1.2/baseline_v1.1.md")


def metric(row: dict, key: str):
    value = row.get(key)
    return value if value not in ("", None) else None


def metric_delta(row: dict, positive_key: str, negative_key: str):
    positive = metric(row, positive_key)
    negative = metric(row, negative_key)
    if positive is None or negative is None:
        return None
    return positive - negative


def build_baseline(log_dir: Path = DEFAULT_LOG_DIR) -> dict:
    rows = collect_rows(log_dir)
    common_ai_rows = [row for row in rows if metric(row, "common_ai_win_rate") is not None]
    if not common_ai_rows:
        return {
            "source_log_dir": log_dir.as_posix(),
            "rows": len(rows),
            "status": "MISSING",
            "reason": "No common_ai win-rate rows found.",
        }

    best_win_row = max(common_ai_rows, key=lambda row: metric(row, "common_ai_win_rate"))
    last_step_row = max(common_ai_rows, key=lambda row: metric(row, "step") or -1)
    best_tower_row = min(
        [row for row in common_ai_rows if metric(row, "common_ai_enemy_tower_hp") is not None],
        key=lambda row: metric(row, "common_ai_enemy_tower_hp"),
        default=best_win_row,
    )

    return {
        "source_log_dir": log_dir.as_posix(),
        "rows": len(rows),
        "status": "PASS",
        "best_win_step": best_win_row.get("step"),
        "best_win_rate": best_win_row.get("common_ai_win_rate"),
        "best_win_enemy_tower_hp": best_win_row.get("common_ai_enemy_tower_hp"),
        "best_win_death": best_win_row.get("common_ai_death"),
        "best_win_hurt_to_hero": best_win_row.get("common_ai_hurt_to_hero"),
        "best_win_hurt_by_hero": best_win_row.get("common_ai_hurt_by_hero"),
        "best_win_hero_damage_balance": metric_delta(
            best_win_row,
            "common_ai_hurt_to_hero",
            "common_ai_hurt_by_hero",
        ),
        "best_tower_step": best_tower_row.get("step"),
        "best_enemy_tower_hp": best_tower_row.get("common_ai_enemy_tower_hp"),
        "late_step": last_step_row.get("step"),
        "late_death": last_step_row.get("common_ai_death"),
        "late_win_rate": last_step_row.get("common_ai_win_rate"),
    }


def write_json(data: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value):
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_markdown(data: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("status", data.get("status")),
        ("source_log_dir", data.get("source_log_dir")),
        ("rows", data.get("rows")),
        ("best_win_step", data.get("best_win_step")),
        ("best_win_rate", data.get("best_win_rate")),
        ("best_win_enemy_tower_hp", data.get("best_win_enemy_tower_hp")),
        ("best_win_death", data.get("best_win_death")),
        ("best_win_hurt_to_hero", data.get("best_win_hurt_to_hero")),
        ("best_win_hurt_by_hero", data.get("best_win_hurt_by_hero")),
        ("best_win_hero_damage_balance", data.get("best_win_hero_damage_balance")),
        ("best_tower_step", data.get("best_tower_step")),
        ("best_enemy_tower_hp", data.get("best_enemy_tower_hp")),
        ("late_step", data.get("late_step")),
        ("late_death", data.get("late_death")),
        ("late_win_rate", data.get("late_win_rate")),
    ]
    lines = ["# v1.1 Baseline For v1.2", ""]
    lines.extend(["| metric | value |", "| --- | --- |"])
    for key, value in rows:
        lines.append(f"| {key} | {fmt(value)} |")
    if data.get("reason"):
        lines.extend(["", f"- reason: {data['reason']}"])
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build v1.2 baseline thresholds from v1.1 training logs.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory containing v1.1 step-*.md logs")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="JSON output path")
    parser.add_argument("--md", type=Path, default=DEFAULT_MD, help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    data = build_baseline(args.log_dir)
    write_json(data, args.json)
    write_markdown(data, args.md)
    print(f"wrote v1.2 baseline to {args.json} and {args.md}: {data.get('status')}")


if __name__ == "__main__":
    main()
