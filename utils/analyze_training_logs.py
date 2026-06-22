#!/usr/bin/env python3
"""
Summarize training step markdown files into CSV and Markdown tables.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


STEP_RE = re.compile(r"step-(\d+)\.md$")
SECTION_RE = re.compile(r"^## (.+)$")
KV_RE = re.compile(r"^- ([^:]+): (.*)$")

FIELDS = [
    ("step", "Sampling", "target_step"),
    ("actual_train_global_step", "Sampling", "actual_train_global_step"),
    ("episode_cnt", "Basic Metrics", "episode_cnt"),
    ("sample_receive_cnt", "Basic Metrics", "sample_receive_cnt"),
    ("common_ai_win_rate", "Environment Metrics: Common AI", "win_rate"),
    ("common_ai_self_tower_hp", "Environment Metrics: Common AI", "self_tower_hp"),
    ("common_ai_enemy_tower_hp", "Environment Metrics: Common AI", "enemy_tower_hp"),
    ("common_ai_frame", "Environment Metrics: Common AI", "frame"),
    ("common_ai_money_per_frame", "Environment Metrics: Common AI", "money_per_frame"),
    ("common_ai_kill", "Environment Metrics: Common AI", "kill"),
    ("common_ai_death", "Environment Metrics: Common AI", "death"),
    ("common_ai_hurt_to_hero", "Environment Metrics: Common AI", "hurt_to_hero"),
    ("common_ai_hurt_by_hero", "Environment Metrics: Common AI", "hurt_by_hero"),
    ("selfplay_win_rate", "Environment Metrics: Selfplay", "win_rate"),
    ("selfplay_self_tower_hp", "Environment Metrics: Selfplay", "self_tower_hp"),
    ("selfplay_enemy_tower_hp", "Environment Metrics: Selfplay", "enemy_tower_hp"),
    ("selfplay_frame", "Environment Metrics: Selfplay", "frame"),
    ("selfplay_money_per_frame", "Environment Metrics: Selfplay", "money_per_frame"),
    ("selfplay_kill", "Environment Metrics: Selfplay", "kill"),
    ("selfplay_death", "Environment Metrics: Selfplay", "death"),
    ("selfplay_hurt_to_hero", "Environment Metrics: Selfplay", "hurt_to_hero"),
    ("selfplay_hurt_by_hero", "Environment Metrics: Selfplay", "hurt_by_hero"),
    ("reward", "Algorithm Metrics", "reward"),
    ("total_loss", "Algorithm Metrics", "total_loss"),
    ("policy_loss", "Algorithm Metrics", "policy_loss"),
    ("value_loss", "Algorithm Metrics", "value_loss"),
    ("entropy_loss", "Algorithm Metrics", "entropy_loss"),
    ("reward_tower_hp", "Reward Detail Metrics", "tower_hp"),
    ("reward_tower_destroy", "Reward Detail Metrics", "tower_destroy"),
    ("reward_hp", "Reward Detail Metrics", "hp"),
    ("reward_money", "Reward Detail Metrics", "money"),
    ("reward_exp", "Reward Detail Metrics", "exp"),
    ("reward_kill", "Reward Detail Metrics", "kill"),
    ("reward_death", "Reward Detail Metrics", "death"),
    ("reward_forward", "Reward Detail Metrics", "forward"),
    ("reward_enemy_tower_hp_down", "Reward Detail Metrics", "enemy_tower_hp_down"),
    ("reward_self_tower_hp_down", "Reward Detail Metrics", "self_tower_hp_down"),
    ("reward_push_window_tower_damage", "Reward Detail Metrics", "push_window_tower_damage"),
    ("reward_unsafe_dive", "Reward Detail Metrics", "unsafe_dive"),
    ("reward_unsafe_dive_severity", "Reward Detail Metrics", "unsafe_dive_severity"),
    ("reward_push_window_active", "Reward Detail Metrics", "push_window_active"),
    ("reward_unsafe_dive_active", "Reward Detail Metrics", "unsafe_dive_active"),
    ("reward_win_result", "Reward Detail Metrics", "win_result"),
    ("reward_timeout_tower_gap", "Reward Detail Metrics", "timeout_tower_gap"),
]


def parse_value(raw: str):
    value = raw.strip()
    if value == "null":
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def parse_step_file(path: Path) -> dict:
    sections: dict[str, dict[str, object]] = {}
    current_section = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        section_match = SECTION_RE.match(line)
        if section_match:
            current_section = section_match.group(1)
            sections.setdefault(current_section, {})
            continue
        kv_match = KV_RE.match(line)
        if kv_match and current_section:
            key, value = kv_match.groups()
            sections[current_section][key] = parse_value(value)

    row = {"source_file": path.as_posix()}
    step_match = STEP_RE.search(path.name)
    row["step"] = int(step_match.group(1)) if step_match else sections.get("Sampling", {}).get("target_step", "")
    for output_key, section, source_key in FIELDS:
        if output_key == "step":
            continue
        row[output_key] = sections.get(section, {}).get(source_key, "")
    return row


def collect_rows(log_dir: Path) -> list[dict]:
    rows = [parse_step_file(path) for path in log_dir.glob("step-*.md")]
    return sorted(rows, key=lambda row: row.get("step", 0))


def write_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_file"] + [field[0] for field in FIELDS]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_markdown(rows: list[dict], output_path: Path, title: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "step",
        "actual_train_global_step",
        "common_ai_win_rate",
        "common_ai_enemy_tower_hp",
        "common_ai_self_tower_hp",
        "common_ai_death",
        "common_ai_hurt_to_hero",
        "common_ai_hurt_by_hero",
        "selfplay_win_rate",
        "selfplay_hurt_to_hero",
        "selfplay_hurt_by_hero",
        "reward_push_window_tower_damage",
        "reward_unsafe_dive",
        "reward_win_result",
        "reward",
        "total_loss",
    ]
    lines = [f"# {title}", "", f"- source_rows: {len(rows)}", ""]
    if rows:
        best_common = max(rows, key=lambda row: row.get("common_ai_win_rate") or -1)
        best_selfplay = max(rows, key=lambda row: row.get("selfplay_win_rate") or -1)
        lines.extend(
            [
                "## Highlights",
                "",
                f"- best_common_ai_win_rate: {fmt(best_common.get('common_ai_win_rate'))} at step {best_common.get('step')}",
                f"- best_selfplay_win_rate: {fmt(best_selfplay.get('selfplay_win_rate'))} at step {best_selfplay.get('step')}",
                "",
            ]
        )

    lines.extend(["## Step Summary", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize logs/v*/step-*.md training records.")
    parser.add_argument("log_dir", type=Path, help="Directory containing step-*.md files")
    parser.add_argument("--csv", type=Path, default=None, help="CSV output path")
    parser.add_argument("--md", type=Path, default=None, help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_rows(args.log_dir)
    csv_path = args.csv or args.log_dir / "summary.csv"
    md_path = args.md or args.log_dir / "summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, f"Training Summary: {args.log_dir.as_posix()}")
    print(f"wrote {len(rows)} rows to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
