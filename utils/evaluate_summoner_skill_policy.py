#!/usr/bin/env python3
"""
Evaluate matchup-conditioned summoner skill recommendations before policy review.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MIN_RECOMMENDED_EPISODES = 20
MIN_POLICY_WIN_DELTA = 0.05
MAX_DEATH_REGRESSION = 0.5
MAX_ENEMY_TOWER_HP_REGRESSION = 500.0


def read_rows(path: Path) -> list[dict]:
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


def is_truthy(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def gate(status: str, name: str, observed, threshold: str, detail: str) -> dict:
    return {
        "status": status,
        "gate": name,
        "observed": observed,
        "threshold": threshold,
        "detail": detail,
    }


def evaluate_recommendations(rows: list[dict]) -> list[dict]:
    gates = []
    if not rows:
        return [
            gate(
                "MISSING",
                "recommendations",
                0,
                ">= 1 recommendation row",
                "No summoner skill recommendation rows are available.",
            )
        ]

    update_rows = [row for row in rows if is_truthy(row.get("needs_policy_update"))]
    gates.append(
        gate(
            "PASS" if update_rows else "WARN",
            "policy_update_candidates",
            len(update_rows),
            ">= 1 when policy differs from evidence",
            "A WARN here means the current rule table already matches the best observed skill or evidence is incomplete.",
        )
    )

    missing_current = [
        row
        for row in rows
        if row.get("current_policy_skill") in ("", None) or row.get("current_policy_win_rate") in ("", None)
    ]
    gates.append(
        gate(
            "PASS" if not missing_current else "WARN",
            "current_policy_coverage",
            len(rows) - len(missing_current),
            f"{len(rows)} recommendation rows",
            "Each matchup recommendation should include the current policy skill result for a fair delta.",
        )
    )

    weak_episode_rows = [
        row
        for row in update_rows
        if (to_float(row.get("recommended_episodes"), 0.0) or 0.0) < MIN_RECOMMENDED_EPISODES
    ]
    gates.append(
        gate(
            "PASS" if not weak_episode_rows else "WARN",
            "recommended_episodes",
            len(update_rows) - len(weak_episode_rows),
            f"all update candidates >= {MIN_RECOMMENDED_EPISODES}",
            "Avoid applying matchup skill updates from tiny samples.",
        )
    )

    win_deltas = []
    missing_delta = 0
    for row in update_rows:
        recommended = to_float(row.get("recommended_win_rate"))
        current = to_float(row.get("current_policy_win_rate"))
        if recommended is None or current is None:
            missing_delta += 1
            continue
        win_deltas.append(recommended - current)
    weak_delta_rows = [value for value in win_deltas if value < MIN_POLICY_WIN_DELTA]
    if missing_delta:
        status = "WARN"
    elif weak_delta_rows:
        status = "WARN"
    else:
        status = "PASS"
    gates.append(
        gate(
            status,
            "policy_win_delta",
            min(win_deltas) if win_deltas else "",
            f">= {MIN_POLICY_WIN_DELTA}",
            "Recommended skill should beat the current policy by a meaningful win-rate margin.",
        )
    )

    death_regressions = []
    enemy_tower_regressions = []
    for row in update_rows:
        recommended_death = to_float(row.get("recommended_avg_death"))
        current_death = to_float(row.get("current_policy_avg_death"))
        if recommended_death is not None and current_death is not None:
            death_regressions.append(recommended_death - current_death)
        recommended_tower = to_float(row.get("recommended_avg_enemy_tower_hp"))
        current_tower = to_float(row.get("current_policy_avg_enemy_tower_hp"))
        if recommended_tower is not None and current_tower is not None:
            enemy_tower_regressions.append(recommended_tower - current_tower)

    max_death_regression = max(death_regressions) if death_regressions else ""
    gates.append(
        gate(
            "PASS" if max_death_regression == "" or max_death_regression <= MAX_DEATH_REGRESSION else "WARN",
            "death_regression",
            max_death_regression,
            f"<= {MAX_DEATH_REGRESSION}",
            "Skill updates should not buy win rate with a large death increase.",
        )
    )

    max_tower_regression = max(enemy_tower_regressions) if enemy_tower_regressions else ""
    gates.append(
        gate(
            "PASS" if max_tower_regression == "" or max_tower_regression <= MAX_ENEMY_TOWER_HP_REGRESSION else "WARN",
            "tower_pressure_regression",
            max_tower_regression,
            f"<= {MAX_ENEMY_TOWER_HP_REGRESSION}",
            "Skill updates should not substantially worsen enemy tower HP.",
        )
    )

    return gates


def overall_status(rows: list[dict]) -> str:
    statuses = {row["status"] for row in rows}
    if "FAIL" in statuses or "MISSING" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def write_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["status", "gate", "observed", "threshold", "detail"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_markdown(rows: list[dict], output_path: Path, title="Summoner Skill Policy Gate"):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["status", "gate", "observed", "threshold", "detail"]
    lines = [f"# {title}", "", f"- overall_status: {overall_status(rows)}", ""]
    lines.extend(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate summoner skill policy recommendations.")
    parser.add_argument("recommendations_csv", type=Path, help="summoner_skill_recommendations.csv")
    parser.add_argument("--csv", type=Path, default=None, help="Gate CSV output path")
    parser.add_argument("--md", type=Path, default=None, help="Gate Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = evaluate_recommendations(read_rows(args.recommendations_csv))
    csv_path = args.csv or args.recommendations_csv.with_name("summoner_skill_policy_gate.csv")
    md_path = args.md or args.recommendations_csv.with_name("summoner_skill_policy_gate.md")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"wrote summoner skill policy gate to {csv_path} and {md_path}: {overall_status(rows)}")


if __name__ == "__main__":
    main()
