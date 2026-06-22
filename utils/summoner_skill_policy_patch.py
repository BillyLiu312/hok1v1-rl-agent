#!/usr/bin/env python3
"""
Export matchup-conditioned summoner skill recommendations as reviewable policy patches.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_ppo.conf.summoner_skill import SUMMONER_SKILL_MAP, select_summoner_skill


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


def to_int(value, default=None):
    number = to_float(value, default=None)
    if number is None:
        return default
    return int(number)


def parse_matchup(raw_matchup: str):
    if not raw_matchup or "_vs_" not in raw_matchup:
        return None, None
    left, right = raw_matchup.split("_vs_", 1)
    return to_int(left), to_int(right)


def build_patch_rows(rows: list[dict], min_episodes=1, min_win_delta=0.0) -> list[dict]:
    patch_rows = []
    for row in rows:
        if str(row.get("needs_policy_update", "")).lower() not in ("true", "1", "yes"):
            continue
        episodes = to_float(row.get("recommended_episodes"), 0.0) or 0.0
        if episodes < min_episodes:
            continue

        hero_id, opponent_hero_id = parse_matchup(row.get("matchup", ""))
        recommended_skill = to_int(row.get("recommended_skill"))
        current_skill = to_int(row.get("current_policy_skill"))
        if hero_id is None or opponent_hero_id is None or recommended_skill is None:
            continue

        current_win_rate = to_float(row.get("current_policy_win_rate"))
        recommended_win_rate = to_float(row.get("recommended_win_rate"))
        win_delta = ""
        if current_win_rate is not None and recommended_win_rate is not None:
            win_delta = recommended_win_rate - current_win_rate
            if win_delta < min_win_delta:
                continue

        patch_rows.append(
            {
                "checkpoint_step": row.get("checkpoint_step", ""),
                "hero_id": hero_id,
                "opponent_hero_id": opponent_hero_id,
                "recommended_skill": recommended_skill,
                "recommended_skill_name": SUMMONER_SKILL_MAP.get(recommended_skill, str(recommended_skill)),
                "current_policy_skill": current_skill if current_skill is not None else select_summoner_skill(hero_id, opponent_hero_id),
                "current_policy_skill_name": SUMMONER_SKILL_MAP.get(current_skill, str(current_skill) if current_skill else ""),
                "recommended_win_rate": recommended_win_rate if recommended_win_rate is not None else "",
                "current_policy_win_rate": current_win_rate if current_win_rate is not None else "",
                "win_rate_delta": win_delta,
                "recommended_avg_death": row.get("recommended_avg_death", ""),
                "recommended_avg_enemy_tower_hp": row.get("recommended_avg_enemy_tower_hp", ""),
                "recommended_episodes": row.get("recommended_episodes", ""),
                "recommendation_score": row.get("recommendation_score", ""),
            }
        )
    return sorted(
        patch_rows,
        key=lambda item: (
            str(item["checkpoint_step"]),
            item["hero_id"],
            item["opponent_hero_id"],
            item["recommended_skill"],
        ),
    )


def fmt(value):
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_python_patch(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Review before copying into agent_ppo/conf/summoner_skill.py.",
        "# Generated from summoner_skill_recommendations.csv.",
        "MATCHUP_SUMMONER_SKILL_OVERRIDES_CANDIDATE = {",
    ]
    for row in rows:
        lines.append(
            "    ({hero_id}, {opponent_hero_id}): {recommended_skill},  # {recommended_skill_name}; "
            "win_delta={win_rate_delta}, episodes={recommended_episodes}, checkpoint={checkpoint_step}".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    lines.append("}")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(rows: list[dict], output_path: Path, title="Summoner Skill Policy Patch Candidates"):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "checkpoint_step",
        "hero_id",
        "opponent_hero_id",
        "recommended_skill",
        "recommended_skill_name",
        "current_policy_skill",
        "current_policy_skill_name",
        "recommended_win_rate",
        "current_policy_win_rate",
        "win_rate_delta",
        "recommended_avg_death",
        "recommended_avg_enemy_tower_hp",
        "recommended_episodes",
    ]
    lines = [f"# {title}", "", f"- candidates: {len(rows)}", ""]
    lines.extend(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Export recommended summoner skill policy override candidates.")
    parser.add_argument("recommendations_csv", type=Path, help="summoner_skill_recommendations.csv")
    parser.add_argument("--py", type=Path, default=None, help="Python dict candidate output path")
    parser.add_argument("--md", type=Path, default=None, help="Markdown review output path")
    parser.add_argument("--min-episodes", type=int, default=1, help="Minimum recommended episodes required")
    parser.add_argument("--min-win-delta", type=float, default=0.0, help="Minimum win-rate delta over current policy")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_patch_rows(
        read_rows(args.recommendations_csv),
        min_episodes=args.min_episodes,
        min_win_delta=args.min_win_delta,
    )
    py_path = args.py or args.recommendations_csv.with_name("summoner_skill_policy_patch.py")
    md_path = args.md or args.recommendations_csv.with_name("summoner_skill_policy_patch.md")
    write_python_patch(rows, py_path)
    write_markdown(rows, md_path)
    print(f"wrote {len(rows)} policy patch candidates to {py_path} and {md_path}")


if __name__ == "__main__":
    main()
