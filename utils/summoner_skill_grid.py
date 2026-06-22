#!/usr/bin/env python3
"""
Generate matchup-conditioned summoner skill experiment grids.
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


DEFAULT_HERO_IDS = [112, 133, 199]
DEFAULT_CANDIDATE_SKILLS = [80107, 80110, 80115]


def build_rows(hero_ids=None, candidate_skills=None):
    hero_ids = hero_ids or DEFAULT_HERO_IDS
    candidate_skills = candidate_skills or DEFAULT_CANDIDATE_SKILLS
    rows = []
    for hero_id in hero_ids:
        for opponent_hero_id in hero_ids:
            selected_skill = select_summoner_skill(hero_id, opponent_hero_id)
            for skill_id in candidate_skills:
                rows.append(
                    {
                        "hero_id": hero_id,
                        "opponent_hero_id": opponent_hero_id,
                        "skill_id": skill_id,
                        "skill_name": SUMMONER_SKILL_MAP.get(skill_id, str(skill_id)),
                        "is_current_policy": skill_id == selected_skill,
                    }
                )
    return rows


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["hero_id", "opponent_hero_id", "skill_id", "skill_name", "is_current_policy"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Summoner Skill Matchup Grid",
        "",
        "| hero_id | opponent_hero_id | skill_id | skill_name | is_current_policy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {hero_id} | {opponent_hero_id} | {skill_id} | {skill_name} | {is_current_policy} |".format(**row)
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_ids(raw: str):
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a summoner skill experiment grid.")
    parser.add_argument("--heroes", default="112,133,199", help="Comma-separated hero IDs")
    parser.add_argument("--skills", default="80107,80110,80115", help="Comma-separated summoner skill IDs")
    parser.add_argument("--csv", type=Path, default=Path("logs/summoner_skill_grid.csv"), help="CSV output path")
    parser.add_argument("--md", type=Path, default=Path("logs/summoner_skill_grid.md"), help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_rows(hero_ids=parse_ids(args.heroes), candidate_skills=parse_ids(args.skills))
    write_csv(rows, args.csv)
    write_markdown(rows, args.md)
    print(f"wrote {len(rows)} rows to {args.csv} and {args.md}")


if __name__ == "__main__":
    main()
