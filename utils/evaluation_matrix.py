#!/usr/bin/env python3
"""
Generate fixed evaluation matrices for checkpoint and matchup comparisons.
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
DEFAULT_CHECKPOINTS = [15000, 17057]
DEFAULT_SKILLS = [80107, 80110, 80115]


def parse_ids(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def skill_name(skill_id: int) -> str:
    return SUMMONER_SKILL_MAP.get(skill_id, str(skill_id))


def build_rows(
    checkpoints=None,
    hero_ids=None,
    repeats=20,
    include_skill_grid=False,
    candidate_skills=None,
    opponent_agent="common_ai",
):
    checkpoints = checkpoints or DEFAULT_CHECKPOINTS
    hero_ids = hero_ids or DEFAULT_HERO_IDS
    candidate_skills = candidate_skills or DEFAULT_SKILLS
    rows = []
    eval_id = 0

    for checkpoint in checkpoints:
        for blue_hero_id in hero_ids:
            for red_hero_id in hero_ids:
                for monitor_side in (0, 1):
                    blue_skill_candidates = [select_summoner_skill(blue_hero_id, red_hero_id)]
                    red_skill_candidates = [select_summoner_skill(red_hero_id, blue_hero_id)]
                    if include_skill_grid:
                        blue_skill_candidates = candidate_skills
                        red_skill_candidates = candidate_skills

                    for blue_skill in blue_skill_candidates:
                        for red_skill in red_skill_candidates:
                            for repeat_index in range(1, repeats + 1):
                                eval_id += 1
                                monitor_hero_id = blue_hero_id if monitor_side == 0 else red_hero_id
                                opponent_hero_id = red_hero_id if monitor_side == 0 else blue_hero_id
                                rows.append(
                                    {
                                        "eval_id": eval_id,
                                        "checkpoint_step": checkpoint,
                                        "opponent_agent": opponent_agent,
                                        "blue_hero_id": blue_hero_id,
                                        "red_hero_id": red_hero_id,
                                        "monitor_side": monitor_side,
                                        "monitor_hero_id": monitor_hero_id,
                                        "opponent_hero_id": opponent_hero_id,
                                        "matchup": f"{monitor_hero_id}_vs_{opponent_hero_id}",
                                        "blue_select_skill": blue_skill,
                                        "blue_select_skill_name": skill_name(blue_skill),
                                        "red_select_skill": red_skill,
                                        "red_select_skill_name": skill_name(red_skill),
                                        "repeat_index": repeat_index,
                                    }
                                )
    return rows


def summarize_rows(rows):
    checkpoint_count = len({row["checkpoint_step"] for row in rows})
    matchup_count = len({row["matchup"] for row in rows})
    side_count = len({row["monitor_side"] for row in rows})
    return {
        "rows": len(rows),
        "checkpoints": checkpoint_count,
        "matchups": matchup_count,
        "monitor_sides": side_count,
    }


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "eval_id",
        "checkpoint_step",
        "opponent_agent",
        "blue_hero_id",
        "red_hero_id",
        "monitor_side",
        "monitor_hero_id",
        "opponent_hero_id",
        "matchup",
        "blue_select_skill",
        "blue_select_skill_name",
        "red_select_skill",
        "red_select_skill_name",
        "repeat_index",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, output_path: Path, title="Evaluation Matrix"):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    columns = [
        "checkpoint_step",
        "matchup",
        "monitor_side",
        "opponent_agent",
        "blue_select_skill",
        "red_select_skill",
        "repeat_index",
    ]
    lines = [
        f"# {title}",
        "",
        f"- rows: {summary['rows']}",
        f"- checkpoints: {summary['checkpoints']}",
        f"- matchups: {summary['matchups']}",
        f"- monitor_sides: {summary['monitor_sides']}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    preview_rows = rows[: min(60, len(rows))]
    for row in preview_rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    if len(rows) > len(preview_rows):
        lines.append(f"| ... | ... | ... | ... | ... | ... | {len(rows) - len(preview_rows)} more rows |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a fixed checkpoint x matchup evaluation matrix.")
    parser.add_argument("--checkpoints", default="15000,17057", help="Comma-separated checkpoint steps")
    parser.add_argument("--heroes", default="112,133,199", help="Comma-separated hero IDs")
    parser.add_argument("--repeats", type=int, default=20, help="Repeats per checkpoint/matchup/side/skill setting")
    parser.add_argument("--opponent-agent", default="common_ai", help="Evaluation opponent agent")
    parser.add_argument("--skill-grid", action="store_true", help="Expand all candidate summoner skills")
    parser.add_argument("--skills", default="80107,80110,80115", help="Comma-separated summoner skill IDs for --skill-grid")
    parser.add_argument("--csv", type=Path, default=Path("logs/evaluation_matrix.csv"), help="CSV output path")
    parser.add_argument("--md", type=Path, default=Path("logs/evaluation_matrix.md"), help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_rows(
        checkpoints=parse_ids(args.checkpoints),
        hero_ids=parse_ids(args.heroes),
        repeats=args.repeats,
        include_skill_grid=args.skill_grid,
        candidate_skills=parse_ids(args.skills),
        opponent_agent=args.opponent_agent,
    )
    write_csv(rows, args.csv)
    write_markdown(rows, args.md)
    print(f"wrote {len(rows)} evaluation rows to {args.csv} and {args.md}")


if __name__ == "__main__":
    main()
