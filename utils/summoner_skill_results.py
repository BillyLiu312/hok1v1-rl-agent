#!/usr/bin/env python3
"""
Aggregate episode results by matchup-conditioned summoner skill choices.
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

from agent_ppo.conf.summoner_skill import SUMMONER_SKILL_MAP, select_summoner_skill
from utils.analyze_run_records import avg, iter_events, summarize_episode


def first_lineup_entry(usr_conf: dict, camp_key: str) -> dict:
    lineups = usr_conf.get("lineups", {})
    entries = lineups.get(camp_key, [])
    return entries[0] if entries else {}


def get_skill(entry: dict):
    raw_skill = entry.get("select_skill") or entry.get("summoner_skill_id") or entry.get("skill_id")
    if raw_skill in ("", None):
        return None
    try:
        return int(raw_skill)
    except (TypeError, ValueError):
        return raw_skill


def summarize_skill_episode(payload: dict) -> dict:
    episode = summarize_episode(payload)
    usr_conf = payload.get("usr_conf", {})
    monitor_side = int(payload.get("monitor_agent_index", payload.get("monitor_side", 0)) or 0)
    blue = first_lineup_entry(usr_conf, "blue_camp")
    red = first_lineup_entry(usr_conf, "red_camp")
    monitor_entry = blue if monitor_side == 0 else red
    opponent_entry = red if monitor_side == 0 else blue
    monitor_skill = get_skill(monitor_entry)
    opponent_skill = get_skill(opponent_entry)
    monitor_hero_id = episode.get("matchup", "_vs_").split("_vs_")[0]
    opponent_hero_id = episode.get("matchup", "_vs_").split("_vs_")[1]

    try:
        current_policy_skill = select_summoner_skill(int(monitor_hero_id), int(opponent_hero_id))
    except (TypeError, ValueError):
        current_policy_skill = None

    return {
        **episode,
        "monitor_skill": monitor_skill,
        "monitor_skill_name": SUMMONER_SKILL_MAP.get(monitor_skill, str(monitor_skill) if monitor_skill else ""),
        "opponent_skill": opponent_skill,
        "opponent_skill_name": SUMMONER_SKILL_MAP.get(opponent_skill, str(opponent_skill) if opponent_skill else ""),
        "is_current_policy_skill": monitor_skill == current_policy_skill if monitor_skill is not None else "",
    }


def collect_rows(record_dir: Path) -> list[dict]:
    episodes = [
        summarize_skill_episode(event.get("payload", {}))
        for event in iter_events(record_dir, "episode_end")
    ]
    groups = defaultdict(list)
    for episode in episodes:
        key = (
            episode["matchup"],
            episode["monitor_skill"],
            episode["opponent_skill"],
            episode["is_eval"],
            episode["opponent_agent"],
        )
        groups[key].append(episode)

    rows = []
    for (matchup, monitor_skill, opponent_skill, is_eval, opponent_agent), items in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        rows.append(
            {
                "matchup": matchup,
                "monitor_skill": monitor_skill,
                "monitor_skill_name": SUMMONER_SKILL_MAP.get(
                    monitor_skill,
                    str(monitor_skill) if monitor_skill else "",
                ),
                "opponent_skill": opponent_skill,
                "opponent_skill_name": SUMMONER_SKILL_MAP.get(
                    opponent_skill,
                    str(opponent_skill) if opponent_skill else "",
                ),
                "is_current_policy_skill": items[0]["is_current_policy_skill"],
                "is_eval": is_eval,
                "opponent_agent": opponent_agent,
                "episodes": len(items),
                "win_rate": avg([item["win"] for item in items]),
                "avg_frame": avg([item["frame_no"] for item in items]),
                "avg_death": avg([item["death"] for item in items]),
                "avg_enemy_tower_hp": avg([item["enemy_tower_hp"] for item in items]),
                "avg_reward_sum": avg([item["reward_sum"] for item in items]),
            }
        )
    return rows


def fmt(value):
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "matchup",
        "monitor_skill",
        "monitor_skill_name",
        "opponent_skill",
        "opponent_skill_name",
        "is_current_policy_skill",
        "is_eval",
        "opponent_agent",
        "episodes",
        "win_rate",
        "avg_frame",
        "avg_death",
        "avg_enemy_tower_hp",
        "avg_reward_sum",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output_path: Path, title: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "matchup",
        "monitor_skill",
        "monitor_skill_name",
        "opponent_skill",
        "is_current_policy_skill",
        "episodes",
        "win_rate",
        "avg_death",
        "avg_enemy_tower_hp",
    ]
    lines = [f"# {title}", "", f"- groups: {len(rows)}", ""]
    lines.extend(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate summoner skill outcomes from run records.")
    parser.add_argument("record_dir", type=Path, help="Directory containing episode_end-*.jsonl files")
    parser.add_argument("--csv", type=Path, default=None, help="CSV output path")
    parser.add_argument("--md", type=Path, default=None, help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_rows(args.record_dir)
    csv_path = args.csv or args.record_dir / "summoner_skill_results.csv"
    md_path = args.md or args.record_dir / "summoner_skill_results.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, f"Summoner Skill Results: {args.record_dir.as_posix()}")
    print(f"wrote {len(rows)} summoner skill groups to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
