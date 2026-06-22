#!/usr/bin/env python3
"""
Aggregate TrainingRecorder episode_end JSONL files by matchup.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def iter_events(record_dir: Path, stream: str):
    for path in sorted(record_dir.glob(f"{stream}-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def avg(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else ""


def get_agent(payload: dict, index: int):
    agents = payload.get("agents", [])
    if 0 <= index < len(agents):
        return agents[index]
    return {}


def summarize_episode(payload: dict) -> dict:
    monitor_index = int(payload.get("monitor_agent_index", payload.get("monitor_side", 0)) or 0)
    agent = get_agent(payload, monitor_index)
    hero = agent.get("hero") or {}
    enemy_hero = agent.get("enemy_hero") or {}
    tower = agent.get("tower") or {}
    enemy_tower = agent.get("enemy_tower") or {}
    monitor_hero_id = payload.get("monitor_hero_id") or hero.get("config_id")
    opponent_hero_id = payload.get("opponent_hero_id") or enemy_hero.get("config_id")

    return {
        "matchup": f"{monitor_hero_id}_vs_{opponent_hero_id}",
        "is_eval": payload.get("is_eval"),
        "opponent_agent": payload.get("opponent_agent"),
        "win": agent.get("win"),
        "frame_no": payload.get("frame_no"),
        "self_tower_hp": tower.get("hp"),
        "enemy_tower_hp": enemy_tower.get("hp"),
        "kill": hero.get("kill_cnt"),
        "death": hero.get("dead_cnt"),
        "money_cnt": hero.get("money_cnt"),
        "reward_sum": payload.get("reward_sum", [None])[monitor_index],
    }


def collect_rows(record_dir: Path) -> list[dict]:
    episodes = [summarize_episode(event.get("payload", {})) for event in iter_events(record_dir, "episode_end")]
    groups = defaultdict(list)
    for episode in episodes:
        key = (episode["matchup"], episode["is_eval"], episode["opponent_agent"])
        groups[key].append(episode)

    rows = []
    for (matchup, is_eval, opponent_agent), items in sorted(groups.items()):
        rows.append(
            {
                "matchup": matchup,
                "is_eval": is_eval,
                "opponent_agent": opponent_agent,
                "episodes": len(items),
                "win_rate": avg([item["win"] for item in items]),
                "avg_frame": avg([item["frame_no"] for item in items]),
                "avg_self_tower_hp": avg([item["self_tower_hp"] for item in items]),
                "avg_enemy_tower_hp": avg([item["enemy_tower_hp"] for item in items]),
                "avg_kill": avg([item["kill"] for item in items]),
                "avg_death": avg([item["death"] for item in items]),
                "avg_money_cnt": avg([item["money_cnt"] for item in items]),
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
        "is_eval",
        "opponent_agent",
        "episodes",
        "win_rate",
        "avg_frame",
        "avg_self_tower_hp",
        "avg_enemy_tower_hp",
        "avg_kill",
        "avg_death",
        "avg_money_cnt",
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
        "is_eval",
        "opponent_agent",
        "episodes",
        "win_rate",
        "avg_enemy_tower_hp",
        "avg_self_tower_hp",
        "avg_death",
        "avg_frame",
    ]
    lines = [f"# {title}", "", f"- groups: {len(rows)}", ""]
    lines.extend(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate logs/run_records episode_end JSONL files.")
    parser.add_argument("record_dir", type=Path, help="Directory containing episode_end-*.jsonl files")
    parser.add_argument("--csv", type=Path, default=None, help="CSV output path")
    parser.add_argument("--md", type=Path, default=None, help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_rows(args.record_dir)
    csv_path = args.csv or args.record_dir / "matchup_summary.csv"
    md_path = args.md or args.record_dir / "matchup_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, f"Run Record Matchup Summary: {args.record_dir.as_posix()}")
    print(f"wrote {len(rows)} matchup groups to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
