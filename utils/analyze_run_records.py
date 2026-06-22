#!/usr/bin/env python3
"""
Aggregate TrainingRecorder episode_end JSONL files by matchup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

TIMEOUT_FRAME = 20000


def iter_events(record_dir: Path, stream: str):
    for path in sorted(record_dir.glob(f"{stream}-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def avg(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else ""


def quantile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return ""
    index = max(0, math.ceil(q * len(values)) - 1)
    return values[min(index, len(values) - 1)]


def safe_ratio(numerator, denominator):
    if numerator in ("", None) or denominator in ("", None):
        return ""
    if denominator == 0:
        return ""
    return numerator / denominator


def first_non_empty(values):
    for value in values:
        if value not in ("", None):
            return value
    return ""


def pearson_corr(x_values, y_values):
    pairs = [(x, y) for x, y in zip(x_values, y_values) if x is not None and y is not None]
    if len(pairs) < 2:
        return ""
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    x_var = sum((x - x_mean) ** 2 for x, _ in pairs)
    y_var = sum((y - y_mean) ** 2 for _, y in pairs)
    if x_var <= 0 or y_var <= 0:
        return ""
    cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    return cov / ((x_var * y_var) ** 0.5)


def get_agent(payload: dict, index: int):
    agents = payload.get("agents", [])
    if 0 <= index < len(agents):
        return agents[index]
    return {}


def get_reward_detail(payload: dict, index: int):
    reward_detail = payload.get("reward_detail", [])
    if 0 <= index < len(reward_detail):
        return reward_detail[index]
    return {}


def list_get(values, index: int):
    if isinstance(values, list) and 0 <= index < len(values):
        return values[index]
    return None


def summarize_episode(payload: dict) -> dict:
    monitor_index = int(payload.get("monitor_agent_index", payload.get("monitor_side", 0)) or 0)
    agent = get_agent(payload, monitor_index)
    hero = agent.get("hero") or {}
    enemy_hero = agent.get("enemy_hero") or {}
    tower = agent.get("tower") or {}
    enemy_tower = agent.get("enemy_tower") or {}
    reward_detail = get_reward_detail(payload, monitor_index)
    evaluation = payload.get("evaluation") or (payload.get("usr_conf", {}) or {}).get("evaluation") or {}
    monitor_hero_id = payload.get("monitor_hero_id") or hero.get("config_id")
    opponent_hero_id = payload.get("opponent_hero_id") or enemy_hero.get("config_id")
    checkpoint = payload.get("checkpoint") or {}
    checkpoint_step = (
        checkpoint.get("actual_train_global_step")
        or checkpoint.get("train_global_step")
        or checkpoint.get("global_step")
        or checkpoint.get("model_id")
        or checkpoint.get("checkpoint")
        or payload.get("checkpoint_step")
        or payload.get("model_id")
        or evaluation.get("checkpoint_step")
    )

    return {
        "checkpoint_step": checkpoint_step,
        "eval_id": evaluation.get("eval_id"),
        "evaluation_checkpoint_step": evaluation.get("checkpoint_step"),
        "repeat_index": evaluation.get("repeat_index"),
        "matchup": f"{monitor_hero_id}_vs_{opponent_hero_id}",
        "is_eval": payload.get("is_eval"),
        "opponent_agent": payload.get("opponent_agent"),
        "win": agent.get("win"),
        "frame_no": payload.get("frame_no"),
        "is_timeout": 1 if (payload.get("frame_no") or 0) >= TIMEOUT_FRAME else 0,
        "self_tower_hp": tower.get("hp"),
        "enemy_tower_hp": enemy_tower.get("hp"),
        "kill": hero.get("kill_cnt"),
        "death": hero.get("dead_cnt"),
        "money_cnt": hero.get("money_cnt"),
        "reward_sum": list_get(payload.get("reward_sum"), monitor_index),
        "reward_enemy_tower_hp_down": reward_detail.get("enemy_tower_hp_down"),
        "reward_self_tower_hp_down": reward_detail.get("self_tower_hp_down"),
        "reward_push_window_tower_damage": reward_detail.get("push_window_tower_damage"),
        "reward_unsafe_dive": reward_detail.get("unsafe_dive"),
        "push_window_active_frames": reward_detail.get("push_window_active"),
        "unsafe_dive_active_frames": reward_detail.get("unsafe_dive_active"),
        "reward_win_result": reward_detail.get("win_result"),
        "reward_timeout_tower_gap": reward_detail.get("timeout_tower_gap"),
    }


def collect_rows(record_dir: Path) -> list[dict]:
    episodes = [summarize_episode(event.get("payload", {})) for event in iter_events(record_dir, "episode_end")]
    groups = defaultdict(list)
    for episode in episodes:
        key = (
            episode["checkpoint_step"],
            episode["matchup"],
            episode["is_eval"],
            episode["opponent_agent"],
        )
        groups[key].append(episode)

    rows = []
    def group_sort_key(item):
        checkpoint_step, matchup, is_eval, opponent_agent = item[0]
        return (
            checkpoint_step is None,
            str(checkpoint_step),
            str(matchup),
            str(is_eval),
            str(opponent_agent),
        )

    for (checkpoint_step, matchup, is_eval, opponent_agent), items in sorted(groups.items(), key=group_sort_key):
        avg_enemy_tower_down = avg([item["reward_enemy_tower_hp_down"] for item in items])
        avg_push_window_tower_damage = avg([item["reward_push_window_tower_damage"] for item in items])
        avg_unsafe_dive_active_frames = avg([item["unsafe_dive_active_frames"] for item in items])
        rows.append(
            {
                "checkpoint_step": checkpoint_step,
                "eval_ids": ",".join(
                    str(item["eval_id"])
                    for item in items
                    if item.get("eval_id") not in ("", None)
                ),
                "evaluation_checkpoint_step": first_non_empty([item["evaluation_checkpoint_step"] for item in items]),
                "repeat_indices": ",".join(
                    str(item["repeat_index"])
                    for item in items
                    if item.get("repeat_index") not in ("", None)
                ),
                "matchup": matchup,
                "is_eval": is_eval,
                "opponent_agent": opponent_agent,
                "episodes": len(items),
                "win_rate": avg([item["win"] for item in items]),
                "avg_frame": avg([item["frame_no"] for item in items]),
                "frame_p90": quantile([item["frame_no"] for item in items], 0.90),
                "timeout_rate": avg([item["is_timeout"] for item in items]),
                "avg_self_tower_hp": avg([item["self_tower_hp"] for item in items]),
                "self_tower_hp_p10": quantile([item["self_tower_hp"] for item in items], 0.10),
                "avg_enemy_tower_hp": avg([item["enemy_tower_hp"] for item in items]),
                "avg_kill": avg([item["kill"] for item in items]),
                "avg_death": avg([item["death"] for item in items]),
                "death_p90": quantile([item["death"] for item in items], 0.90),
                "avg_money_cnt": avg([item["money_cnt"] for item in items]),
                "avg_reward_sum": avg([item["reward_sum"] for item in items]),
                "avg_reward_enemy_tower_hp_down": avg_enemy_tower_down,
                "avg_reward_self_tower_hp_down": avg([item["reward_self_tower_hp_down"] for item in items]),
                "avg_push_window_tower_damage": avg_push_window_tower_damage,
                "avg_unsafe_dive": avg([item["reward_unsafe_dive"] for item in items]),
                "avg_push_window_active_frames": avg([item["push_window_active_frames"] for item in items]),
                "avg_unsafe_dive_active_frames": avg_unsafe_dive_active_frames,
                "push_window_tower_damage_share": safe_ratio(avg_push_window_tower_damage, avg_enemy_tower_down),
                "unsafe_dive_death_corr": pearson_corr(
                    [item["unsafe_dive_active_frames"] for item in items],
                    [item["death"] for item in items],
                ),
                "avg_win_result": avg([item["reward_win_result"] for item in items]),
                "avg_timeout_tower_gap": avg([item["reward_timeout_tower_gap"] for item in items]),
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
        "checkpoint_step",
        "eval_ids",
        "evaluation_checkpoint_step",
        "repeat_indices",
        "matchup",
        "is_eval",
        "opponent_agent",
        "episodes",
        "win_rate",
        "avg_frame",
        "frame_p90",
        "timeout_rate",
        "avg_self_tower_hp",
        "self_tower_hp_p10",
        "avg_enemy_tower_hp",
        "avg_kill",
        "avg_death",
        "death_p90",
        "avg_money_cnt",
        "avg_reward_sum",
        "avg_reward_enemy_tower_hp_down",
        "avg_reward_self_tower_hp_down",
        "avg_push_window_tower_damage",
        "avg_unsafe_dive",
        "avg_push_window_active_frames",
        "avg_unsafe_dive_active_frames",
        "push_window_tower_damage_share",
        "unsafe_dive_death_corr",
        "avg_win_result",
        "avg_timeout_tower_gap",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output_path: Path, title: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "checkpoint_step",
        "eval_ids",
        "repeat_indices",
        "matchup",
        "is_eval",
        "opponent_agent",
        "episodes",
        "win_rate",
        "avg_enemy_tower_hp",
        "avg_self_tower_hp",
        "self_tower_hp_p10",
        "avg_death",
        "death_p90",
        "timeout_rate",
        "avg_push_window_tower_damage",
        "avg_unsafe_dive",
        "push_window_tower_damage_share",
        "unsafe_dive_death_corr",
        "avg_push_window_active_frames",
        "avg_unsafe_dive_active_frames",
        "avg_frame",
        "frame_p90",
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
