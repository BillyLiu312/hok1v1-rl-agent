#!/usr/bin/env python3
"""
Build checkpoint-vs-opponent win-rate matrices and Elo rankings from run records.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.analyze_run_records import iter_events, summarize_episode


def player_key(value):
    if value in ("", None):
        return None
    return str(value)


def collect_games(record_dir: Path) -> list[dict]:
    games = []
    for event in iter_events(record_dir, "episode_end"):
        payload = event.get("payload", {})
        episode = summarize_episode(payload)
        checkpoint = player_key(episode.get("checkpoint_step"))
        opponent = player_key(episode.get("opponent_agent"))
        win = episode.get("win")
        if checkpoint is None or opponent is None or win is None:
            continue
        try:
            score = float(win)
        except (TypeError, ValueError):
            continue
        score = max(0.0, min(1.0, score))
        games.append(
            {
                "checkpoint": checkpoint,
                "opponent": opponent,
                "score": score,
                "matchup": episode.get("matchup"),
                "is_eval": episode.get("is_eval"),
                "frame_no": episode.get("frame_no"),
                "death": episode.get("death"),
                "enemy_tower_hp": episode.get("enemy_tower_hp"),
            }
        )
    return games


def avg(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else ""


def summarize_matrix(games: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for game in games:
        groups[(game["checkpoint"], game["opponent"])].append(game)

    rows = []
    for (checkpoint, opponent), items in sorted(groups.items()):
        rows.append(
            {
                "checkpoint": checkpoint,
                "opponent": opponent,
                "games": len(items),
                "win_rate": avg([item["score"] for item in items]),
                "avg_frame": avg([item["frame_no"] for item in items]),
                "avg_death": avg([item["death"] for item in items]),
                "avg_enemy_tower_hp": avg([item["enemy_tower_hp"] for item in items]),
            }
        )
    return rows


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def compute_elo(games: list[dict], base_rating=1000.0, k_factor=32.0) -> list[dict]:
    ratings = defaultdict(lambda: float(base_rating))
    game_counts = defaultdict(int)

    for game in games:
        checkpoint = game["checkpoint"]
        opponent = game["opponent"]
        score = game["score"]
        rating_a = ratings[checkpoint]
        rating_b = ratings[opponent]
        expected_a = expected_score(rating_a, rating_b)
        delta = k_factor * (score - expected_a)
        ratings[checkpoint] = rating_a + delta
        ratings[opponent] = rating_b - delta
        game_counts[checkpoint] += 1
        game_counts[opponent] += 1

    return sorted(
        [
            {
                "player": player,
                "elo": rating,
                "games": game_counts[player],
            }
            for player, rating in ratings.items()
        ],
        key=lambda row: (row["elo"], row["games"], row["player"]),
        reverse=True,
    )


def fmt(value):
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_csv(rows: list[dict], output_path: Path, fieldnames: list[str]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(matrix_rows: list[dict], elo_rows: list[dict], output_path: Path, title: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_columns = ["checkpoint", "opponent", "games", "win_rate", "avg_death", "avg_enemy_tower_hp"]
    elo_columns = ["player", "elo", "games"]
    lines = [f"# {title}", "", f"- matrix_rows: {len(matrix_rows)}", f"- elo_players: {len(elo_rows)}", ""]
    lines.extend(["## Elo Ranking", "", "| " + " | ".join(elo_columns) + " |", "| " + " | ".join(["---"] * len(elo_columns)) + " |"])
    for row in elo_rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in elo_columns) + " |")
    lines.extend(["", "## Win-Rate Matrix", "", "| " + " | ".join(matrix_columns) + " |", "| " + " | ".join(["---"] * len(matrix_columns)) + " |"])
    for row in matrix_rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in matrix_columns) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_checkpoint_matrix(record_dir: Path):
    games = collect_games(record_dir)
    matrix_rows = summarize_matrix(games)
    elo_rows = compute_elo(games)
    return matrix_rows, elo_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Build checkpoint-vs-opponent matrix and Elo ranking.")
    parser.add_argument("record_dir", type=Path, help="Directory containing episode_end-*.jsonl files")
    parser.add_argument("--matrix-csv", type=Path, default=None, help="Matrix CSV output path")
    parser.add_argument("--elo-csv", type=Path, default=None, help="Elo CSV output path")
    parser.add_argument("--md", type=Path, default=None, help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    matrix_rows, elo_rows = build_checkpoint_matrix(args.record_dir)
    matrix_csv = args.matrix_csv or args.record_dir / "checkpoint_matrix.csv"
    elo_csv = args.elo_csv or args.record_dir / "checkpoint_elo.csv"
    md_path = args.md or args.record_dir / "checkpoint_matrix.md"
    write_csv(
        matrix_rows,
        matrix_csv,
        ["checkpoint", "opponent", "games", "win_rate", "avg_frame", "avg_death", "avg_enemy_tower_hp"],
    )
    write_csv(elo_rows, elo_csv, ["player", "elo", "games"])
    write_markdown(matrix_rows, elo_rows, md_path, f"Checkpoint Matrix: {args.record_dir.as_posix()}")
    print(f"wrote {len(matrix_rows)} matrix rows and {len(elo_rows)} Elo rows to {matrix_csv}, {elo_csv}, {md_path}")


if __name__ == "__main__":
    main()
