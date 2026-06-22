#!/usr/bin/env python3
"""
Evaluate a v1.2 checkpoint candidate against the documented acceptance gates.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


V1_1_BEST_WIN_RATE = 0.84
V1_1_BEST_ENEMY_TOWER_HP = 1401.0
V1_1_LATE_DEATH = 3.09
MIN_EXPECTED_MATCHUPS = 9
MIN_MATCHUP_WIN_RATE_GAP = 0.25
MIN_PUSH_WINDOW_TOWER_DAMAGE_SHARE = 0.10
MAX_UNSAFE_DIVE_DEATH_CORR = 0.30


def read_csv_rows(path: Path) -> list[dict]:
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
    number = to_float(value)
    if number is None:
        return default
    return int(number)


def fmt(value):
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def find_candidate(ranking_rows: list[dict], checkpoint_step=None) -> dict:
    if not ranking_rows:
        return {}
    if checkpoint_step is None:
        return ranking_rows[0]
    for row in ranking_rows:
        if str(row.get("checkpoint_step")) == str(checkpoint_step):
            return row
    return {}


def build_matchup_index(matchup_rows: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for row in matchup_rows:
        step = str(row.get("checkpoint_step"))
        grouped.setdefault(step, []).append(row)
    return grouped


def get_metric(candidate: dict, primary_key: str, fallback_key: str | None = None):
    value = to_float(candidate.get(primary_key))
    if value is None and fallback_key:
        value = to_float(candidate.get(fallback_key))
    return value


def gate(status: str, name: str, observed, threshold: str, detail: str) -> dict:
    return {
        "status": status,
        "gate": name,
        "observed": observed,
        "threshold": threshold,
        "detail": detail,
    }


def evaluate_candidate(candidate: dict, matchup_rows: list[dict] | None = None) -> list[dict]:
    if not candidate:
        return [gate("MISSING", "candidate", "", "checkpoint exists", "No checkpoint candidate found.")]

    gates = []
    win_rate = get_metric(candidate, "matchup_avg_win_rate", "common_ai_win_rate")
    death = get_metric(candidate, "matchup_avg_death", "common_ai_death")
    enemy_tower_hp = get_metric(candidate, "matchup_avg_enemy_tower_hp", "common_ai_enemy_tower_hp")
    matchup_groups = to_int(candidate.get("matchup_groups"), 0) or 0
    min_win_rate = to_float(candidate.get("matchup_min_win_rate"))
    push_window_tower_damage_share = to_float(candidate.get("matchup_avg_push_window_tower_damage_share"))
    unsafe_dive_death_corr = to_float(candidate.get("matchup_avg_unsafe_dive_death_corr"))

    if win_rate is None:
        gates.append(gate("MISSING", "common_ai_win_rate", "", f"> {V1_1_BEST_WIN_RATE}", "Win rate is unavailable."))
    elif win_rate > V1_1_BEST_WIN_RATE:
        gates.append(gate("PASS", "common_ai_win_rate", win_rate, f"> {V1_1_BEST_WIN_RATE}", "Beats v1.1 step-15000 win rate."))
    elif death is not None and death < V1_1_LATE_DEATH:
        gates.append(
            gate(
                "WARN",
                "common_ai_win_rate",
                win_rate,
                f"> {V1_1_BEST_WIN_RATE} or lower death",
                "Win rate does not beat v1.1 best, but death improved; keep as candidate only with matchup evidence.",
            )
        )
    else:
        gates.append(gate("FAIL", "common_ai_win_rate", win_rate, f"> {V1_1_BEST_WIN_RATE}", "Does not beat v1.1 best win rate."))

    if matchup_groups >= MIN_EXPECTED_MATCHUPS:
        gates.append(gate("PASS", "matchup_coverage", matchup_groups, f">= {MIN_EXPECTED_MATCHUPS}", "All 9 hero matchups are covered."))
    elif matchup_groups > 0:
        gates.append(gate("FAIL", "matchup_coverage", matchup_groups, f">= {MIN_EXPECTED_MATCHUPS}", "Matrix evaluation is incomplete."))
    else:
        gates.append(gate("MISSING", "matchup_coverage", matchup_groups, f">= {MIN_EXPECTED_MATCHUPS}", "No matchup matrix evidence."))

    if min_win_rate is None or win_rate is None:
        gates.append(gate("MISSING", "worst_matchup_gap", "", f"<= {MIN_MATCHUP_WIN_RATE_GAP}", "Missing min or average matchup win rate."))
    else:
        gap = win_rate - min_win_rate
        status = "PASS" if gap <= MIN_MATCHUP_WIN_RATE_GAP else "FAIL"
        gates.append(
            gate(
                status,
                "worst_matchup_gap",
                gap,
                f"<= {MIN_MATCHUP_WIN_RATE_GAP}",
                "Worst matchup should not fall far behind the average.",
            )
        )

    if enemy_tower_hp is None:
        gates.append(gate("MISSING", "enemy_tower_hp", "", f"< {V1_1_BEST_ENEMY_TOWER_HP}", "Enemy tower HP is unavailable."))
    elif enemy_tower_hp < V1_1_BEST_ENEMY_TOWER_HP:
        gates.append(gate("PASS", "enemy_tower_hp", enemy_tower_hp, f"< {V1_1_BEST_ENEMY_TOWER_HP}", "Improves v1.1 tower pressure."))
    else:
        gates.append(gate("WARN", "enemy_tower_hp", enemy_tower_hp, f"< {V1_1_BEST_ENEMY_TOWER_HP}", "Tower pressure does not beat v1.1 best."))

    if death is None:
        gates.append(gate("MISSING", "avg_death", "", f"< {V1_1_LATE_DEATH}", "Death metric is unavailable."))
    elif death < V1_1_LATE_DEATH:
        gates.append(gate("PASS", "avg_death", death, f"< {V1_1_LATE_DEATH}", "Death is below v1.1 late-training level."))
    else:
        gates.append(gate("FAIL", "avg_death", death, f"< {V1_1_LATE_DEATH}", "Death remains too high."))

    if push_window_tower_damage_share is None:
        gates.append(
            gate(
                "MISSING",
                "push_window_evidence",
                "",
                f">= {MIN_PUSH_WINDOW_TOWER_DAMAGE_SHARE}",
                "Missing push-window tower damage share; cannot verify the v1.2 tactical-window hypothesis.",
            )
        )
    elif push_window_tower_damage_share >= MIN_PUSH_WINDOW_TOWER_DAMAGE_SHARE:
        gates.append(
            gate(
                "PASS",
                "push_window_evidence",
                push_window_tower_damage_share,
                f">= {MIN_PUSH_WINDOW_TOWER_DAMAGE_SHARE}",
                "A measurable share of tower damage happens inside detected push windows.",
            )
        )
    else:
        gates.append(
            gate(
                "WARN",
                "push_window_evidence",
                push_window_tower_damage_share,
                f">= {MIN_PUSH_WINDOW_TOWER_DAMAGE_SHARE}",
                "Push-window tower damage share is low; inspect whether the window detector or policy is too conservative.",
            )
        )

    if unsafe_dive_death_corr is None:
        gates.append(
            gate(
                "WARN",
                "unsafe_dive_risk",
                "",
                f"<= {MAX_UNSAFE_DIVE_DEATH_CORR}",
                "Missing unsafe-dive/death correlation; keep death and dive diagnostics in the evidence package.",
            )
        )
    else:
        status = "PASS" if unsafe_dive_death_corr <= MAX_UNSAFE_DIVE_DEATH_CORR else "WARN"
        gates.append(
            gate(
                status,
                "unsafe_dive_risk",
                unsafe_dive_death_corr,
                f"<= {MAX_UNSAFE_DIVE_DEATH_CORR}",
                "Unsafe-dive frames should not strongly correlate with deaths.",
            )
        )

    checkpoint_step = candidate.get("checkpoint_step")
    if checkpoint_step not in ("", None):
        gates.append(gate("PASS", "checkpoint_selection", checkpoint_step, "explicit checkpoint", "Candidate checkpoint is explicit."))
    else:
        gates.append(gate("FAIL", "checkpoint_selection", "", "explicit checkpoint", "No explicit checkpoint selected."))

    if matchup_rows is not None and matchup_groups:
        actual_matchups = len({row.get("matchup") for row in matchup_rows if row.get("matchup")})
        if actual_matchups >= MIN_EXPECTED_MATCHUPS:
            gates.append(gate("PASS", "raw_matchup_rows", actual_matchups, f">= {MIN_EXPECTED_MATCHUPS}", "Raw matchup table covers all matchup names."))
        else:
            gates.append(gate("WARN", "raw_matchup_rows", actual_matchups, f">= {MIN_EXPECTED_MATCHUPS}", "Aggregated matchup count and raw matchup names differ."))

    return gates


def overall_status(gates: list[dict]) -> str:
    statuses = {row["status"] for row in gates}
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


def write_markdown(rows: list[dict], output_path: Path, title: str, candidate: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["status", "gate", "observed", "threshold", "detail"]
    lines = [
        f"# {title}",
        "",
        f"- overall_status: {overall_status(rows)}",
        f"- checkpoint_step: {candidate.get('checkpoint_step', '') if candidate else ''}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a v1.2 checkpoint candidate against acceptance gates.")
    parser.add_argument("--ranking-csv", type=Path, required=True, help="CSV produced by utils/select_checkpoint.py")
    parser.add_argument("--matchup-csv", type=Path, default=None, help="CSV produced by utils/analyze_run_records.py")
    parser.add_argument("--checkpoint-step", default=None, help="Checkpoint step to evaluate; defaults to top ranked row")
    parser.add_argument("--csv", type=Path, default=Path("logs/v1.2_candidate_gate.csv"), help="CSV output path")
    parser.add_argument("--md", type=Path, default=Path("logs/v1.2_candidate_gate.md"), help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    ranking_rows = read_csv_rows(args.ranking_csv)
    candidate = find_candidate(ranking_rows, checkpoint_step=args.checkpoint_step)
    matchup_rows = None
    if args.matchup_csv:
        all_matchup_rows = read_csv_rows(args.matchup_csv)
        matchup_index = build_matchup_index(all_matchup_rows)
        if candidate:
            matchup_rows = matchup_index.get(str(candidate.get("checkpoint_step")), [])
        else:
            matchup_rows = []
    rows = evaluate_candidate(candidate, matchup_rows=matchup_rows)
    write_csv(rows, args.csv)
    write_markdown(rows, args.md, "v1.2 Candidate Gate", candidate)
    print(f"{overall_status(rows)} v1.2 candidate gate -> {args.csv} and {args.md}")


if __name__ == "__main__":
    main()
