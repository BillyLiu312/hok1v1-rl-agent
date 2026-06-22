#!/usr/bin/env python3
"""
Evaluate a v1.2 checkpoint candidate against the documented acceptance gates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


V1_1_BEST_WIN_RATE = 0.84
V1_1_BEST_ENEMY_TOWER_HP = 1401.0
V1_1_LATE_DEATH = 3.09
MIN_EXPECTED_MATCHUPS = 9
MIN_EPISODES_PER_MATCHUP = 20
MIN_MATCHUP_WIN_RATE_GAP = 0.25
MIN_PUSH_WINDOW_TOWER_DAMAGE_SHARE = 0.10
MAX_UNSAFE_DIVE_DEATH_CORR = 0.30
MAX_UNSAFE_DIVE_SEVERITY = 1.0
MAX_DEATH_P90 = 4.0
MIN_SELF_TOWER_HP_P10 = 1000.0
MAX_TIMEOUT_RATE = 0.15
DEFAULT_BASELINE = {
    "best_win_rate": V1_1_BEST_WIN_RATE,
    "best_win_enemy_tower_hp": V1_1_BEST_ENEMY_TOWER_HP,
    "late_death": V1_1_LATE_DEATH,
}


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_baseline(path: Path | None) -> dict:
    baseline = dict(DEFAULT_BASELINE)
    if not path:
        return baseline
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return baseline
    baseline["best_win_rate"] = to_float(data.get("best_win_rate"), baseline["best_win_rate"])
    baseline["best_win_enemy_tower_hp"] = to_float(
        data.get("best_win_enemy_tower_hp"),
        baseline["best_win_enemy_tower_hp"],
    )
    baseline["late_death"] = to_float(data.get("late_death"), baseline["late_death"])
    baseline["source"] = data.get("source_log_dir", path.as_posix())
    return baseline


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


def evaluate_candidate(candidate: dict, matchup_rows: list[dict] | None = None, baseline: dict | None = None) -> list[dict]:
    if not candidate:
        return [gate("MISSING", "candidate", "", "checkpoint exists", "No checkpoint candidate found.")]

    baseline = baseline or DEFAULT_BASELINE
    best_win_rate = to_float(baseline.get("best_win_rate"), V1_1_BEST_WIN_RATE)
    best_enemy_tower_hp = to_float(baseline.get("best_win_enemy_tower_hp"), V1_1_BEST_ENEMY_TOWER_HP)
    late_death = to_float(baseline.get("late_death"), V1_1_LATE_DEATH)
    gates = []
    win_rate = get_metric(candidate, "matchup_avg_win_rate", "common_ai_win_rate")
    death = get_metric(candidate, "matchup_avg_death", "common_ai_death")
    enemy_tower_hp = get_metric(candidate, "matchup_avg_enemy_tower_hp", "common_ai_enemy_tower_hp")
    matchup_groups = to_int(candidate.get("matchup_groups"), 0) or 0
    min_win_rate = to_float(candidate.get("matchup_min_win_rate"))
    push_window_tower_damage_share = to_float(candidate.get("matchup_avg_push_window_tower_damage_share"))
    unsafe_dive_death_corr = to_float(candidate.get("matchup_avg_unsafe_dive_death_corr"))
    unsafe_dive_severity = to_float(candidate.get("matchup_avg_unsafe_dive_severity"))
    death_p90 = to_float(candidate.get("matchup_max_death_p90"))
    self_tower_hp_p10 = to_float(candidate.get("matchup_min_self_tower_hp_p10"))
    timeout_rate = to_float(candidate.get("matchup_avg_timeout_rate"))

    if win_rate is None:
        gates.append(gate("MISSING", "common_ai_win_rate", "", f"> {best_win_rate}", "Win rate is unavailable."))
    elif win_rate > best_win_rate:
        gates.append(gate("PASS", "common_ai_win_rate", win_rate, f"> {best_win_rate}", "Beats v1.1 best win-rate baseline."))
    elif death is not None and death < late_death:
        gates.append(
            gate(
                "WARN",
                "common_ai_win_rate",
                win_rate,
                f"> {best_win_rate} or lower death",
                "Win rate does not beat v1.1 best, but death improved; keep as candidate only with matchup evidence.",
            )
        )
    else:
        gates.append(gate("FAIL", "common_ai_win_rate", win_rate, f"> {best_win_rate}", "Does not beat v1.1 best win-rate baseline."))

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
        gates.append(gate("MISSING", "enemy_tower_hp", "", f"< {best_enemy_tower_hp}", "Enemy tower HP is unavailable."))
    elif enemy_tower_hp < best_enemy_tower_hp:
        gates.append(gate("PASS", "enemy_tower_hp", enemy_tower_hp, f"< {best_enemy_tower_hp}", "Improves v1.1 tower pressure."))
    else:
        gates.append(gate("WARN", "enemy_tower_hp", enemy_tower_hp, f"< {best_enemy_tower_hp}", "Tower pressure does not beat v1.1 best."))

    if death is None:
        gates.append(gate("MISSING", "avg_death", "", f"< {late_death}", "Death metric is unavailable."))
    elif death < late_death:
        gates.append(gate("PASS", "avg_death", death, f"< {late_death}", "Death is below v1.1 late-training level."))
    else:
        gates.append(gate("FAIL", "avg_death", death, f"< {late_death}", "Death remains too high."))

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

    if death_p90 is None:
        gates.append(
            gate(
                "WARN",
                "death_tail_risk",
                "",
                f"<= {MAX_DEATH_P90}",
                "Missing p90 death evidence; cannot verify that deaths are controlled beyond the mean.",
            )
        )
    else:
        status = "PASS" if death_p90 <= MAX_DEATH_P90 else "WARN"
        gates.append(
            gate(
                status,
                "death_tail_risk",
                death_p90,
                f"<= {MAX_DEATH_P90}",
                "High-percentile deaths should stay bounded across matchups.",
            )
        )

    if self_tower_hp_p10 is None:
        gates.append(
            gate(
                "WARN",
                "self_tower_tail_risk",
                "",
                f">= {MIN_SELF_TOWER_HP_P10}",
                "Missing p10 self-tower evidence; cannot verify defensive stability.",
            )
        )
    else:
        status = "PASS" if self_tower_hp_p10 >= MIN_SELF_TOWER_HP_P10 else "WARN"
        gates.append(
            gate(
                status,
                "self_tower_tail_risk",
                self_tower_hp_p10,
                f">= {MIN_SELF_TOWER_HP_P10}",
                "Low-percentile self tower HP should not collapse in weak matchups.",
            )
        )

    if timeout_rate is None:
        gates.append(
            gate(
                "WARN",
                "timeout_rate",
                "",
                f"<= {MAX_TIMEOUT_RATE}",
                "Missing timeout evidence; cannot separate stable wins from slow stalling.",
            )
        )
    else:
        status = "PASS" if timeout_rate <= MAX_TIMEOUT_RATE else "WARN"
        gates.append(
            gate(
                status,
                "timeout_rate",
                timeout_rate,
                f"<= {MAX_TIMEOUT_RATE}",
                "Timeout rate should remain low so tower pressure is decisive.",
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

    if unsafe_dive_severity is None:
        gates.append(
            gate(
                "WARN",
                "unsafe_dive_severity",
                "",
                f"<= {MAX_UNSAFE_DIVE_SEVERITY}",
                "Missing unsafe-dive severity; keep layered risk diagnostics in the evidence package.",
            )
        )
    else:
        status = "PASS" if unsafe_dive_severity <= MAX_UNSAFE_DIVE_SEVERITY else "WARN"
        gates.append(
            gate(
                status,
                "unsafe_dive_severity",
                unsafe_dive_severity,
                f"<= {MAX_UNSAFE_DIVE_SEVERITY}",
                "Layered unsafe-dive severity should stay bounded across fixed matchups.",
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

        episode_counts = [
            to_float(row.get("episodes"), 0.0) or 0.0
            for row in matchup_rows
            if row.get("matchup")
        ]
        if not episode_counts:
            gates.append(
                gate(
                    "MISSING",
                    "matchup_min_episodes",
                    "",
                    f">= {MIN_EPISODES_PER_MATCHUP}",
                    "Raw matchup rows do not include episode counts.",
                )
            )
        else:
            min_episodes = min(episode_counts)
            if min_episodes >= MIN_EPISODES_PER_MATCHUP:
                gates.append(
                    gate(
                        "PASS",
                        "matchup_min_episodes",
                        min_episodes,
                        f">= {MIN_EPISODES_PER_MATCHUP}",
                        "Every matchup has enough repeated games for the fixed matrix gate.",
                    )
                )
            else:
                gates.append(
                    gate(
                        "WARN",
                        "matchup_min_episodes",
                        min_episodes,
                        f">= {MIN_EPISODES_PER_MATCHUP}",
                        "At least one matchup has too few episodes; treat matchup conclusions as provisional.",
                    )
                )

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
    parser.add_argument("--baseline-json", type=Path, default=None, help="Optional JSON produced by utils/v1_2_baseline.py")
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
    rows = evaluate_candidate(candidate, matchup_rows=matchup_rows, baseline=read_baseline(args.baseline_json))
    write_csv(rows, args.csv)
    write_markdown(rows, args.md, "v1.2 Candidate Gate", candidate)
    print(f"{overall_status(rows)} v1.2 candidate gate -> {args.csv} and {args.md}")


if __name__ == "__main__":
    main()
