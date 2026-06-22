#!/usr/bin/env python3
"""
Generate a reproducible v1.2 experiment plan for ablations and reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_HEROES = [112, 133, 199]
DEFAULT_CHECKPOINTS = [15000, 17057]
DEFAULT_SKILLS = [80107, 80110, 80115]

STORY = {
    "research_question": "How can a PPO 1v1 MOBA agent learn stable tower-taking wins from sparse long-horizon objectives?",
    "main_hypothesis": (
        "Explicit push-window features and rewards improve tower pressure while unsafe-dive diagnostics keep the policy "
        "from converting extra aggression into deaths."
    ),
    "comparison_principle": (
        "Compare v1.2 against v1.1 step-15000, then run reward ablations with the same matchup matrix, checkpoint "
        "selection rule, and candidate gates."
    ),
}

ABLATIONS = [
    {
        "name": "v1.2",
        "reward_profile": "v1.2",
        "hypothesis": "Full terminal, tower-pressure, and tactical-window rewards should beat or stabilize the v1.1 peak.",
        "record_dir": "logs/run_records/v1.2-a",
        "report_dir": "logs/v1.2/report-v1.2",
    },
    {
        "name": "no_window_reward",
        "reward_profile": "no_window_reward",
        "hypothesis": "Removing tactical-window reward tests whether gains come from window modeling rather than terminal reward alone.",
        "record_dir": "logs/run_records/v1.2-no-window",
        "report_dir": "logs/v1.2/report-no-window",
    },
    {
        "name": "no_terminal_reward",
        "reward_profile": "no_terminal_reward",
        "hypothesis": "Removing terminal win/tower-gap reward tests whether dense tower and risk shaping are sufficient.",
        "record_dir": "logs/run_records/v1.2-no-terminal",
        "report_dir": "logs/v1.2/report-no-terminal",
    },
    {
        "name": "death_only_risk",
        "reward_profile": "death_only_risk",
        "hypothesis": "A death-risk-only variant checks whether stronger risk aversion explains lower deaths without tactical windows.",
        "record_dir": "logs/run_records/v1.2-death-risk",
        "report_dir": "logs/v1.2/report-death-risk",
    },
]

SUCCESS_METRICS = [
    {
        "metric": "avg_win_rate",
        "target": "> 0.84 or comparable with lower death",
        "reason": "v1.1 step-15000 reached 0.84 common_ai win rate.",
    },
    {
        "metric": "matchup_coverage",
        "target": "9 hero matchups",
        "reason": "The v1.2 claim is stability across the full 3x3 hero matrix.",
    },
    {
        "metric": "min_win_rate_gap",
        "target": "<= 0.25 below average",
        "reason": "Worst-matchup behavior should not hide behind the mean.",
    },
    {
        "metric": "avg_enemy_tower_hp",
        "target": "< 1401 or shorter win frames",
        "reason": "Tower pressure must improve beyond the v1.1 selected checkpoint.",
    },
    {
        "metric": "avg_death",
        "target": "< 3.09",
        "reason": "v1.1 late checkpoints showed death inflation.",
    },
    {
        "metric": "push_window_tower_damage_share",
        "target": ">= 0.10",
        "reason": "The tactical-window story needs measurable tower damage inside detected windows.",
    },
    {
        "metric": "unsafe_dive_death_corr",
        "target": "<= 0.30 or explicitly inspected",
        "reason": "Unsafe-dive diagnostics should not strongly explain deaths.",
    },
]


def csv_list(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def build_report_command(group: dict, checkpoints: list[int], heroes: list[int], repeats: int, skills: list[int]) -> str:
    return (
        "python3 utils/build_experiment_report.py "
        "--log-dir logs/v1.2 "
        f"--record-dir {group['record_dir']} "
        "--launch-manifest logs/v1.2/launch_manifest.json "
        "--experiment-plan logs/v1.2/experiment_plan.json "
        f"--experiment-name {group['name']} "
        f"--output-dir {group['report_dir']} "
        f"--checkpoints {csv_list(checkpoints)} "
        f"--heroes {csv_list(heroes)} "
        f"--repeats {repeats} "
        "--skill-grid "
        f"--skills {csv_list(skills)}"
    )


def build_manifest(
    stage: str = "v1.2-a",
    checkpoints: list[int] | None = None,
    heroes: list[int] | None = None,
    repeats: int = 20,
    skills: list[int] | None = None,
) -> dict:
    checkpoints = checkpoints or DEFAULT_CHECKPOINTS
    heroes = heroes or DEFAULT_HEROES
    skills = skills or DEFAULT_SKILLS
    ablations = []
    for group in ABLATIONS:
        item = dict(group)
        item["env"] = {
            "HOK_TRAINING_RECORDER": "1",
            "HOK_TRAINING_RECORD_DIR": group["record_dir"],
            "HOK_TRAINING_RUN_ID": f"{stage}-{group['name']}",
            "HOK_REWARD_PROFILE": group["reward_profile"],
        }
        item["report_command"] = build_report_command(group, checkpoints, heroes, repeats, skills)
        ablations.append(item)

    return {
        "stage": stage,
        "story": dict(STORY),
        "matrix": {
            "checkpoints": checkpoints,
            "heroes": heroes,
            "matchups": len(heroes) * len(heroes),
            "side_swaps": 2,
            "repeats": repeats,
            "skills": skills,
            "skill_pairs": len(skills) * len(skills),
        },
        "ablations": ablations,
        "success_metrics": list(SUCCESS_METRICS),
        "comparison_command": build_comparison_command(ablations),
    }


def build_comparison_command(ablations: list[dict]) -> str:
    report_dirs = " ".join(group["report_dir"] for group in ablations)
    return (
        "python3 utils/compare_experiment_reports.py "
        f"{report_dirs} "
        "--csv logs/v1.2/report_comparison.csv "
        "--md logs/v1.2/report_comparison.md"
    )


def write_json(manifest: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_list(values: list[int]) -> str:
    return ", ".join(str(value) for value in values)


def write_markdown(manifest: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    story = manifest["story"]
    matrix = manifest["matrix"]
    lines = [
        "# v1.2 Experiment Plan",
        "",
        f"- stage: {manifest['stage']}",
        f"- research_question: {story['research_question']}",
        f"- main_hypothesis: {story['main_hypothesis']}",
        f"- comparison_principle: {story['comparison_principle']}",
        "",
        "## Fixed Matrix",
        "",
        f"- checkpoints: {format_list(matrix['checkpoints'])}",
        f"- heroes: {format_list(matrix['heroes'])}",
        f"- matchups: {matrix['matchups']}",
        f"- side_swaps: {matrix['side_swaps']}",
        f"- repeats: {matrix['repeats']}",
        f"- skills: {format_list(matrix['skills'])}",
        f"- skill_pairs: {matrix['skill_pairs']}",
        "",
        "## Ablations",
        "",
        "| name | reward_profile | training_run_id | hypothesis | report_dir |",
        "| --- | --- | --- | --- | --- |",
    ]
    for group in manifest["ablations"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    group["name"],
                    group["reward_profile"],
                    group["env"]["HOK_TRAINING_RUN_ID"],
                    group["hypothesis"],
                    group["report_dir"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Success Metrics",
            "",
            "| metric | target | reason |",
            "| --- | --- | --- |",
        ]
    )
    for metric in manifest["success_metrics"]:
        lines.append("| " + " | ".join([metric["metric"], metric["target"], metric["reason"]]) + " |")
    lines.extend(["", "## Commands", ""])
    for group in manifest["ablations"]:
        lines.extend([f"### {group['name']}", "", "```bash"])
        for key, value in group["env"].items():
            lines.append(f"export {key}={value}")
        lines.append(group["report_command"])
        lines.extend(["```", ""])
    lines.extend(["### Compare Reports", "", "```bash", manifest["comparison_command"], "```", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_ids(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a v1.2 experiment plan manifest.")
    parser.add_argument("--stage", default="v1.2-a", help="Experiment stage label")
    parser.add_argument("--checkpoints", default=csv_list(DEFAULT_CHECKPOINTS), help="Comma-separated checkpoint steps")
    parser.add_argument("--heroes", default=csv_list(DEFAULT_HEROES), help="Comma-separated hero IDs")
    parser.add_argument("--repeats", type=int, default=20, help="Repeats per checkpoint/matchup/side group")
    parser.add_argument("--skills", default=csv_list(DEFAULT_SKILLS), help="Comma-separated summoner skill IDs")
    parser.add_argument("--json", type=Path, default=Path("logs/v1.2/experiment_plan.json"), help="JSON output path")
    parser.add_argument("--md", type=Path, default=Path("logs/v1.2/experiment_plan.md"), help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(
        stage=args.stage,
        checkpoints=parse_ids(args.checkpoints),
        heroes=parse_ids(args.heroes),
        repeats=args.repeats,
        skills=parse_ids(args.skills),
    )
    write_json(manifest, args.json)
    write_markdown(manifest, args.md)
    print(f"wrote v1.2 experiment plan to {args.json} and {args.md}")


if __name__ == "__main__":
    main()
