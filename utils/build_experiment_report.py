#!/usr/bin/env python3
"""
Build a local experiment evidence package for v1.2 training runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.analyze_run_records import collect_rows as collect_run_rows
from utils.analyze_run_records import write_csv as write_run_csv
from utils.analyze_run_records import write_markdown as write_run_markdown
from utils.analyze_training_logs import collect_rows as collect_training_rows
from utils.analyze_training_logs import write_csv as write_training_csv
from utils.analyze_training_logs import write_markdown as write_training_markdown
from utils.checkpoint_matrix import build_checkpoint_matrix, write_markdown as write_checkpoint_matrix_markdown
from utils.checkpoint_matrix import write_csv as write_checkpoint_matrix_csv
from utils.evaluation_matrix import build_rows as build_eval_rows
from utils.evaluation_matrix import parse_ids
from utils.evaluation_matrix import write_csv as write_eval_csv
from utils.evaluation_matrix import write_markdown as write_eval_markdown
from utils.evaluation_config_export import export_configs
from utils.evaluate_v1_2_candidate import evaluate_candidate
from utils.evaluate_v1_2_candidate import overall_status as candidate_gate_status
from utils.evaluate_v1_2_candidate import write_csv as write_candidate_gate_csv
from utils.evaluate_v1_2_candidate import write_markdown as write_candidate_gate_markdown
from utils.select_checkpoint import attach_matchup_metrics, collect_candidates, rank_candidates
from utils.select_checkpoint import write_csv as write_checkpoint_csv
from utils.select_checkpoint import write_markdown as write_checkpoint_markdown
from utils.run_metadata_summary import collect_rows as collect_metadata_rows
from utils.run_metadata_summary import write_csv as write_metadata_csv
from utils.run_metadata_summary import write_markdown as write_metadata_markdown
from utils.summoner_skill_grid import build_rows as build_skill_rows
from utils.summoner_skill_grid import write_csv as write_skill_csv
from utils.summoner_skill_grid import write_markdown as write_skill_markdown
from utils.summoner_skill_results import collect_rows as collect_skill_result_rows
from utils.summoner_skill_results import recommend_skill_rows
from utils.summoner_skill_results import write_csv as write_skill_result_csv
from utils.summoner_skill_results import write_markdown as write_skill_result_markdown
from utils.summoner_skill_results import write_recommendation_csv as write_skill_recommendation_csv
from utils.summoner_skill_results import write_recommendation_markdown as write_skill_recommendation_markdown


DEFAULT_OUTPUT_DIR = Path("logs/v1.2/report-v1.2")


def build_report(
    log_dir: Path,
    output_dir: Path,
    record_dir: Path | None = None,
    launch_manifest: Path | None = None,
    experiment_plan: Path | None = None,
    experiment_name: str | None = None,
    checkpoints=None,
    heroes=None,
    repeats=20,
    include_skill_grid=False,
    skills=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    launch_metadata = read_launch_manifest(launch_manifest)
    experiment_metadata = resolve_experiment_metadata(read_json_file(experiment_plan), experiment_name)

    training_rows = collect_training_rows(log_dir)
    training_csv = output_dir / "training_summary.csv"
    training_md = output_dir / "training_summary.md"
    write_training_csv(training_rows, training_csv)
    write_training_markdown(training_rows, training_md, f"Training Summary: {log_dir.as_posix()}")
    artifacts["training_summary_csv"] = training_csv
    artifacts["training_summary_md"] = training_md

    matchup_csv = None
    run_rows = None
    if record_dir and record_dir.exists():
        run_rows = collect_run_rows(record_dir)
        matchup_csv = output_dir / "matchup_summary.csv"
        matchup_md = output_dir / "matchup_summary.md"
        write_run_csv(run_rows, matchup_csv)
        write_run_markdown(run_rows, matchup_md, f"Run Record Matchup Summary: {record_dir.as_posix()}")
        artifacts["matchup_summary_csv"] = matchup_csv
        artifacts["matchup_summary_md"] = matchup_md

        metadata_rows = collect_metadata_rows(record_dir)
        metadata_csv = output_dir / "run_metadata_summary.csv"
        metadata_md = output_dir / "run_metadata_summary.md"
        write_metadata_csv(metadata_rows, metadata_csv)
        write_metadata_markdown(metadata_rows, metadata_md, f"Run Metadata Summary: {record_dir.as_posix()}")
        artifacts["run_metadata_summary_csv"] = metadata_csv
        artifacts["run_metadata_summary_md"] = metadata_md

        checkpoint_matrix_rows, checkpoint_elo_rows = build_checkpoint_matrix(record_dir)
        checkpoint_matrix_csv = output_dir / "checkpoint_matrix.csv"
        checkpoint_elo_csv = output_dir / "checkpoint_elo.csv"
        checkpoint_matrix_md = output_dir / "checkpoint_matrix.md"
        write_checkpoint_matrix_csv(
            checkpoint_matrix_rows,
            checkpoint_matrix_csv,
            ["checkpoint", "opponent", "games", "win_rate", "avg_frame", "avg_death", "avg_enemy_tower_hp"],
        )
        write_checkpoint_matrix_csv(checkpoint_elo_rows, checkpoint_elo_csv, ["player", "elo", "games"])
        write_checkpoint_matrix_markdown(
            checkpoint_matrix_rows,
            checkpoint_elo_rows,
            checkpoint_matrix_md,
            f"Checkpoint Matrix: {record_dir.as_posix()}",
        )
        artifacts["checkpoint_matrix_csv"] = checkpoint_matrix_csv
        artifacts["checkpoint_elo_csv"] = checkpoint_elo_csv
        artifacts["checkpoint_matrix_md"] = checkpoint_matrix_md

        skill_result_rows = collect_skill_result_rows(record_dir)
        skill_result_csv = output_dir / "summoner_skill_results.csv"
        skill_result_md = output_dir / "summoner_skill_results.md"
        write_skill_result_csv(skill_result_rows, skill_result_csv)
        write_skill_result_markdown(
            skill_result_rows,
            skill_result_md,
            f"Summoner Skill Results: {record_dir.as_posix()}",
        )
        artifacts["summoner_skill_results_csv"] = skill_result_csv
        artifacts["summoner_skill_results_md"] = skill_result_md

        skill_recommendation_rows = recommend_skill_rows(skill_result_rows)
        skill_recommendation_csv = output_dir / "summoner_skill_recommendations.csv"
        skill_recommendation_md = output_dir / "summoner_skill_recommendations.md"
        write_skill_recommendation_csv(skill_recommendation_rows, skill_recommendation_csv)
        write_skill_recommendation_markdown(
            skill_recommendation_rows,
            skill_recommendation_md,
            f"Summoner Skill Recommendations: {record_dir.as_posix()}",
        )
        artifacts["summoner_skill_recommendations_csv"] = skill_recommendation_csv
        artifacts["summoner_skill_recommendations_md"] = skill_recommendation_md

    candidates = collect_candidates(training_csv=training_csv)
    attach_matchup_metrics(candidates, matchup_csv)
    checkpoint_rows = rank_candidates(candidates)
    checkpoint_csv = output_dir / "checkpoint_ranking.csv"
    checkpoint_md = output_dir / "checkpoint_ranking.md"
    write_checkpoint_csv(checkpoint_rows, checkpoint_csv)
    write_checkpoint_markdown(checkpoint_rows, checkpoint_md, "Checkpoint Ranking")
    artifacts["checkpoint_ranking_csv"] = checkpoint_csv
    artifacts["checkpoint_ranking_md"] = checkpoint_md

    candidate = checkpoint_rows[0] if checkpoint_rows else {}
    candidate_matchup_rows = filter_rows_for_checkpoint(run_rows, candidate)
    candidate_gate_rows = evaluate_candidate(candidate, matchup_rows=candidate_matchup_rows)
    candidate_gate_csv = output_dir / "v1.2_candidate_gate.csv"
    candidate_gate_md = output_dir / "v1.2_candidate_gate.md"
    write_candidate_gate_csv(candidate_gate_rows, candidate_gate_csv)
    write_candidate_gate_markdown(
        candidate_gate_rows,
        candidate_gate_md,
        "v1.2 Candidate Gate",
        candidate,
    )
    artifacts["v1.2_candidate_gate_csv"] = candidate_gate_csv
    artifacts["v1.2_candidate_gate_md"] = candidate_gate_md

    eval_rows = build_eval_rows(
        checkpoints=checkpoints,
        hero_ids=heroes,
        repeats=repeats,
        include_skill_grid=include_skill_grid,
        candidate_skills=skills,
    )
    eval_csv = output_dir / "evaluation_matrix.csv"
    eval_md = output_dir / "evaluation_matrix.md"
    write_eval_csv(eval_rows, eval_csv)
    write_eval_markdown(eval_rows, eval_md)
    artifacts["evaluation_matrix_csv"] = eval_csv
    artifacts["evaluation_matrix_md"] = eval_md

    eval_config_artifacts = export_configs(eval_csv, output_dir / "evaluation_configs")
    artifacts["evaluation_usr_conf_jsonl"] = eval_config_artifacts["jsonl"]
    artifacts["evaluation_config_manifest"] = eval_config_artifacts["manifest"]

    skill_rows = build_skill_rows(hero_ids=heroes, candidate_skills=skills)
    skill_csv = output_dir / "summoner_skill_grid.csv"
    skill_md = output_dir / "summoner_skill_grid.md"
    write_skill_csv(skill_rows, skill_csv)
    write_skill_markdown(skill_rows, skill_md)
    artifacts["summoner_skill_grid_csv"] = skill_csv
    artifacts["summoner_skill_grid_md"] = skill_md

    manifest = output_dir / "manifest.md"
    write_manifest(
        manifest,
        artifacts,
        training_rows,
        checkpoint_rows,
        candidate_gate_rows,
        eval_rows,
        launch_metadata,
        experiment_metadata,
    )
    artifacts["manifest"] = manifest
    return artifacts


def read_json_file(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_launch_manifest(path: Path | None) -> dict:
    return read_json_file(path)


def resolve_experiment_metadata(plan: dict, experiment_name: str | None = None) -> dict:
    if not plan:
        return {}
    ablations = plan.get("ablations") or []
    selected = {}
    if experiment_name:
        selected = next((item for item in ablations if item.get("name") == experiment_name), {})
    elif len(ablations) == 1:
        selected = ablations[0]
    return {
        "plan_stage": plan.get("stage", ""),
        "plan_research_question": (plan.get("story") or {}).get("research_question", ""),
        "plan_main_hypothesis": (plan.get("story") or {}).get("main_hypothesis", ""),
        "success_metrics": [
            item.get("metric", "")
            for item in plan.get("success_metrics", [])
            if item.get("metric")
        ],
        "experiment_name": selected.get("name", experiment_name or ""),
        "experiment_reward_profile": selected.get("reward_profile", ""),
        "experiment_hypothesis": selected.get("hypothesis", ""),
        "experiment_report_dir": selected.get("report_dir", ""),
    }


def filter_rows_for_checkpoint(rows: list[dict] | None, candidate: dict | None) -> list[dict] | None:
    if rows is None or not candidate:
        return rows
    aliases = {
        str(value)
        for value in (candidate.get("checkpoint_step"), candidate.get("actual_train_global_step"))
        if value not in ("", None)
    }
    if not aliases:
        return rows
    return [row for row in rows if str(row.get("checkpoint_step")) in aliases]


def count_status(rows: list[dict], status: str) -> int:
    return sum(1 for row in rows if row.get("status") == status)


def manifest_value(value):
    return "" if value is None else value


def write_manifest(
    path: Path,
    artifacts: dict,
    training_rows: list[dict],
    checkpoint_rows: list[dict],
    candidate_gate_rows: list[dict],
    eval_rows: list[dict],
    launch_metadata: dict | None = None,
    experiment_metadata: dict | None = None,
):
    checkpoint_count = len({row["checkpoint_step"] for row in eval_rows})
    matchup_count = len({row["matchup"] for row in eval_rows})
    skill_pairs = len({(row["blue_select_skill"], row["red_select_skill"]) for row in eval_rows})
    lines = [
        "# v1.2 Experiment Evidence Package",
        "",
        f"- training_rows: {len(training_rows)}",
        f"- evaluation_rows: {len(eval_rows)}",
        f"- evaluation_checkpoints: {checkpoint_count}",
        f"- evaluation_matchups: {matchup_count}",
        f"- evaluation_skill_pairs: {skill_pairs}",
        f"- candidate_gate_status: {candidate_gate_status(candidate_gate_rows)}",
        f"- candidate_gate_pass: {count_status(candidate_gate_rows, 'PASS')}",
        f"- candidate_gate_warn: {count_status(candidate_gate_rows, 'WARN')}",
        f"- candidate_gate_fail: {count_status(candidate_gate_rows, 'FAIL')}",
        f"- candidate_gate_missing: {count_status(candidate_gate_rows, 'MISSING')}",
    ]
    if launch_metadata:
        launch_env = launch_metadata.get("env") or {}
        lines.append(f"- launch_stage: {launch_metadata.get('stage', '')}")
        lines.append(f"- launch_run_id: {launch_metadata.get('run_id', '')}")
        lines.append(f"- launch_git_commit: {launch_metadata.get('git_commit', '')}")
        lines.append(f"- launch_preflight_status: {launch_metadata.get('preflight_status', '')}")
        lines.append(f"- launch_sync_package_sha256: {launch_metadata.get('sync_package_sha256', '')}")
        lines.append(f"- launch_training_record_dir: {launch_env.get('HOK_TRAINING_RECORD_DIR', '')}")
        lines.append(f"- launch_training_run_id: {launch_env.get('HOK_TRAINING_RUN_ID', '')}")
        lines.append(f"- launch_reward_profile: {launch_env.get('HOK_REWARD_PROFILE', '')}")
        lines.append(f"- launch_reward_weight_overrides: {launch_env.get('HOK_REWARD_WEIGHT_OVERRIDES', '')}")
        lines.append(f"- launch_opponent_schedule: {launch_env.get('HOK_OPPONENT_SCHEDULE', '')}")
    if experiment_metadata:
        lines.append(f"- experiment_plan_stage: {experiment_metadata.get('plan_stage', '')}")
        lines.append(f"- experiment_name: {experiment_metadata.get('experiment_name', '')}")
        lines.append(f"- experiment_reward_profile: {experiment_metadata.get('experiment_reward_profile', '')}")
        lines.append(f"- experiment_hypothesis: {experiment_metadata.get('experiment_hypothesis', '')}")
        lines.append(f"- experiment_research_question: {experiment_metadata.get('plan_research_question', '')}")
        lines.append(f"- experiment_main_hypothesis: {experiment_metadata.get('plan_main_hypothesis', '')}")
        success_metrics = experiment_metadata.get("success_metrics", [])
        lines.append(f"- experiment_success_metric_count: {len(success_metrics)}")
        lines.append(f"- experiment_success_metrics: {','.join(success_metrics)}")
    if checkpoint_rows:
        best = checkpoint_rows[0]
        lines.append(f"- recommended_checkpoint: {best.get('checkpoint_step')}")
        lines.append(f"- checkpoint_score: {best.get('score'):.4g}")
        lines.append(f"- recommended_matchup_groups: {manifest_value(best.get('matchup_groups'))}")
        lines.append(f"- recommended_matchup_rows: {manifest_value(best.get('matchup_rows'))}")
        win_rate = best.get("matchup_avg_win_rate") if best.get("matchup_avg_win_rate") is not None else best.get("common_ai_win_rate")
        death = best.get("matchup_avg_death") if best.get("matchup_avg_death") is not None else best.get("common_ai_death")
        lines.append(f"- recommended_win_rate: {manifest_value(win_rate)}")
        lines.append(f"- recommended_min_win_rate: {manifest_value(best.get('matchup_min_win_rate'))}")
        lines.append(f"- recommended_death: {manifest_value(death)}")
        lines.append(
            f"- recommended_push_window_tower_damage_share: {manifest_value(best.get('matchup_avg_push_window_tower_damage_share'))}"
        )
        lines.append(f"- recommended_unsafe_dive_death_corr: {manifest_value(best.get('matchup_avg_unsafe_dive_death_corr'))}")
    lines.extend(["", "## Artifacts", ""])
    for name, artifact_path in sorted(artifacts.items()):
        lines.append(f"- {name}: `{artifact_path.as_posix()}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build v1.2 local experiment report artifacts.")
    parser.add_argument("--log-dir", type=Path, required=True, help="Directory containing step-*.md training logs")
    parser.add_argument("--record-dir", type=Path, default=None, help="Directory containing training recorder JSONL files")
    parser.add_argument("--launch-manifest", type=Path, default=Path("logs/v1.2/launch_manifest.json"), help="JSON from utils/v1_2_launch_manifest.py")
    parser.add_argument("--experiment-plan", type=Path, default=Path("logs/v1.2/experiment_plan.json"), help="JSON from utils/v1_2_experiment_plan.py")
    parser.add_argument("--experiment-name", default=None, help="Ablation name in the experiment plan")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--checkpoints", default="15000,17057", help="Comma-separated checkpoint steps for eval matrix")
    parser.add_argument("--heroes", default="112,133,199", help="Comma-separated hero IDs")
    parser.add_argument("--repeats", type=int, default=20, help="Repeats per eval matrix group")
    parser.add_argument("--skill-grid", action="store_true", help="Expand all candidate summoner skills in evaluation_matrix")
    parser.add_argument("--skills", default="80107,80110,80115", help="Comma-separated summoner skill IDs for skill grids")
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts = build_report(
        log_dir=args.log_dir,
        record_dir=args.record_dir,
        launch_manifest=args.launch_manifest,
        experiment_plan=args.experiment_plan,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        checkpoints=parse_ids(args.checkpoints),
        heroes=parse_ids(args.heroes),
        repeats=args.repeats,
        include_skill_grid=args.skill_grid,
        skills=parse_ids(args.skills),
    )
    print(f"wrote v1.2 experiment package to {args.output_dir}")
    print(f"manifest: {artifacts['manifest']}")


if __name__ == "__main__":
    main()
