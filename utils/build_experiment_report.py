#!/usr/bin/env python3
"""
Build a local experiment evidence package for v1.2 training runs.
"""

from __future__ import annotations

import argparse
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
from utils.select_checkpoint import attach_matchup_metrics, collect_candidates, rank_candidates
from utils.select_checkpoint import write_csv as write_checkpoint_csv
from utils.select_checkpoint import write_markdown as write_checkpoint_markdown
from utils.summoner_skill_grid import build_rows as build_skill_rows
from utils.summoner_skill_grid import write_csv as write_skill_csv
from utils.summoner_skill_grid import write_markdown as write_skill_markdown
from utils.summoner_skill_results import collect_rows as collect_skill_result_rows
from utils.summoner_skill_results import write_csv as write_skill_result_csv
from utils.summoner_skill_results import write_markdown as write_skill_result_markdown


def build_report(
    log_dir: Path,
    output_dir: Path,
    record_dir: Path | None = None,
    checkpoints=None,
    heroes=None,
    repeats=20,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    training_rows = collect_training_rows(log_dir)
    training_csv = output_dir / "training_summary.csv"
    training_md = output_dir / "training_summary.md"
    write_training_csv(training_rows, training_csv)
    write_training_markdown(training_rows, training_md, f"Training Summary: {log_dir.as_posix()}")
    artifacts["training_summary_csv"] = training_csv
    artifacts["training_summary_md"] = training_md

    matchup_csv = None
    if record_dir and record_dir.exists():
        run_rows = collect_run_rows(record_dir)
        matchup_csv = output_dir / "matchup_summary.csv"
        matchup_md = output_dir / "matchup_summary.md"
        write_run_csv(run_rows, matchup_csv)
        write_run_markdown(run_rows, matchup_md, f"Run Record Matchup Summary: {record_dir.as_posix()}")
        artifacts["matchup_summary_csv"] = matchup_csv
        artifacts["matchup_summary_md"] = matchup_md

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

    candidates = collect_candidates(training_csv=training_csv)
    attach_matchup_metrics(candidates, matchup_csv)
    checkpoint_rows = rank_candidates(candidates)
    checkpoint_csv = output_dir / "checkpoint_ranking.csv"
    checkpoint_md = output_dir / "checkpoint_ranking.md"
    write_checkpoint_csv(checkpoint_rows, checkpoint_csv)
    write_checkpoint_markdown(checkpoint_rows, checkpoint_md, "Checkpoint Ranking")
    artifacts["checkpoint_ranking_csv"] = checkpoint_csv
    artifacts["checkpoint_ranking_md"] = checkpoint_md

    eval_rows = build_eval_rows(checkpoints=checkpoints, hero_ids=heroes, repeats=repeats)
    eval_csv = output_dir / "evaluation_matrix.csv"
    eval_md = output_dir / "evaluation_matrix.md"
    write_eval_csv(eval_rows, eval_csv)
    write_eval_markdown(eval_rows, eval_md)
    artifacts["evaluation_matrix_csv"] = eval_csv
    artifacts["evaluation_matrix_md"] = eval_md

    eval_config_artifacts = export_configs(eval_csv, output_dir / "evaluation_configs")
    artifacts["evaluation_usr_conf_jsonl"] = eval_config_artifacts["jsonl"]
    artifacts["evaluation_config_manifest"] = eval_config_artifacts["manifest"]

    skill_rows = build_skill_rows(hero_ids=heroes)
    skill_csv = output_dir / "summoner_skill_grid.csv"
    skill_md = output_dir / "summoner_skill_grid.md"
    write_skill_csv(skill_rows, skill_csv)
    write_skill_markdown(skill_rows, skill_md)
    artifacts["summoner_skill_grid_csv"] = skill_csv
    artifacts["summoner_skill_grid_md"] = skill_md

    manifest = output_dir / "manifest.md"
    write_manifest(manifest, artifacts, training_rows, checkpoint_rows, eval_rows)
    artifacts["manifest"] = manifest
    return artifacts


def write_manifest(path: Path, artifacts: dict, training_rows: list[dict], checkpoint_rows: list[dict], eval_rows: list[dict]):
    lines = [
        "# v1.2 Experiment Evidence Package",
        "",
        f"- training_rows: {len(training_rows)}",
        f"- evaluation_rows: {len(eval_rows)}",
    ]
    if checkpoint_rows:
        best = checkpoint_rows[0]
        lines.append(f"- recommended_checkpoint: {best.get('checkpoint_step')}")
        lines.append(f"- checkpoint_score: {best.get('score'):.4g}")
    lines.extend(["", "## Artifacts", ""])
    for name, artifact_path in sorted(artifacts.items()):
        lines.append(f"- {name}: `{artifact_path.as_posix()}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build v1.2 local experiment report artifacts.")
    parser.add_argument("--log-dir", type=Path, required=True, help="Directory containing step-*.md training logs")
    parser.add_argument("--record-dir", type=Path, default=None, help="Directory containing training recorder JSONL files")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/v1.2/report"), help="Output directory")
    parser.add_argument("--checkpoints", default="15000,17057", help="Comma-separated checkpoint steps for eval matrix")
    parser.add_argument("--heroes", default="112,133,199", help="Comma-separated hero IDs")
    parser.add_argument("--repeats", type=int, default=20, help="Repeats per eval matrix group")
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts = build_report(
        log_dir=args.log_dir,
        record_dir=args.record_dir,
        output_dir=args.output_dir,
        checkpoints=parse_ids(args.checkpoints),
        heroes=parse_ids(args.heroes),
        repeats=args.repeats,
    )
    print(f"wrote v1.2 experiment package to {args.output_dir}")
    print(f"manifest: {artifacts['manifest']}")


if __name__ == "__main__":
    main()
