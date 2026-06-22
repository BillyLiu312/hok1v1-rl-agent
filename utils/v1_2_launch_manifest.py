#!/usr/bin/env python3
"""
Generate a launch manifest for a v1.2 training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.v1_2_preflight import collect_rows as collect_preflight_rows
from utils.v1_2_preflight import overall_status as preflight_status
from utils.v1_2_experiment_plan import ABLATIONS


DEFAULT_ENV = {
    "HOK_TRAINING_RECORDER": "1",
    "HOK_TRAINING_RECORD_DIR": "logs/run_records/v1.2-a",
    "HOK_TRAINING_RUN_ID": "v1.2-a-001",
    "HOK_REWARD_PROFILE": "v1.2",
    "HOK_REWARD_WEIGHT_OVERRIDES": "",
    "HOK_OPPONENT_SCHEDULE": "",
}


def file_sha256(path: Path):
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def ablation_for_profile(reward_profile: str) -> dict:
    for ablation in ABLATIONS:
        if ablation.get("reward_profile") == reward_profile:
            return ablation
    return ABLATIONS[0]


def build_commands(stage: str, reward_profile: str = "v1.2") -> dict:
    ablation = ablation_for_profile(reward_profile)
    experiment_name = ablation["name"]
    record_dir = "logs/run_records/v1.2-b" if stage == "v1.2-b" and reward_profile == "v1.2" else ablation["record_dir"]
    output_dir = "logs/v1.2/report-v1.2-b" if stage == "v1.2-b" and reward_profile == "v1.2" else ablation["report_dir"]
    return {
        "preflight": "python3 utils/v1_2_preflight.py --md logs/v1.2/preflight.md --csv logs/v1.2/preflight.csv",
        "pack": "python3 utils/offline_sync.py pack --preset v1.2 --note v1.2-a-ready -o sync_package.txt",
        "experiment_plan": f"python3 utils/v1_2_experiment_plan.py --stage {stage} --json logs/v1.2/experiment_plan.json --md logs/v1.2/experiment_plan.md",
        "report": f"python3 utils/build_experiment_report.py --log-dir logs/v1.2 --record-dir {record_dir} --launch-manifest logs/v1.2/launch_manifest.json --experiment-plan logs/v1.2/experiment_plan.json --experiment-name {experiment_name} --output-dir {output_dir} --checkpoints 15000,17057 --heroes 112,133,199 --repeats 20",
    }


def build_manifest(
    sync_package: Path,
    run_id: str,
    stage: str,
    reward_profile: str,
    reward_weight_overrides: str = "",
    opponent_schedule: str | None = None,
) -> dict:
    preflight_rows = collect_preflight_rows()
    env = dict(DEFAULT_ENV)
    env["HOK_TRAINING_RUN_ID"] = run_id
    env["HOK_REWARD_PROFILE"] = reward_profile
    env["HOK_REWARD_WEIGHT_OVERRIDES"] = reward_weight_overrides
    ablation = ablation_for_profile(reward_profile)
    env["HOK_TRAINING_RECORD_DIR"] = ablation["record_dir"]
    if stage == "v1.2-b" and reward_profile == "v1.2":
        env["HOK_TRAINING_RECORD_DIR"] = "logs/run_records/v1.2-b"
        env["HOK_OPPONENT_SCHEDULE"] = "common_ai:4,historical:4,selfplay:2"
    if opponent_schedule is not None:
        env["HOK_OPPONENT_SCHEDULE"] = opponent_schedule

    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "stage": stage,
        "run_id": run_id,
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "sync_package": sync_package.as_posix(),
        "sync_package_exists": sync_package.exists(),
        "sync_package_sha256": file_sha256(sync_package),
        "sync_package_bytes": sync_package.stat().st_size if sync_package.exists() else 0,
        "preflight_status": preflight_status(preflight_rows),
        "preflight_checks": len(preflight_rows),
        "env": env,
        "commands": build_commands(stage, reward_profile),
    }


def write_json(manifest: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(manifest: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# v1.2 Launch Manifest",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- stage: {manifest['stage']}",
        f"- run_id: {manifest['run_id']}",
        f"- git_branch: {manifest['git_branch']}",
        f"- git_commit: {manifest['git_commit']}",
        f"- sync_package: `{manifest['sync_package']}`",
        f"- sync_package_bytes: {manifest['sync_package_bytes']}",
        f"- sync_package_sha256: `{manifest['sync_package_sha256']}`",
        f"- preflight_status: {manifest['preflight_status']}",
        f"- preflight_checks: {manifest['preflight_checks']}",
        "",
        "## Environment",
        "",
    ]
    for key, value in manifest["env"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Commands", ""])
    for key, value in manifest["commands"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a v1.2 launch manifest.")
    parser.add_argument("--sync-package", type=Path, default=Path("sync_package.txt"), help="Offline sync package path")
    parser.add_argument("--run-id", default="v1.2-a-001", help="Training run ID")
    parser.add_argument("--stage", choices=["v1.2-a", "v1.2-b"], default="v1.2-a", help="Training stage")
    parser.add_argument("--reward-profile", default="v1.2", help="Reward profile for this launch")
    parser.add_argument("--reward-weight-overrides", default="", help="Value for HOK_REWARD_WEIGHT_OVERRIDES")
    parser.add_argument("--opponent-schedule", default=None, help="Value for HOK_OPPONENT_SCHEDULE; defaults to stage preset")
    parser.add_argument("--json", type=Path, default=Path("logs/v1.2/launch_manifest.json"), help="JSON output path")
    parser.add_argument("--md", type=Path, default=Path("logs/v1.2/launch_manifest.md"), help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(
        args.sync_package,
        args.run_id,
        args.stage,
        args.reward_profile,
        reward_weight_overrides=args.reward_weight_overrides,
        opponent_schedule=args.opponent_schedule,
    )
    write_json(manifest, args.json)
    write_markdown(manifest, args.md)
    print(f"wrote v1.2 launch manifest to {args.json} and {args.md}")
    return 0 if manifest["preflight_status"] != "FAIL" and manifest["sync_package_exists"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
