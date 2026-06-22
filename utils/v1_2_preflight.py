#!/usr/bin/env python3
"""
Run local preflight checks before launching v1.2 training.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_ppo.conf.conf import Config, GameConfig
from utils.offline_sync import preset_include_patterns, repo_root, v1_2_readiness


TRAIN_ENV_CONF = Path("agent_ppo/conf/train_env_conf.toml")
REQUIRED_TOOLS = [
    "utils/analyze_training_logs.py",
    "utils/analyze_run_records.py",
    "utils/build_experiment_report.py",
    "utils/checkpoint_matrix.py",
    "utils/compare_experiment_reports.py",
    "utils/evaluate_v1_2_candidate.py",
    "utils/evaluation_matrix.py",
    "utils/evaluation_config_export.py",
    "utils/run_metadata_summary.py",
    "utils/select_checkpoint.py",
    "utils/summoner_skill_grid.py",
    "utils/summoner_skill_results.py",
    "utils/offline_sync.py",
    "utils/v1_2_launch_manifest.py",
]


def parse_simple_toml_value(text: str, key: str):
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*[\"']?([^\"'\n#]+)", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def row(status: str, check: str, observed, expected: str, detail: str) -> dict:
    return {
        "status": status,
        "check": check,
        "observed": observed,
        "expected": expected,
        "detail": detail,
    }


def check_train_env_conf(path: Path = TRAIN_ENV_CONF) -> list[dict]:
    if not path.exists():
        return [row("FAIL", "train_env_conf_exists", path.as_posix(), "exists", "Training environment config is missing.")]

    text = path.read_text(encoding="utf-8")
    opponent_agent = parse_simple_toml_value(text, "opponent_agent")
    eval_opponent_type = parse_simple_toml_value(text, "eval_opponent_type")
    rows = []
    rows.append(
        row(
            "PASS" if opponent_agent == "common_ai" else "FAIL",
            "v1.2_a_opponent_agent",
            opponent_agent,
            "common_ai",
            "v1.2-a should isolate reward/feature changes before selfplay drift.",
        )
    )
    rows.append(
        row(
            "PASS" if eval_opponent_type == "common_ai" else "FAIL",
            "eval_opponent_type",
            eval_opponent_type,
            "common_ai",
            "Evaluation should stay anchored to common_ai for comparability.",
        )
    )
    rows.append(
        row(
            "PASS" if "v1.2-a" in text else "WARN",
            "v1.2_a_comment",
            "present" if "v1.2-a" in text else "missing",
            "v1.2-a marker",
            "The config should document why common_ai is the first training stage.",
        )
    )
    return rows


def check_ppo_config() -> list[dict]:
    return [
        row("PASS" if Config.INIT_LEARNING_RATE_START == 5e-4 else "FAIL", "learning_rate", Config.INIT_LEARNING_RATE_START, "5e-4", "v1.2-a uses conservative PPO stability settings."),
        row("PASS" if Config.GAMMA == 0.995 else "FAIL", "gamma", Config.GAMMA, "0.995", "Keep long-horizon tower objective discounting."),
        row("PASS" if Config.BETA_START == 0.025 else "FAIL", "entropy_beta", Config.BETA_START, "0.025", "Do not change entropy at the same time as reward/features."),
        row("PASS" if Config.USE_GRAD_CLIP and Config.GRAD_CLIP_RANGE == 0.5 else "FAIL", "grad_clip", Config.GRAD_CLIP_RANGE if Config.USE_GRAD_CLIP else "disabled", "enabled:0.5", "Keep gradient clipping stable."),
    ]


def check_reward_profile() -> list[dict]:
    required_weights = {
        "win_result": 20.0,
        "timeout_tower_gap": 8.0,
        "death": 4.0,
        "push_window_tower_damage": 2.0,
        "unsafe_dive": 2.0,
        "push_window_active": 0.0,
        "unsafe_dive_active": 0.0,
    }
    rows = [
        row(
            "PASS" if GameConfig.REWARD_PROFILE == os.environ.get("HOK_REWARD_PROFILE", "v1.2") else "WARN",
            "reward_profile",
            GameConfig.REWARD_PROFILE,
            os.environ.get("HOK_REWARD_PROFILE", "v1.2"),
            "Reward profile should be explicit in the training environment.",
        )
    ]
    for key, expected in required_weights.items():
        observed = GameConfig.REWARD_WEIGHT_DICT.get(key)
        rows.append(
            row(
                "PASS" if observed == expected else "WARN",
                f"reward_weight_{key}",
                observed,
                str(expected),
                "Different values are allowed for ablations but should be intentional.",
            )
        )
    return rows


def check_required_tools(root: Path) -> list[dict]:
    rows = []
    for path in REQUIRED_TOOLS:
        exists = (root / path).exists()
        rows.append(row("PASS" if exists else "FAIL", f"tool_exists:{path}", "exists" if exists else "missing", "exists", "Required v1.2 tool should be present."))
    return rows


def check_sync_preset(root: Path) -> list[dict]:
    readiness = v1_2_readiness(root, preset_include_patterns("v1.2"), [])
    return [
        row(
            "PASS" if readiness["ready"] else "FAIL",
            "offline_sync_v1.2_preset",
            f"{readiness['included_count']} files; missing={len(readiness['missing'])}",
            "ready=true",
            "Offline sync preset should include all v1.2 required files.",
        )
    ]


def collect_rows() -> list[dict]:
    root = repo_root()
    rows = []
    rows.extend(check_train_env_conf(root / TRAIN_ENV_CONF))
    rows.extend(check_ppo_config())
    rows.extend(check_reward_profile())
    rows.extend(check_required_tools(root))
    rows.extend(check_sync_preset(root))
    return rows


def overall_status(rows: list[dict]) -> str:
    statuses = {item["status"] for item in rows}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def write_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["status", "check", "observed", "expected", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["status", "check", "observed", "expected", "detail"]
    lines = [
        "# v1.2 Preflight",
        "",
        f"- overall_status: {overall_status(rows)}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for item in rows:
        lines.append("| " + " | ".join(str(item.get(column, "")) for column in columns) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run local v1.2 training readiness checks.")
    parser.add_argument("--csv", type=Path, default=None, help="CSV output path")
    parser.add_argument("--md", type=Path, default=None, help="Markdown output path")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_rows()
    if args.csv:
        write_csv(rows, args.csv)
    if args.md:
        write_markdown(rows, args.md)
    print(f"{overall_status(rows)} v1.2 preflight ({len(rows)} checks)")
    for item in rows:
        if item["status"] != "PASS":
            print(f"{item['status']} {item['check']}: observed={item['observed']} expected={item['expected']}")
    return 0 if overall_status(rows) != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
