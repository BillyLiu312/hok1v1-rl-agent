#!/usr/bin/env python3
"""
Run local preflight checks before launching v1.2 training.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_ppo.conf.conf import Config, GameConfig
from agent_ppo.conf.runtime_config import runtime_value
from agent_ppo.feature.feature_process import V1_2_REQUIRED_FEATURES, feature_schema
from utils.offline_sync import preset_include_patterns, repo_root, v1_2_readiness
from utils.v1_2_experiment_plan import build_manifest as build_experiment_plan_manifest


TRAIN_ENV_CONF = Path("agent_ppo/conf/train_env_conf.toml")
REQUIRED_TOOLS = [
    "utils/analyze_training_logs.py",
    "utils/analyze_run_records.py",
    "utils/build_experiment_report.py",
    "utils/checkpoint_matrix.py",
    "utils/compare_experiment_reports.py",
    "utils/evaluate_summoner_skill_policy.py",
    "utils/evaluate_v1_2_candidate.py",
    "utils/evaluation_matrix.py",
    "utils/evaluation_config_export.py",
    "utils/run_metadata_summary.py",
    "utils/opponent_curriculum_summary.py",
    "utils/select_checkpoint.py",
    "utils/summoner_skill_grid.py",
    "utils/summoner_skill_policy_patch.py",
    "utils/summoner_skill_results.py",
    "utils/offline_sync.py",
    "utils/v1_2_baseline.py",
    "utils/v1_2_experiment_plan.py",
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
        "unsafe_dive_severity": 1.0,
        "push_window_active": 0.0,
        "unsafe_dive_active": 0.0,
    }
    rows = [
        row(
            "PASS" if GameConfig.REWARD_PROFILE == runtime_value("HOK_REWARD_PROFILE", "v1.2") else "WARN",
            "reward_profile",
            GameConfig.REWARD_PROFILE,
            runtime_value("HOK_REWARD_PROFILE", "v1.2"),
            "Reward profile should be explicit in runtime_config.ini or the training environment.",
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


def check_feature_schema() -> list[dict]:
    schema = feature_schema()
    missing = [name for name in V1_2_REQUIRED_FEATURES if name not in schema]
    return [
        row(
            "PASS" if len(schema) == Config.FEATURE_DIM else "FAIL",
            "feature_schema_length",
            len(schema),
            str(Config.FEATURE_DIM),
            "Feature schema should document every model input dimension.",
        ),
        row(
            "PASS" if len(schema) == len(set(schema)) else "FAIL",
            "feature_schema_unique",
            len(schema) - len(set(schema)),
            "0 duplicates",
            "Feature names should be unique so evidence can map metrics to inputs.",
        ),
        row(
            "PASS" if not missing else "FAIL",
            "feature_schema_v1.2_required",
            ",".join(missing) if missing else "present",
            "all required v1.2 tactical features",
            "Push-window, unsafe-dive, economy, damage and revive-time inputs should be auditable before training.",
        ),
    ]


def check_required_tools(root: Path) -> list[dict]:
    rows = []
    for path in REQUIRED_TOOLS:
        exists = (root / path).exists()
        rows.append(row("PASS" if exists else "FAIL", f"tool_exists:{path}", "exists" if exists else "missing", "exists", "Required v1.2 tool should be present."))
    return rows


def check_evidence_chain_fields(root: Path) -> list[dict]:
    required_fields = {
        "utils/build_experiment_report.py": [
            "candidate_gate_matchup_filter",
            "filter_fixed_eval_rows_for_checkpoint",
            "read_candidate_baseline",
            "baseline_best_hero_damage_balance",
            "summoner_skill_policy_patch",
            "summoner_skill_policy_gate",
            "recommended_death_p90",
            "opponent_curriculum_summary",
            "resolved_reward_weight_dict_sha",
            "research_story_summary",
            "build_research_story_rows",
            "candidate_damage_balance",
            "matchup_avg_hurt_to_hero",
            "evaluation_coverage",
            "build_evaluation_coverage_rows",
        ],
        "utils/evaluate_summoner_skill_policy.py": [
            "current_policy_coverage",
            "policy_win_delta",
            "death_regression",
        ],
        "utils/opponent_curriculum_summary.py": [
            "opponent_source",
            "configured_opponent_agent",
        ],
        "utils/compare_experiment_reports.py": [
            "research_story_verdict",
            "candidate_gate_matchup_filter",
            "matchup_filter_opponent_agent",
            "baseline_source",
            "baseline_best_hero_damage_balance",
            "reward_weight_dict_sha",
            "matchup_eval_ids",
            "max_death_p90",
            "hero_damage_balance",
        ],
        "utils/select_checkpoint.py": [
            "matchup_filter_eval_only",
            "matchup_filter_opponent_agent",
            "matchup_eval_ids",
            "matchup_repeat_indices",
            "matchup_max_death_p90",
            "matchup_avg_hurt_to_hero",
            "matchup_avg_hurt_by_hero",
        ],
        "utils/evaluate_v1_2_candidate.py": [
            "read_baseline",
            "matchup_min_episodes",
            "hero_damage_balance",
            "death_tail_risk",
            "timeout_rate",
            "unsafe_dive_severity",
        ],
        "utils/v1_2_baseline.py": [
            "best_win_rate",
            "best_win_enemy_tower_hp",
            "best_win_hero_damage_balance",
            "late_death",
        ],
        "agent_ppo/workflow/train_workflow.py": [
            "_extract_evaluation_metadata",
            '"evaluation": self._extract_evaluation_metadata(usr_conf)',
        ],
        "utils/evaluation_config_export.py": [
            '"evaluation": build_eval_metadata(row)',
            "write_toml_metadata",
            "toml_metadata.csv",
        ],
        "utils/analyze_run_records.py": [
            "eval_ids",
            "repeat_indices",
            "evaluation_checkpoint_step",
            "avg_hurt_to_hero",
            "avg_hurt_by_hero",
        ],
        "utils/analyze_training_logs.py": [
            "common_ai_hurt_to_hero",
            "common_ai_hurt_by_hero",
            "selfplay_hurt_to_hero",
            "selfplay_hurt_by_hero",
        ],
        "utils/run_metadata_summary.py": [
            "reward_weight_dict",
            "reward_weight_dict_sha",
            "canonical_reward_weight_dict",
        ],
    }
    rows = []
    for path, fields in required_fields.items():
        full_path = root / path
        text = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
        missing = [field for field in fields if field not in text]
        rows.append(
            row(
                "PASS" if not missing else "FAIL",
                f"evidence_fields:{path}",
                "present" if not missing else ",".join(missing),
                ",".join(fields),
                "v1.2 evidence reports should preserve filtering, gate, story and skill-policy fields.",
            )
        )
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


def check_experiment_plan() -> list[dict]:
    manifest = build_experiment_plan_manifest()
    ablation_names = [item.get("name") for item in manifest.get("ablations", [])]
    reward_profiles = [item.get("reward_profile") for item in manifest.get("ablations", [])]
    matrix = manifest.get("matrix", {})
    commands = [item.get("report_command", "") for item in manifest.get("ablations", [])]
    success_metrics = [item.get("metric") for item in manifest.get("success_metrics", [])]
    expected_ablations = ["v1.2", "no_window_reward", "no_terminal_reward", "death_only_risk"]
    rows = [
        row(
            "PASS" if ablation_names == expected_ablations else "FAIL",
            "experiment_plan_ablations",
            ",".join(ablation_names),
            ",".join(expected_ablations),
            "Default v1.2 experiment plan should keep the planned reward ablations.",
        ),
        row(
            "PASS" if reward_profiles == expected_ablations else "FAIL",
            "experiment_plan_reward_profiles",
            ",".join(reward_profiles),
            ",".join(expected_ablations),
            "Each ablation should map cleanly to one reward profile.",
        ),
        row(
            "PASS" if matrix.get("matchups") == 9 else "FAIL",
            "experiment_plan_matchups",
            matrix.get("matchups"),
            "9",
            "The default fixed matrix should cover the full 3x3 hero matchup set.",
        ),
        row(
            "PASS" if matrix.get("skill_pairs") == 9 else "FAIL",
            "experiment_plan_skill_pairs",
            matrix.get("skill_pairs"),
            "9",
            "The default skill grid should evaluate all 3x3 candidate skill pairs.",
        ),
        row(
            "PASS" if len(success_metrics) >= 7 else "FAIL",
            "experiment_plan_success_metrics",
            len(success_metrics),
            ">=7",
            "The plan should preserve the v1.2 acceptance and research-story metrics.",
        ),
        row(
            "PASS" if all("--experiment-plan" in command and "--experiment-name" in command for command in commands) else "FAIL",
            "experiment_plan_report_bindings",
            "present" if all("--experiment-plan" in command and "--experiment-name" in command for command in commands) else "missing",
            "--experiment-plan and --experiment-name",
            "Generated report commands should bind each report to the experiment plan.",
        ),
        row(
            "PASS" if all("--baseline-json logs/v1.2/baseline_v1.1.json" in command for command in commands) else "FAIL",
            "experiment_plan_report_baseline",
            "present" if all("--baseline-json logs/v1.2/baseline_v1.1.json" in command for command in commands) else "missing",
            "--baseline-json logs/v1.2/baseline_v1.1.json",
            "Generated report commands should bind every ablation gate to the v1.1 baseline file.",
        ),
    ]
    return rows


def check_launch_manifest_commands(root: Path) -> list[dict]:
    from utils.v1_2_launch_manifest import build_commands

    commands = build_commands("v1.2-a")
    report_command = commands["report"]
    baseline_command = commands["baseline"]
    experiment_plan_command = commands["experiment_plan"]
    return [
        row(
            "PASS" if "utils/v1_2_baseline.py" in baseline_command else "FAIL",
            "launch_manifest_baseline_command",
            "present" if "utils/v1_2_baseline.py" in baseline_command else "missing",
            "utils/v1_2_baseline.py",
            "Launch manifest should tell the operator how to derive v1.1 baseline gates.",
        ),
        row(
            "PASS" if "utils/v1_2_experiment_plan.py" in experiment_plan_command else "FAIL",
            "launch_manifest_experiment_plan_command",
            "present" if "utils/v1_2_experiment_plan.py" in experiment_plan_command else "missing",
            "utils/v1_2_experiment_plan.py",
            "Launch manifest should tell the operator how to generate the experiment plan.",
        ),
        row(
            "PASS" if "--experiment-plan" in report_command and "--experiment-name" in report_command else "FAIL",
            "launch_manifest_report_binding",
            "present" if "--experiment-plan" in report_command and "--experiment-name" in report_command else "missing",
            "--experiment-plan and --experiment-name",
            "Launch manifest report command should bind reports to the v1.2 experiment plan.",
        ),
        row(
            "PASS" if "--baseline-json logs/v1.2/baseline_v1.1.json" in report_command else "FAIL",
            "launch_manifest_report_baseline",
            "logs/v1.2/baseline_v1.1.json" if "--baseline-json logs/v1.2/baseline_v1.1.json" in report_command else "missing",
            "--baseline-json logs/v1.2/baseline_v1.1.json",
            "Report command should bind candidate gates to the generated v1.1 baseline.",
        ),
        row(
            "PASS" if "--record-dir logs/run_records/v1.2-a" in report_command else "FAIL",
            "launch_manifest_report_record_dir",
            "logs/run_records/v1.2-a" if "--record-dir logs/run_records/v1.2-a" in report_command else "missing",
            "logs/run_records/v1.2-a",
            "The v1.2-a launch report command should read the v1.2-a recorder directory.",
        ),
        row(
            "PASS" if "--output-dir logs/v1.2/report-v1.2" in report_command else "FAIL",
            "launch_manifest_report_output_dir",
            "logs/v1.2/report-v1.2" if "--output-dir logs/v1.2/report-v1.2" in report_command else "missing",
            "logs/v1.2/report-v1.2",
            "The v1.2-a launch report command should write the canonical v1.2 report directory.",
        ),
    ]


def collect_rows() -> list[dict]:
    root = repo_root()
    rows = []
    rows.extend(check_train_env_conf(root / TRAIN_ENV_CONF))
    rows.extend(check_ppo_config())
    rows.extend(check_reward_profile())
    rows.extend(check_feature_schema())
    rows.extend(check_experiment_plan())
    rows.extend(check_launch_manifest_commands(root))
    rows.extend(check_required_tools(root))
    rows.extend(check_evidence_chain_fields(root))
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
