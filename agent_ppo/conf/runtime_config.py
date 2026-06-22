#!/usr/bin/env python3
"""
Runtime defaults for web-launched training jobs.

Kaiwu web launches do not necessarily inherit shell environment variables, so
v1.2 keeps its training defaults in a synced file. Environment variables still
win when they are available, which keeps one-off local experiments easy.
"""

from __future__ import annotations

import configparser
import os
from functools import lru_cache
from pathlib import Path


RUNTIME_CONFIG_PATH = Path(__file__).with_name("runtime_config.ini")

DEFAULTS = {
    "training": {
        "recorder": "1",
        "record_dir": "logs/run_records/v1.2-a",
        "run_id": "",
    },
    "reward": {
        "profile": "v1.2",
        "weight_overrides": "",
    },
    "opponent": {
        "schedule": "",
    },
}

ENV_BINDINGS = {
    "HOK_TRAINING_RECORDER": ("training", "recorder"),
    "HOK_TRAINING_RECORD_DIR": ("training", "record_dir"),
    "HOK_TRAINING_RUN_ID": ("training", "run_id"),
    "HOK_REWARD_PROFILE": ("reward", "profile"),
    "HOK_REWARD_WEIGHT_OVERRIDES": ("reward", "weight_overrides"),
    "HOK_OPPONENT_SCHEDULE": ("opponent", "schedule"),
}


@lru_cache(maxsize=1)
def load_runtime_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_dict(DEFAULTS)
    if RUNTIME_CONFIG_PATH.exists():
        config.read(RUNTIME_CONFIG_PATH, encoding="utf-8")
    return config


def runtime_value(env_var: str, default: str = "") -> str:
    env_value = os.environ.get(env_var)
    if env_value is not None:
        return env_value

    section, key = ENV_BINDINGS.get(env_var, ("", ""))
    if not section:
        return default

    value = load_runtime_config().get(section, key, fallback=default)
    if value == "" and default:
        return default
    return value


def runtime_snapshot() -> dict[str, str]:
    return {env_var: runtime_value(env_var) for env_var in ENV_BINDINGS}
