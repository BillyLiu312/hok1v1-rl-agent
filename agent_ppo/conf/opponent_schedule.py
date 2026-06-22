#!/usr/bin/env python3
"""
Opponent curriculum helpers for PPO training.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path


DEFAULT_OPPONENT_SCHEDULE = {
    "common_ai": 4,
    "historical": 4,
    "selfplay": 2,
}


def parse_schedule(raw_schedule: str | None):
    if not raw_schedule:
        return None

    schedule = {}
    for item in raw_schedule.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            continue
        opponent_type, weight = item.split(":", 1)
        opponent_type = opponent_type.strip()
        if not opponent_type:
            continue
        try:
            schedule[opponent_type] = float(weight.strip())
        except (TypeError, ValueError):
            continue
    return schedule or None


def load_opponent_schedule(env_var="HOK_OPPONENT_SCHEDULE"):
    return parse_schedule(os.environ.get(env_var))


def load_model_pool(path="kaiwu.json"):
    model_pool_path = Path(path)
    if not model_pool_path.exists():
        return []
    data = json.loads(model_pool_path.read_text(encoding="utf-8"))
    return [str(model_id) for model_id in data.get("model_pool", [])]


def normalize_schedule(raw_schedule=None):
    schedule = raw_schedule or DEFAULT_OPPONENT_SCHEDULE
    normalized = {}
    for opponent_type, weight in schedule.items():
        try:
            weight_value = float(weight)
        except (TypeError, ValueError):
            continue
        if weight_value > 0:
            normalized[str(opponent_type)] = weight_value
    return normalized


def select_curriculum_opponent(model_pool=None, schedule=None, rng=None):
    model_pool = [str(model_id) for model_id in (model_pool or [])]
    schedule = normalize_schedule(schedule)
    rng = rng or random

    candidates = []
    weights = []
    for opponent_type, weight in schedule.items():
        if opponent_type == "historical":
            if not model_pool:
                continue
            candidates.extend(model_pool)
            weights.extend([weight / len(model_pool)] * len(model_pool))
        else:
            candidates.append(opponent_type)
            weights.append(weight)

    if not candidates:
        return "selfplay"
    return rng.choices(candidates, weights=weights, k=1)[0]


def classify_opponent_source(opponent_agent, model_pool=None):
    opponent_agent = str(opponent_agent)
    if opponent_agent in ("common_ai", "selfplay"):
        return opponent_agent
    model_pool = {str(model_id) for model_id in (model_pool or [])}
    if opponent_agent in model_pool:
        return "historical"
    return "custom"


def apply_opponent_agent(usr_conf, opponent_agent):
    usr_conf.setdefault("episode", {})
    usr_conf["episode"]["opponent_agent"] = str(opponent_agent)
    return usr_conf
