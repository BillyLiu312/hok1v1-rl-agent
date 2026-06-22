#!/usr/bin/env python3
"""
Opponent curriculum helpers for PPO training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


DEFAULT_OPPONENT_SCHEDULE = {
    "common_ai": 4,
    "historical": 4,
    "selfplay": 2,
}


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


def apply_opponent_agent(usr_conf, opponent_agent):
    usr_conf.setdefault("episode", {})
    usr_conf["episode"]["opponent_agent"] = str(opponent_agent)
    return usr_conf
