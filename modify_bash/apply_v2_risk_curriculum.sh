#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

python3 - <<'PY'
import os
from pathlib import Path


root = Path(os.environ["REPO_ROOT"])


def read(rel_path):
    return (root / rel_path).read_text(encoding="utf-8")


def write(rel_path, text):
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[write] {rel_path}")


def replace_once(rel_path, old, new, marker=None):
    path = root / rel_path
    text = path.read_text(encoding="utf-8")
    if marker and marker in text:
        print(f"[skip] {rel_path}: {marker}")
        return
    if new in text:
        print(f"[skip] {rel_path}: already applied")
        return
    if old not in text:
        raise RuntimeError(f"Cannot find expected text in {rel_path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[patch] {rel_path}")


def delete_once(rel_path, old, marker):
    path = root / rel_path
    text = path.read_text(encoding="utf-8")
    if old not in text:
        print(f"[skip] {rel_path}: {marker}")
        return
    path.write_text(text.replace(old, "", 1), encoding="utf-8")
    print(f"[delete] {rel_path}: {marker}")


def insert_before(rel_path, needle, insert_text, marker):
    path = root / rel_path
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"[skip] {rel_path}: {marker}")
        return
    if needle not in text:
        raise RuntimeError(f"Cannot find insertion point in {rel_path}")
    path.write_text(text.replace(needle, insert_text + needle, 1), encoding="utf-8")
    print(f"[insert] {rel_path}: {marker}")


def insert_after(rel_path, needle, insert_text, marker):
    path = root / rel_path
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"[skip] {rel_path}: {marker}")
        return
    if needle not in text:
        raise RuntimeError(f"Cannot find insertion point in {rel_path}")
    path.write_text(text.replace(needle, needle + insert_text, 1), encoding="utf-8")
    print(f"[insert] {rel_path}: {marker}")


def ensure_line_after_each(rel_path, anchor, line_to_insert, marker):
    path = root / rel_path
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = False
    output = []
    inserted_count = 0

    for index, line in enumerate(lines):
        output.append(line)
        if anchor not in line:
            continue

        lookahead = "".join(lines[index + 1 : index + 4])
        if marker in lookahead:
            continue

        indent = line[: len(line) - len(line.lstrip())]
        output.append(f"{indent}{line_to_insert}\n")
        changed = True
        inserted_count += 1

    if changed:
        path.write_text("".join(output), encoding="utf-8")
        print(f"[insert] {rel_path}: {marker} x{inserted_count}")
    else:
        print(f"[skip] {rel_path}: {marker}")


def patch_conf():
    rel = "agent_ppo/conf/conf.py"
    replace_once(rel, '"damage_to_enemy": 2.0,', '"damage_to_enemy": 1.0,')
    replace_once(rel, '"kill": 8.0,', '"kill": 5.0,')
    replace_once(rel, '"death": -8.0,', '"death": -10.0,')
    insert_after(
        rel,
        '        "under_enemy_tower": -0.02,\n',
        '        "risk_exposure": -2.0,\n',
        '"risk_exposure": -2.0',
    )
    insert_after(
        rel,
        "    MODEL_SAVE_INTERVAL = 1800\n",
        """

    # Risk-sensitive shaping and opponent curriculum.
    # 风险敏感奖励塑形与对手课程配置。
    RISK_REWARD_WEIGHT = -2.0
    COMMON_AI_RATIO_START = 0.7
    COMMON_AI_RATIO_END = 0.2
    COMMON_AI_WARMUP_EPISODES = 500
    COMMON_AI_DECAY_EPISODES = 3000
    RISK_LOW_HP_THRESHOLD = 0.35
    ENEMY_TOWER_DANGER_RADIUS = 6500
    ENEMY_NEARBY_RADIUS = 4500
""",
        "COMMON_AI_RATIO_START",
    )
    replace_once(rel, "    DIM_OF_FEATURE = [58]\n", "    DIM_OF_FEATURE = [50]\n")


def patch_agent_load_model():
    rel = "agent_ppo/agent.py"
    replace_once(
        rel,
        '''            self.model.load_state_dict(
                torch.load(
                    model_file_path,
                    map_location=self.device,
                )
            )
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")
''',
        '''            checkpoint_state = torch.load(
                model_file_path,
                map_location=self.device,
            )
            if isinstance(checkpoint_state, dict):
                for state_key in ("state_dict", "model_state_dict", "model"):
                    if state_key in checkpoint_state and isinstance(checkpoint_state[state_key], dict):
                        checkpoint_state = checkpoint_state[state_key]
                        break

            model_state = self.model.state_dict()
            compatible_state = {}
            skipped_keys = []
            for key, value in checkpoint_state.items():
                if key not in model_state:
                    skipped_keys.append(key)
                    continue
                if model_state[key].shape != value.shape:
                    skipped_keys.append(key)
                    continue
                compatible_state[key] = value

            model_state.update(compatible_state)
            self.model.load_state_dict(model_state)
            self.cur_model_name = model_file_path
            if skipped_keys:
                self.logger.info(
                    f"load model {model_file_path} with {len(skipped_keys)} incompatible keys skipped: "
                    f"{skipped_keys[:5]}"
                )
            self.logger.info(f"load model {model_file_path} successfully")
''',
        marker="compatible_state",
    )


def patch_feature_process():
    rel = "agent_ppo/feature/feature_process/__init__.py"
    replace_once(
        rel,
        "from agent_ppo.conf.conf import Config\n",
        "from agent_ppo.conf.conf import Config, GameConfig\n",
    )
    insert_after(
        rel,
        "        self.organ_process = OrganProcess(camp)\n",
        "        self.last_state_summary = None\n",
        "last_state_summary",
    )
    insert_after(
        rel,
        "    def reset(self, camp):\n        self.camp = camp\n        self.hero_process = HeroProcess(camp)\n        self.organ_process = OrganProcess(camp)\n",
        "        self.last_state_summary = None\n",
        "self.last_state_summary = None\n\n    def process_organ_feature",
    )
    delete_once(
        rel,
        "        state_summary = self._state_summary(main_hero, enemy_hero, main_tower, enemy_tower)\n",
        "state_summary assignment",
    )
    delete_once(
        rel,
        "        feature.extend(self._temporal_feature(state_summary))\n        self.last_state_summary = state_summary\n",
        "temporal feature append",
    )
    insert_before(
        rel,
        "    def _hp_rate(self, unit):\n",
        r'''    def _state_summary(self, main_hero, enemy_hero, main_tower, enemy_tower):
        main_location = main_hero.get("location", {})
        enemy_location = enemy_hero.get("location", {})
        main_tower_location = main_tower.get("location", {})
        enemy_tower_location = enemy_tower.get("location", {})
        return {
            "main_hp": self._hp_rate(main_hero),
            "enemy_hp": self._hp_rate(enemy_hero),
            "main_tower_hp": self._hp_rate(main_tower),
            "enemy_tower_hp": self._hp_rate(enemy_tower),
            "enemy_dist": self._norm_distance(main_location, enemy_location),
            "enemy_tower_dist": self._norm_distance(main_location, enemy_tower_location),
            "main_tower_dist": self._norm_distance(main_location, main_tower_location),
            "risk_exposure": self._risk_exposure(main_hero, enemy_hero, enemy_tower),
        }

    def _temporal_feature(self, state_summary):
        if self.last_state_summary is None:
            return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, state_summary["risk_exposure"]]

        return [
            self._centered_delta(state_summary["main_hp"] - self.last_state_summary["main_hp"]),
            self._centered_delta(state_summary["enemy_hp"] - self.last_state_summary["enemy_hp"]),
            self._centered_delta(state_summary["main_tower_hp"] - self.last_state_summary["main_tower_hp"]),
            self._centered_delta(state_summary["enemy_tower_hp"] - self.last_state_summary["enemy_tower_hp"]),
            self._centered_delta(state_summary["enemy_dist"] - self.last_state_summary["enemy_dist"]),
            self._centered_delta(state_summary["enemy_tower_dist"] - self.last_state_summary["enemy_tower_dist"]),
            self._centered_delta(state_summary["main_tower_dist"] - self.last_state_summary["main_tower_dist"]),
            self.last_state_summary["risk_exposure"],
        ]

    def _risk_exposure(self, main_hero, enemy_hero, enemy_tower):
        main_location = main_hero.get("location", {})
        enemy_location = enemy_hero.get("location", {})
        enemy_tower_location = enemy_tower.get("location", {})
        hp_rate = self._hp_rate(main_hero)
        low_hp_factor = _clip((GameConfig.RISK_LOW_HP_THRESHOLD - hp_rate) / GameConfig.RISK_LOW_HP_THRESHOLD)
        tower_danger = 1.0 if self._distance(main_location, enemy_tower_location) < GameConfig.ENEMY_TOWER_DANGER_RADIUS else 0.0
        enemy_nearby = 1.0 if self._distance(main_location, enemy_location) < GameConfig.ENEMY_NEARBY_RADIUS else 0.0
        return _clip(low_hp_factor * (0.7 * tower_danger + 0.3 * enemy_nearby))

''',
        "def _state_summary",
    )
    insert_after(
        rel,
        "    def _norm_distance(self, pos1, pos2):\n        return _clip(self._distance(pos1, pos2) / 120000.0)\n",
        "\n    def _centered_delta(self, value):\n        return _clip(0.5 + value * 0.5)\n",
        "def _centered_delta",
    )


def patch_reward_process():
    rel = "agent_ppo/feature/reward_process.py"
    replace_once(
        rel,
        "from agent_ppo.conf.conf import GameConfig\n",
        "from agent_ppo.conf.conf import Config, GameConfig\n",
    )
    insert_after(
        rel,
        "def safe_div(numerator, denominator, default=0.0):\n    if denominator in (0, None):\n        return default\n    return numerator / denominator\n",
        "\n\ndef clip(value, min_value=0.0, max_value=1.0):\n    return max(min_value, min(max_value, value))\n",
        "def clip(",
    )
    insert_after(
        rel,
        "        self.time_scale_arg = GameConfig.TIME_SCALE_ARG\n",
        "        self.gamma = Config.GAMMA\n",
        "self.gamma = Config.GAMMA",
    )
    replace_once(
        rel,
        "                danger = 1.0 if dist_to_enemy_tower < 6500 else 0.0\n",
        "                danger = 1.0 if dist_to_enemy_tower < GameConfig.ENEMY_TOWER_DANGER_RADIUS else 0.0\n",
    )
    insert_after(
        rel,
        '                reward_struct.cur_frame_value = danger * (1.0 + (1.0 - hp_rate))\n',
        '            # Explicit risk exposure penalty for low HP near enemy tower or enemy hero.\n'
        '            # 低血时靠近敌塔或敌方英雄的显式风险暴露惩罚。\n'
        '            elif reward_name == "risk_exposure":\n'
        '                reward_struct.cur_frame_value = self.calculate_risk_exposure(main_hero, enemy_hero, enemy_tower)\n',
        'reward_name == "risk_exposure"',
    )
    insert_before(
        rel,
        "    # Calculate the reward item information for both sides using frame data\n",
        r'''    def calculate_risk_exposure(self, main_hero, enemy_hero, enemy_tower):
        hp_rate = safe_div(main_hero.get("hp", 0), main_hero.get("max_hp", 1))
        low_hp_factor = clip((GameConfig.RISK_LOW_HP_THRESHOLD - hp_rate) / GameConfig.RISK_LOW_HP_THRESHOLD)
        main_location = main_hero.get("location", {})
        tower_danger = 1.0 if distance(main_location, enemy_tower.get("location", {})) < GameConfig.ENEMY_TOWER_DANGER_RADIUS else 0.0
        enemy_nearby = 1.0 if distance(main_location, enemy_hero.get("location", {})) < GameConfig.ENEMY_NEARBY_RADIUS else 0.0
        return clip(low_hp_factor * (0.7 * tower_danger + 0.3 * enemy_nearby))

''',
        "def calculate_risk_exposure",
    )
    insert_after(
        rel,
        '            elif reward_name == "damage_to_enemy":\n',
        '                risk_gate = 1.0 - 0.7 * self.m_main_calc_frame_map["risk_exposure"].cur_frame_value\n',
        "risk_gate = 1.0",
    )
    replace_once(
        rel,
        "                )\n            elif reward_name == \"damage_taken\":\n",
        "                ) * risk_gate\n            elif reward_name == \"damage_taken\":\n",
    )
    replace_once(
        rel,
        '            elif reward_name in ("retreat_low_hp", "under_enemy_tower"):\n                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value\n',
        '            elif reward_name in ("retreat_low_hp", "under_enemy_tower"):\n'
        '                reward_struct.value = self._potential_diff(self.m_main_calc_frame_map[reward_name])\n'
        '            elif reward_name == "risk_exposure":\n'
        '                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value\n',
    )
    insert_after(
        rel,
        '        reward_dict["reward_sum"] = reward_sum\n',
        "\n    def _potential_diff(self, reward_struct):\n        return self.gamma * reward_struct.cur_frame_value - reward_struct.last_frame_value\n",
        "def _potential_diff",
    )


def patch_workflow():
    rel = "agent_ppo/workflow/train_workflow.py"
    insert_after(
        rel,
        "            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)\n",
        "            episode_opponent_agent = self._select_episode_opponent(is_eval)\n",
        "episode_opponent_agent",
    )
    replace_once(
        rel,
        "            self.reset_agents(observation)\n",
        "            self.reset_agents(observation, episode_opponent_agent)\n",
    )
    insert_after(
        rel,
        "            reward_detail_sum_list = [dict() for _ in range(self.agent_num)]\n",
        "            behavior_stats_list = [self._new_behavior_stats() for _ in range(self.agent_num)]\n",
        "behavior_stats_list",
    )
    replace_once(
        rel,
        '            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")\n',
        '            self.logger.info(\n'
        '                f"Episode {self.episode_cnt} start, opponent={episode_opponent_agent}, usr_conf is {usr_conf}"\n'
        '            )\n',
    )
    ensure_line_after_each(
        rel,
        "self._accumulate_reward_detail(reward_detail_sum_list[i], reward)",
        'self._accumulate_behavior_stats(behavior_stats_list[i], observation[str(i)], reward)',
        "_accumulate_behavior_stats(behavior_stats_list[i], observation[str(i)], reward)",
    )
    insert_after(
        rel,
        "                            for reward_name, reward_value in reward_detail_sum_list[monitor_side].items():\n                                monitor_data[f\"reward_{reward_name}\"] = round(reward_value, 4)\n",
        "                            monitor_data.update(self._format_behavior_stats(behavior_stats_list[monitor_side]))\n",
        "_format_behavior_stats",
    )
    insert_before(
        rel,
        "    def reset_agents(self, observation):\n",
        r'''    def _select_episode_opponent(self, is_eval):
        configured_opponent = self.env_conf_manager.get_opponent_agent()
        if is_eval or configured_opponent not in ("selfplay", "common_ai"):
            return configured_opponent

        common_ai_ratio = self._common_ai_ratio(self.episode_cnt)
        return "common_ai" if random.random() < common_ai_ratio else "selfplay"

    def _common_ai_ratio(self, episode_cnt):
        if episode_cnt < GameConfig.COMMON_AI_WARMUP_EPISODES:
            return GameConfig.COMMON_AI_RATIO_START

        decay_episode = episode_cnt - GameConfig.COMMON_AI_WARMUP_EPISODES
        decay_progress = min(decay_episode / max(GameConfig.COMMON_AI_DECAY_EPISODES, 1), 1.0)
        return (
            GameConfig.COMMON_AI_RATIO_START
            + (GameConfig.COMMON_AI_RATIO_END - GameConfig.COMMON_AI_RATIO_START) * decay_progress
        )

    def _new_behavior_stats(self):
        return {
            "frame_count": 0,
            "low_hp_enemy_tower_frames": 0,
            "risk_exposure_sum": 0.0,
            "tower_death_count": 0,
        }

    def _accumulate_behavior_stats(self, stats, observation, reward):
        frame_state = observation.get("frame_state", {})
        camp = observation.get("camp")
        main_hero, enemy_tower = self._select_main_hero_and_enemy_tower(frame_state, camp)
        if not main_hero:
            return

        stats["frame_count"] += 1
        hp_rate = self._hp_rate(main_hero)
        dist_to_enemy_tower = self._distance(main_hero.get("location", {}), enemy_tower.get("location", {}))
        near_enemy_tower = dist_to_enemy_tower < GameConfig.ENEMY_TOWER_DANGER_RADIUS
        if hp_rate < GameConfig.RISK_LOW_HP_THRESHOLD and near_enemy_tower:
            stats["low_hp_enemy_tower_frames"] += 1
        if main_hero.get("hp", 0) <= 0 and near_enemy_tower:
            stats["tower_death_count"] += 1
        stats["risk_exposure_sum"] += reward.get("risk_exposure", 0.0)

    def _format_behavior_stats(self, stats):
        frame_count = max(stats["frame_count"], 1)
        return {
            "low_hp_enemy_tower_ratio": round(stats["low_hp_enemy_tower_frames"] / frame_count, 4),
            "avg_risk_exposure": round(stats["risk_exposure_sum"] / frame_count, 4),
            "tower_death_count": stats["tower_death_count"],
        }

    def _select_main_hero_and_enemy_tower(self, frame_state, camp):
        main_hero, enemy_tower = None, {}
        for hero in frame_state.get("hero_states", []):
            if hero.get("camp") == camp:
                main_hero = hero
                break
        for organ in frame_state.get("npc_states", []):
            if organ.get("camp") != camp and organ.get("sub_type") == 21:
                enemy_tower = organ
                break
        return main_hero, enemy_tower

    def _hp_rate(self, unit):
        max_hp = unit.get("max_hp", 0)
        if max_hp <= 0:
            return 0.0
        return unit.get("hp", 0) / max_hp

    def _distance(self, pos1, pos2):
        if not pos1 or not pos2:
            return 0.0
        dx = pos1.get("x", 0) - pos2.get("x", 0)
        dz = pos1.get("z", 0) - pos2.get("z", 0)
        return (dx * dx + dz * dz) ** 0.5

''',
        "def _select_episode_opponent",
    )
    replace_once(
        rel,
        "    def reset_agents(self, observation):\n        opponent_agent = self.env_conf_manager.get_opponent_agent()\n",
        "    def reset_agents(self, observation, opponent_agent=None):\n"
        "        if opponent_agent is None:\n"
        "            opponent_agent = self.env_conf_manager.get_opponent_agent()\n",
    )


def patch_monitor():
    rel = "agent_ppo/conf/monitor_builder.py"
    insert_before(
        rel,
        "    config_dict = monitor.end_group().build()\n",
        r'''    monitor.add_panel(name="低血敌塔暴露比例", name_en="low_hp_enemy_tower_ratio", type="line").add_metric(
        metrics_name="low_hp_enemy_tower_ratio",
        expr="round(avg(low_hp_enemy_tower_ratio{}), 0.0001)",
    ).end_panel()
    monitor.add_panel(name="平均风险暴露", name_en="avg_risk_exposure", type="line").add_metric(
        metrics_name="avg_risk_exposure",
        expr="round(avg(avg_risk_exposure{}), 0.0001)",
    ).end_panel()
    monitor.add_panel(name="塔下死亡次数", name_en="tower_death_count", type="line").add_metric(
        metrics_name="tower_death_count",
        expr="round(avg(tower_death_count{}), 0.01)",
    ).end_panel()
''',
        "low_hp_enemy_tower_ratio",
    )


def patch_tests():
    write("tests/test_ppo_core.py", r'''#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import os
import sys
import types


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config, GameConfig
from agent_ppo.feature.feature_process import FeatureProcess
from agent_ppo.feature.reward_process import GameRewardManager
from agent_ppo.model.model import Model


def install_workflow_import_stubs():
    tools_module = types.ModuleType("tools")
    env_conf_module = types.ModuleType("tools.env_conf_manager")
    model_pool_module = types.ModuleType("tools.model_pool_utils")
    metrics_module = types.ModuleType("tools.metrics_utils")
    common_python_module = types.ModuleType("common_python")
    common_python_utils_module = types.ModuleType("common_python.utils")
    common_func_module = types.ModuleType("common_python.utils.common_func")
    disaster_module = types.ModuleType("common_python.utils.workflow_disaster_recovery")

    class EnvConfManager:
        pass

    env_conf_module.EnvConfManager = EnvConfManager
    common_func_module.Frame = type("Frame", (), {})
    common_func_module.create_cls = lambda name, **kwargs: type(name, (), {})
    model_pool_module.get_valid_model_pool = lambda logger=None: []
    metrics_module.get_training_metrics = lambda: {}
    disaster_module.handle_disaster_recovery = lambda env_obs, logger=None: False

    sys.modules.setdefault("tools", tools_module)
    sys.modules.setdefault("tools.env_conf_manager", env_conf_module)
    sys.modules.setdefault("tools.model_pool_utils", model_pool_module)
    sys.modules.setdefault("tools.metrics_utils", metrics_module)
    sys.modules.setdefault("common_python", common_python_module)
    sys.modules.setdefault("common_python.utils", common_python_utils_module)
    sys.modules.setdefault("common_python.utils.common_func", common_func_module)
    sys.modules.setdefault("common_python.utils.workflow_disaster_recovery", disaster_module)


class SampleData:
    def __init__(self, sample):
        self.sample = sample


def fake_frame_state(
    main_hp=1000,
    enemy_hp=1000,
    frame_no=1,
    main_location=None,
    enemy_location=None,
    enemy_tower_location=None,
):
    main_location = main_location or {"x": -10000, "z": 0}
    enemy_location = enemy_location or {"x": 10000, "z": 0}
    enemy_tower_location = enemy_tower_location or {"x": 30000, "z": 0}
    return {
        "frame_no": frame_no,
        "hero_states": [
            {
                "runtime_id": 1,
                "config_id": 112,
                "camp": 1,
                "hp": main_hp,
                "max_hp": 1000,
                "level": 3,
                "exp": 80,
                "max_exp": 160,
                "money": 1200,
                "location": main_location,
                "skill_state": [{"usable": True}, {"cooldown": 2, "max_cooldown": 10}],
            },
            {
                "runtime_id": 2,
                "config_id": 133,
                "camp": 2,
                "hp": enemy_hp,
                "max_hp": 1000,
                "level": 3,
                "exp": 80,
                "max_exp": 160,
                "money": 1200,
                "location": enemy_location,
            },
        ],
        "npc_states": [
            {"camp": 1, "sub_type": 21, "hp": 5000, "max_hp": 5000, "location": {"x": -30000, "z": 0}},
            {"camp": 2, "sub_type": 21, "hp": 5000, "max_hp": 5000, "location": enemy_tower_location},
        ],
    }


def build_fake_sample():
    per_frame = []
    feature = torch.zeros(Config.FEATURE_DIM)
    legal_action = torch.ones(Config.REDUCED_LEGAL_ACTION_DIM)
    seri_vec = torch.cat([feature, legal_action])
    per_frame.append(seri_vec.repeat(Config.LSTM_TIME_STEPS))
    per_frame.append(torch.ones(Config.LSTM_TIME_STEPS))
    per_frame.append(torch.linspace(-1.0, 1.0, Config.LSTM_TIME_STEPS))

    for label_size in Config.LABEL_SIZE_LIST:
        per_frame.append(torch.zeros(Config.LSTM_TIME_STEPS))

    for label_size in Config.LABEL_SIZE_LIST:
        per_frame.append(torch.full((label_size * Config.LSTM_TIME_STEPS,), 1.0 / label_size))

    for _ in Config.LABEL_SIZE_LIST:
        per_frame.append(torch.ones(Config.LSTM_TIME_STEPS))

    per_frame.append(torch.ones(Config.LSTM_TIME_STEPS))
    per_frame.append(torch.zeros(Config.LSTM_UNIT_SIZE))
    per_frame.append(torch.zeros(Config.LSTM_UNIT_SIZE))
    return torch.cat(per_frame).float()


def test_config_shapes_are_consistent():
    assert Config.FEATURE_DIM == Config.SERI_VEC_SPLIT_SHAPE[0][0]
    assert Config.REDUCED_LEGAL_ACTION_DIM == sum(Config.LABEL_SIZE_LIST)
    assert Config.DATA_SPLIT_SHAPE[0] == Config.FEATURE_DIM + Config.REDUCED_LEGAL_ACTION_DIM
    assert Config.SAMPLE_DIM == sum(shape[0] for shape in Config.data_shapes)


def test_feature_process_outputs_configured_dim():
    processor = FeatureProcess(camp=1)
    feature = processor.process_feature({"frame_state": fake_frame_state()})
    assert len(feature) == Config.FEATURE_DIM
    assert all(0.0 <= value <= 1.0 for value in feature)


def test_model_keeps_checkpoint_compatible_input_shape():
    model = Model()
    assert Config.FEATURE_DIM == 50
    assert tuple(model.concat_mlp.fc_layers.concat_mlp_fc1.weight.shape) == (256, 50)


def test_reward_manager_emits_all_reward_items():
    manager = GameRewardManager(main_hero_runtime_id=1)
    reward_1 = dict(manager.result(fake_frame_state(enemy_hp=1000, frame_no=1)))
    reward_2 = dict(manager.result(fake_frame_state(enemy_hp=900, frame_no=2)))
    assert set(GameConfig.REWARD_WEIGHT_DICT).issubset(reward_2.keys())
    assert "reward_sum" in reward_2
    assert reward_2["damage_to_enemy"] > reward_1["damage_to_enemy"]


def test_reward_manager_penalizes_risk_exposure():
    manager = GameRewardManager(main_hero_runtime_id=1)
    reward = dict(
        manager.result(
            fake_frame_state(
                main_hp=200,
                enemy_hp=1000,
                frame_no=1,
                main_location={"x": 27000, "z": 0},
                enemy_location={"x": 28000, "z": 0},
                enemy_tower_location={"x": 30000, "z": 0},
            )
        )
    )
    assert reward["risk_exposure"] > 0
    assert GameConfig.REWARD_WEIGHT_DICT["risk_exposure"] < 0


def test_curriculum_common_ai_ratio_decays():
    install_workflow_import_stubs()
    from agent_ppo.workflow.train_workflow import EpisodeRunner

    runner = EpisodeRunner.__new__(EpisodeRunner)
    assert runner._common_ai_ratio(0) == GameConfig.COMMON_AI_RATIO_START
    assert runner._common_ai_ratio(GameConfig.COMMON_AI_WARMUP_EPISODES) == GameConfig.COMMON_AI_RATIO_START
    late_episode = GameConfig.COMMON_AI_WARMUP_EPISODES + GameConfig.COMMON_AI_DECAY_EPISODES + 1
    assert runner._common_ai_ratio(late_episode) == GameConfig.COMMON_AI_RATIO_END


def test_ppo_learn_accepts_fake_sample():
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.INIT_LEARNING_RATE_START)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    algorithm = Algorithm(model, optimizer, scheduler, device=torch.device("cpu"))
    sample = SampleData(build_fake_sample())
    assert sample.sample.numel() == Config.SAMPLE_DIM
    result = algorithm.learn([sample])
    assert "total_loss" in result
    assert torch.isfinite(torch.tensor(result["total_loss"]))


if __name__ == "__main__":
    test_config_shapes_are_consistent()
    test_feature_process_outputs_configured_dim()
    test_model_keeps_checkpoint_compatible_input_shape()
    test_reward_manager_emits_all_reward_items()
    test_reward_manager_penalizes_risk_exposure()
    test_curriculum_common_ai_ratio_decays()
    test_ppo_learn_accepts_fake_sample()
''')


def patch_changelog():
    rel = "CHANG_LOG.md"
    path = root / rel
    if not path.exists():
        write(rel, "# 修改日志\n\n本文档用于记录项目每次代码修改的设计思路、核心改动点和验证方式。\n")
    text = read(rel)
    if "v2 风险敏感 PPO 与对手课程" in text:
        print("[skip] CHANG_LOG.md: v2 entry")
        return
    entry = r'''## 2026-06-21 - v2 风险敏感 PPO 与对手课程

### 修改思路
- `results/v1` 中 common_ai 评估胜率能上升但后期平台化，self-play 胜率长期震荡，说明策略有一定学习能力但泛化和稳定性不足。
- 击杀、伤害上升的同时死亡也明显上升，说明上一版奖励把进攻意愿拉起来了，但没有足够惩罚低血、敌塔附近和敌人贴近时的高风险进攻。
- 本轮引入风险敏感奖励、势函数差分塑形、课程化对手和轻量时序差分特征，让训练信号更稳定。

### 核心修改点
- 新增 `risk_exposure`，并在高风险状态下打折 `damage_to_enemy`。
- 将 `retreat_low_hp` 和 `under_enemy_tower` 改为势函数差分形式。
- 保持 50 维模型输入不变，避免和旧 checkpoint 的第一层参数 shape 冲突。
- 训练局加入 common_ai 到 self-play 的课程调度，评估局保持原配置逻辑。
- 新增低血敌塔暴露比例、平均风险暴露、塔下死亡次数监控。

### 验证方式
- 建议运行 `python tests/test_ppo_core.py`。
- 建议运行相关 Python 文件的 `py_compile` 语法检查。

### 备注
- 本轮仍未真正启用 LSTM 主干，只用时序差分特征做低成本替代。

'''
    if "## 2026-06-21 - PPO 主线特征" in text:
        text = text.replace("## 2026-06-21 - PPO 主线特征", entry + "## 2026-06-21 - PPO 主线特征", 1)
    else:
        text = text.rstrip() + "\n\n" + entry
    write(rel, text)


patch_conf()
patch_agent_load_model()
patch_feature_process()
patch_reward_process()
patch_workflow()
patch_monitor()
patch_tests()
patch_changelog()

print("\n[done] v2 risk-sensitive PPO curriculum changes applied.")
print("[hint] Run: python tests/test_ppo_core.py")
PY
