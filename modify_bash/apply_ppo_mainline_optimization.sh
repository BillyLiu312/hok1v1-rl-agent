#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

python3 - <<'PY'
import os
from pathlib import Path


root = Path(os.environ["REPO_ROOT"])


def write_file(rel_path, content):
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[write] {rel_path}")


def replace_once(rel_path, old, new):
    path = root / rel_path
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"[skip] {rel_path}: already applied")
        return
    if old not in text:
        raise RuntimeError(f"Cannot find expected text in {rel_path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[patch] {rel_path}")


def replace_all(rel_path, old, new):
    path = root / rel_path
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"[skip] {rel_path}: already applied")
        return
    if old not in text:
        raise RuntimeError(f"Cannot find expected text in {rel_path}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"[patch] {rel_path}")


write_file("agent_ppo/conf/conf.py", r'''#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT = {
        "tower_hp_point": 5.0,
        "hero_hp_point": 2.0,
        "damage_to_enemy": 2.0,
        "damage_taken": -1.5,
        "kill": 8.0,
        "death": -8.0,
        "forward": 0.01,
        "retreat_low_hp": 0.05,
        "under_enemy_tower": -0.02,
    }
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 0
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    DIM_OF_FEATURE = [50]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    FEATURE_DIM = DimConfig.DIM_OF_FEATURE[0]
    REDUCED_LEGAL_ACTION_DIM = 85
    DATA_SPLIT_SHAPE = []
    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (REDUCED_LEGAL_ACTION_DIM,)]
    INIT_LEARNING_RATE_START = 1e-3
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = []

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = 0


Config.REDUCED_LEGAL_ACTION_DIM = sum(Config.LABEL_SIZE_LIST)
Config.SERI_VEC_SPLIT_SHAPE = [(Config.FEATURE_DIM,), (Config.REDUCED_LEGAL_ACTION_DIM,)]
Config.DATA_SPLIT_SHAPE = (
    [Config.FEATURE_DIM + Config.REDUCED_LEGAL_ACTION_DIM, 1, 1]
    + [1 for _ in Config.LABEL_SIZE_LIST]
    + Config.LABEL_SIZE_LIST
    + [1 for _ in Config.LABEL_SIZE_LIST]
    + [1, Config.LSTM_UNIT_SIZE, Config.LSTM_UNIT_SIZE]
)
Config.data_shapes = (
    [[shape * Config.LSTM_TIME_STEPS] for shape in Config.DATA_SPLIT_SHAPE[:-2]]
    + [[Config.LSTM_UNIT_SIZE], [Config.LSTM_UNIT_SIZE]]
)
Config.SAMPLE_DIM = sum(Config.DATA_SPLIT_SHAPE[:-2]) * Config.LSTM_TIME_STEPS + sum(Config.DATA_SPLIT_SHAPE[-2:])
''')


write_file("agent_ppo/feature/feature_process/__init__.py", r'''#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess
from agent_ppo.conf.conf import Config


def _clip(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def _safe_div(numerator, denominator, default=0.0):
    if denominator in (0, None):
        return default
    return numerator / denominator


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        feature = self._process_expanded_feature(frame_state)
        assert len(feature) == Config.FEATURE_DIM, "feature dim mismatch: {}/{}".format(len(feature), Config.FEATURE_DIM)
        return feature

    def _process_expanded_feature(self, frame_state):
        main_hero, enemy_hero = self._select_heroes(frame_state)
        main_tower, enemy_tower = self._select_towers(frame_state)

        feature = []
        feature.extend(self._hero_feature(main_hero, enemy_hero, enemy_tower))
        feature.extend(self._hero_feature(enemy_hero, main_hero, main_tower))
        feature.extend(self._tower_feature(main_tower, main_hero))
        feature.extend(self._tower_feature(enemy_tower, main_hero))
        feature.extend(self._pair_feature(main_hero, enemy_hero, main_tower, enemy_tower, frame_state))
        feature.extend(self._skill_feature(main_hero, 6))
        return feature

    def _select_heroes(self, frame_state):
        main_hero, enemy_hero = None, None
        for hero in frame_state.get("hero_states", []):
            if hero.get("camp") == self.camp and main_hero is None:
                main_hero = hero
            elif hero.get("camp") != self.camp and enemy_hero is None:
                enemy_hero = hero
        return main_hero or {}, enemy_hero or {}

    def _select_towers(self, frame_state):
        main_tower, enemy_tower = None, None
        for organ in frame_state.get("npc_states", []):
            if organ.get("sub_type") != 21:
                continue
            if organ.get("camp") == self.camp and main_tower is None:
                main_tower = organ
            elif organ.get("camp") != self.camp and enemy_tower is None:
                enemy_tower = organ
        return main_tower or {}, enemy_tower or {}

    def _hero_feature(self, hero, other_hero, enemy_tower):
        location = hero.get("location", {})
        other_location = other_hero.get("location", {})
        tower_location = enemy_tower.get("location", {})
        hp_rate = self._hp_rate(hero)
        return [
            1.0 if hero.get("hp", 0) > 0 else 0.0,
            hp_rate,
            _clip(hero.get("level", hero.get("lv", 1)) / 15.0),
            _clip(_safe_div(hero.get("exp", 0), hero.get("max_exp", 1))),
            _clip(hero.get("money", hero.get("gold", 0)) / 20000.0),
            self._norm_coord(location.get("x", 0)),
            self._norm_coord(location.get("z", 0)),
            self._norm_distance(location, other_location),
            self._norm_distance(location, tower_location),
        ]

    def _tower_feature(self, tower, main_hero):
        location = tower.get("location", {})
        hero_location = main_hero.get("location", {})
        return [
            1.0 if tower.get("hp", 0) > 0 else 0.0,
            self._hp_rate(tower),
            self._norm_coord(location.get("x", 0)),
            self._norm_coord(location.get("z", 0)),
            self._relative_coord(location.get("x", 0), hero_location.get("x", 0)),
            self._relative_coord(location.get("z", 0), hero_location.get("z", 0)),
        ]

    def _pair_feature(self, main_hero, enemy_hero, main_tower, enemy_tower, frame_state):
        main_location = main_hero.get("location", {})
        enemy_location = enemy_hero.get("location", {})
        main_tower_location = main_tower.get("location", {})
        enemy_tower_location = enemy_tower.get("location", {})
        main_hp_rate = self._hp_rate(main_hero)
        enemy_hp_rate = self._hp_rate(enemy_hero)
        main_level = main_hero.get("level", main_hero.get("lv", 1))
        enemy_level = enemy_hero.get("level", enemy_hero.get("lv", 1))
        return [
            1.0 if enemy_hero else 0.0,
            _clip((main_hp_rate - enemy_hp_rate + 1.0) / 2.0),
            _clip((main_level - enemy_level + 15.0) / 30.0),
            self._norm_distance(main_location, enemy_location),
            self._norm_distance(main_location, enemy_tower_location),
            self._norm_distance(main_location, main_tower_location),
            self._norm_distance(enemy_location, main_tower_location),
            self._lane_progress(main_location, main_tower_location, enemy_tower_location),
            self._lane_progress(enemy_location, enemy_tower_location, main_tower_location),
            1.0 if self._distance(main_location, enemy_tower_location) < 6500 else 0.0,
            1.0 if self._distance(main_location, main_tower_location) < 6500 else 0.0,
            1.0 if main_hero.get("hp", 0) <= 0 else 0.0,
            1.0 if enemy_hero.get("hp", 0) <= 0 else 0.0,
            _clip(frame_state.get("frame_no", 0) / 18000.0),
        ]

    def _skill_feature(self, hero, count):
        skills = hero.get("skill_state") or hero.get("skill_states") or hero.get("skills") or hero.get("skill_list") or []
        if isinstance(skills, dict):
            skills = list(skills.values())

        result = []
        for skill in list(skills)[:count]:
            if not isinstance(skill, dict):
                result.append(0.0)
                continue
            if "usable" in skill:
                result.append(1.0 if skill.get("usable") else 0.0)
                continue
            cooldown = skill.get("cooldown", skill.get("cool_down", skill.get("cd", skill.get("remaining_cooldown", 0))))
            max_cooldown = skill.get("max_cooldown", skill.get("max_cd", cooldown if cooldown else 1))
            result.append(_clip(1.0 - _safe_div(cooldown, max_cooldown)))

        while len(result) < count:
            result.append(0.0)
        return result

    def _hp_rate(self, unit):
        return _clip(_safe_div(unit.get("hp", 0), unit.get("max_hp", 1)))

    def _norm_coord(self, value):
        if self.camp == 2 and value != 100000:
            value = -value
        return _clip((value + 60000.0) / 120000.0)

    def _relative_coord(self, target_value, origin_value):
        if self.camp == 2 and target_value != 100000:
            target_value = -target_value
            origin_value = -origin_value
        return _clip((target_value - origin_value + 30000.0) / 60000.0)

    def _distance(self, pos1, pos2):
        if not pos1 or not pos2:
            return 0.0
        dx = pos1.get("x", 0) - pos2.get("x", 0)
        dz = pos1.get("z", 0) - pos2.get("z", 0)
        return (dx * dx + dz * dz) ** 0.5

    def _norm_distance(self, pos1, pos2):
        return _clip(self._distance(pos1, pos2) / 120000.0)

    def _lane_progress(self, hero_pos, own_tower_pos, enemy_tower_pos):
        full_dist = self._distance(own_tower_pos, enemy_tower_pos)
        if full_dist <= 0:
            return 0.0
        dist_to_enemy = self._distance(hero_pos, enemy_tower_pos)
        return _clip(1.0 - dist_to_enemy / full_dist)
''')


write_file("agent_ppo/feature/reward_process.py", r'''#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
from agent_ppo.conf.conf import GameConfig


def safe_div(numerator, denominator, default=0.0):
    if denominator in (0, None):
        return default
    return numerator / denominator


def distance(pos1, pos2):
    if not pos1 or not pos2:
        return 0.0
    return math.dist((pos1.get("x", 0), pos1.get("z", 0)), (pos2.get("x", 0), pos2.get("z", 0)))


# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        frame_no = frame_data["frame_no"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero

        # Get both defense towers
        # 获取双方防御塔
        main_tower, enemy_tower = None, None
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == camp:
                if organ_subtype == 21:
                    main_tower = organ
            else:
                if organ_subtype == 21:
                    enemy_tower = organ

        default_location = {"x": 0, "z": 0}
        main_hero = main_hero or {"hp": 0, "max_hp": 1, "location": default_location}
        enemy_hero = enemy_hero or {"hp": 0, "max_hp": 1, "location": default_location}
        main_tower = main_tower or {"hp": 0, "max_hp": 1, "location": default_location}
        enemy_tower = enemy_tower or {"hp": 0, "max_hp": 1, "location": default_location}

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            # Tower health points
            # 塔血量
            if reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = safe_div(main_tower.get("hp", 0), main_tower.get("max_hp", 1))
            # Hero health points
            # 英雄血量
            elif reward_name in ("hero_hp_point", "damage_taken"):
                reward_struct.cur_frame_value = safe_div(main_hero.get("hp", 0), main_hero.get("max_hp", 1))
            # Enemy hero health, used to derive damage dealt
            # 敌方英雄血量，用于计算造成伤害
            elif reward_name == "damage_to_enemy":
                reward_struct.cur_frame_value = safe_div(enemy_hero.get("hp", 0), enemy_hero.get("max_hp", 1))
            # Kill/death terminal signals
            # 击杀/死亡终局信号
            elif reward_name == "kill":
                reward_struct.cur_frame_value = 1.0 if enemy_hero.get("hp", 0) <= 0 else 0.0
            elif reward_name == "death":
                reward_struct.cur_frame_value = 1.0 if main_hero.get("hp", 0) <= 0 else 0.0
            # Forward
            # 前进
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            # Low-health safety shaping
            # 低血量安全位置塑形
            elif reward_name == "retreat_low_hp":
                hp_rate = safe_div(main_hero.get("hp", 0), main_hero.get("max_hp", 1))
                dist_to_own_tower = distance(main_hero.get("location", {}), main_tower.get("location", {}))
                safety = 1.0 - min(dist_to_own_tower / 30000.0, 1.0)
                reward_struct.cur_frame_value = (1.0 - hp_rate) * safety
            # Penalize standing near enemy tower, especially when low HP
            # 惩罚靠近敌方塔，低血时惩罚更强
            elif reward_name == "under_enemy_tower":
                hp_rate = safe_div(main_hero.get("hp", 0), main_hero.get("max_hp", 1))
                dist_to_enemy_tower = distance(main_hero.get("location", {}), enemy_tower.get("location", {}))
                danger = 1.0 if dist_to_enemy_tower < 6500 else 0.0
                reward_struct.cur_frame_value = danger * (1.0 + (1.0 - hp_rate))

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        main_tower_location = main_tower.get("location", {})
        enemy_tower_location = enemy_tower.get("location", {})
        main_tower_pos = (main_tower_location.get("x", 0), main_tower_location.get("z", 0))
        enemy_tower_pos = (enemy_tower_location.get("x", 0), enemy_tower_location.get("z", 0))
        main_hero_location = main_hero.get("location", {})
        hero_pos = (
            main_hero_location.get("x", 0),
            main_hero_location.get("z", 0),
        )
        forward_value = 0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        if dist_main2emy <= 0:
            return 0
        if safe_div(main_hero.get("hp", 0), main_hero.get("max_hp", 1)) > 0.99 and dist_hero2emy > dist_main2emy:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                main_camp = hero["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["camp"]
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "damage_to_enemy":
                reward_struct.value = max(
                    0.0,
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_main_calc_frame_map[reward_name].cur_frame_value,
                )
            elif reward_name == "damage_taken":
                reward_struct.value = max(
                    0.0,
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_main_calc_frame_map[reward_name].cur_frame_value,
                )
            elif reward_name in ("retreat_low_hp", "under_enemy_tower"):
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            else:
                # Calculate zero-sum reward
                # 计算零和奖励
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = reward_sum
''')


write_file("agent_ppo/conf/monitor_builder.py", r'''#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder
from agent_ppo.conf.conf import GameConfig


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    monitor.title("智能决策1v1").add_group(
        group_name="算法指标",
        group_name_en="algorithm",
    )
    monitor.add_panel(name="累积回报", name_en="reward", type="line").add_metric(
        metrics_name="reward",
        expr="round(avg(reward{}), 0.01)",
    ).end_panel()
    monitor.add_panel(name="总损失", name_en="total_loss", type="line").add_metric(
        metrics_name="total_loss",
        expr="round(avg(total_loss{}), 0.01)",
    ).end_panel()
    monitor.add_panel(name="价值损失", name_en="value_loss", type="line").add_metric(
        metrics_name="value_loss",
        expr="round(avg(value_loss{}), 0.01)",
    ).end_panel()
    monitor.add_panel(name="策略损失", name_en="policy_loss", type="line").add_metric(
        metrics_name="policy_loss",
        expr="round(avg(policy_loss{}), 0.01)",
    ).end_panel()
    monitor.add_panel(name="熵损失", name_en="entropy_loss", type="line").add_metric(
        metrics_name="entropy_loss",
        expr="round(avg(entropy_loss{}), 0.01)",
    ).end_panel()
    for reward_name in GameConfig.REWARD_WEIGHT_DICT:
        metric_name = f"reward_{reward_name}"
        monitor.add_panel(name=f"回报_{reward_name}", name_en=metric_name, type="line").add_metric(
            metrics_name=metric_name,
            expr=f"round(avg({metric_name}{{}}), 0.0001)",
        ).end_panel()
    config_dict = monitor.end_group().build()
    return config_dict
''')


write_file("agent_ppo/algorithm/algorithm.py", r'''#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
import os
import time
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.train_step = 0

        self.logger = logger
        self.monitor = monitor

        self.cut_points = [value[0] for value in Config.data_shapes]
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE

        self.last_report_monitor_time = 0

    def learn(self, list_sample_data):
        """
        list_sample_data: list[SampleData]
        SampleData对象列表
        """
        # Extract sample field from SampleData objects and stack into tensor
        # 从 SampleData 对象中提取 sample 字段并 stack 成 tensor
        _input_datas = torch.stack([sample.sample for sample in list_sample_data]).to(self.device)
        results = {}

        data_list = list(_input_datas.split(self.cut_points, dim=1))
        for i, data in enumerate(data_list):
            data = data.reshape(-1)
            data_list[i] = data.float()

        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        feature, legal_action = seri_vec.split(
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]

        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(format_inputs)
        total_loss, info_list = self.model.compute_loss(data_list, rst_list)
        results["total_loss"] = total_loss.item()

        total_loss.backward()

        # grad clip
        # 梯度剪裁
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        # update the learning rate
        # 更新学习率
        self.scheduler.step()

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            else:
                _info = info.item()
            _info_list.append(_info)

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = _info_list
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

        return results
''')


replace_once(
    "agent_ppo/agent.py",
    "        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]",
    "        torch_inputs = [torch.from_numpy(nparr).to(self.device).to(torch.float32) for nparr in input_list]",
)
replace_once(
    "agent_ppo/agent.py",
    "            np_output.append(output.numpy())",
    "            np_output.append(output.detach().cpu().numpy())",
)


replace_once(
    "agent_ppo/model/model.py",
    "    def compute_loss(self, data_list, rst_list):\n        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])",
    "    def compute_loss(self, data_list, rst_list):\n        device = rst_list[-1].device\n        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])",
)
replace_once(
    "agent_ppo/model/model.py",
    "        frame_is_train = usq_is_train.squeeze(dim=1)\n\n        label_result = rst_list[:-1]",
    "        frame_is_train = usq_is_train.squeeze(dim=1)\n        train_frame_count = torch.sum(frame_is_train)\n        if train_frame_count > 1:\n            advantage_mean = torch.sum(advantage * frame_is_train) / train_frame_count\n            advantage_var = torch.sum(torch.square(advantage - advantage_mean) * frame_is_train) / train_frame_count\n            advantage = (advantage - advantage_mean) / torch.sqrt(advantage_var + 1e-8)\n\n        label_result = rst_list[:-1]",
)
replace_once(
    "agent_ppo/model/model.py",
    "        self.value_cost = 0.5 * torch.mean(torch.square(reward - fc2_value_result_squeezed), dim=0)\n        new_advantage = reward - fc2_value_result_squeezed",
    "        new_advantage = reward - fc2_value_result_squeezed",
)
replace_once(
    "agent_ppo/model/model.py",
    "        self.policy_cost = torch.tensor(0.0)",
    "        self.policy_cost = torch.zeros((), device=device)",
)
replace_once(
    "agent_ppo/model/model.py",
    "                final_log_p = torch.tensor(0.0)\n                boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))",
    "                final_log_p = torch.zeros((), device=device)\n                boundary = torch.tensor(1e20, device=device)",
)
replace_all(
    "agent_ppo/model/model.py",
    ") / torch.maximum(torch.sum((weight_list[task_index].float()) * frame_is_train), torch.tensor(1.0))",
    ") / torch.clamp(torch.sum((weight_list[task_index].float()) * frame_is_train), min=1.0)",
)
replace_all(
    "agent_ppo/model/model.py",
    ") / torch.maximum(torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0))",
    ") / torch.clamp(torch.sum(weight_list[task_index].float() * frame_is_train), min=1.0)",
)
replace_once(
    "agent_ppo/model/model.py",
    "                temp_entropy_loss = torch.tensor(0.0)",
    "                temp_entropy_loss = torch.zeros((), device=device)",
)
replace_once(
    "agent_ppo/model/model.py",
    "        self.entropy_cost = torch.tensor(0.0)",
    "        self.entropy_cost = torch.zeros((), device=device)",
)
replace_once(
    "agent_ppo/model/model.py",
    "    nn.init.orthogonal(fc_layer.weight)",
    "    nn.init.orthogonal_(fc_layer.weight)",
)


replace_once(
    "agent_ppo/workflow/train_workflow.py",
    "            reward_sum_list = [0] * self.agent_num\n            is_train_test = os.environ.get(\"is_train_test\", \"False\").lower() == \"true\"",
    "            reward_sum_list = [0] * self.agent_num\n            reward_detail_sum_list = [dict() for _ in range(self.agent_num)]\n            is_train_test = os.environ.get(\"is_train_test\", \"False\").lower() == \"true\"",
)
replace_all(
    "agent_ppo/workflow/train_workflow.py",
    "                    reward_sum_list[i] += reward[\"reward_sum\"]",
    "                    reward_sum_list[i] += reward[\"reward_sum\"]\n                    self._accumulate_reward_detail(reward_detail_sum_list[i], reward)",
)
replace_once(
    "agent_ppo/workflow/train_workflow.py",
    "                        monitor_data = {\"episode_cnt\": self.episode_cnt}\n                        if self.monitor:\n                            if is_eval:",
    "                        monitor_data = {\"episode_cnt\": self.episode_cnt}\n                        if self.monitor:\n                            for reward_name, reward_value in reward_detail_sum_list[monitor_side].items():\n                                monitor_data[f\"reward_{reward_name}\"] = round(reward_value, 4)\n                            if is_eval:",
)
replace_once(
    "agent_ppo/workflow/train_workflow.py",
    "                    break\n\n    def reset_agents(self, observation):",
    "                    break\n\n    def _accumulate_reward_detail(self, reward_detail_sum, reward):\n        for reward_name, reward_value in reward.items():\n            if reward_name == \"reward_sum\":\n                continue\n            reward_detail_sum[reward_name] = reward_detail_sum.get(reward_name, 0.0) + reward_value\n\n    def reset_agents(self, observation):",
)


write_file("tests/test_ppo_core.py", r'''#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config, GameConfig
from agent_ppo.feature.feature_process import FeatureProcess
from agent_ppo.feature.reward_process import GameRewardManager
from agent_ppo.model.model import Model


class SampleData:
    def __init__(self, sample):
        self.sample = sample


def fake_frame_state(enemy_hp=1000, frame_no=1):
    return {
        "frame_no": frame_no,
        "hero_states": [
            {
                "runtime_id": 1,
                "config_id": 112,
                "camp": 1,
                "hp": 1000,
                "max_hp": 1000,
                "level": 3,
                "exp": 80,
                "max_exp": 160,
                "money": 1200,
                "location": {"x": -10000, "z": 0},
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
                "location": {"x": 10000, "z": 0},
            },
        ],
        "npc_states": [
            {"camp": 1, "sub_type": 21, "hp": 5000, "max_hp": 5000, "location": {"x": -30000, "z": 0}},
            {"camp": 2, "sub_type": 21, "hp": 5000, "max_hp": 5000, "location": {"x": 30000, "z": 0}},
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


def test_reward_manager_emits_all_reward_items():
    manager = GameRewardManager(main_hero_runtime_id=1)
    reward_1 = dict(manager.result(fake_frame_state(enemy_hp=1000, frame_no=1)))
    reward_2 = dict(manager.result(fake_frame_state(enemy_hp=900, frame_no=2)))
    assert set(GameConfig.REWARD_WEIGHT_DICT).issubset(reward_2.keys())
    assert "reward_sum" in reward_2
    assert reward_2["damage_to_enemy"] > reward_1["damage_to_enemy"]


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
    test_reward_manager_emits_all_reward_items()
    test_ppo_learn_accepts_fake_sample()
''')

print("[done] PPO mainline optimization patch has been applied.")
PY

python3 -m py_compile \
  "${REPO_ROOT}/agent_ppo/conf/conf.py" \
  "${REPO_ROOT}/agent_ppo/feature/feature_process/__init__.py" \
  "${REPO_ROOT}/agent_ppo/feature/reward_process.py" \
  "${REPO_ROOT}/agent_ppo/model/model.py" \
  "${REPO_ROOT}/agent_ppo/algorithm/algorithm.py" \
  "${REPO_ROOT}/agent_ppo/agent.py" \
  "${REPO_ROOT}/agent_ppo/workflow/train_workflow.py" \
  "${REPO_ROOT}/agent_ppo/conf/monitor_builder.py" \
  "${REPO_ROOT}/tests/test_ppo_core.py"

echo "[ok] Syntax check passed."
echo "Optional smoke test: python3 tests/test_ppo_core.py"
