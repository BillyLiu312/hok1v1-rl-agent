#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
import random
from agent_ppo.feature.definition import (
    sample_process,
    build_frame,
    FrameCollector,
    NONE_ACTION,
    lineup_iterator_roundrobin_camp_heroes,
)
from agent_ppo.conf.conf import GameConfig
from agent_ppo.conf.evaluation_config import camp_has_preset_skills
from agent_ppo.conf.opponent_schedule import (
    apply_opponent_agent,
    load_model_pool,
    load_opponent_schedule,
    select_curriculum_opponent,
)
from utils.training_recorder import TrainingRecorder
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    # Whether the agent is training, corresponding to do_predicts
    # 智能体是否进行训练
    do_learns = [True, True]
    last_save_model_time = time.time()
    training_recorder = TrainingRecorder(logger=logger)
    training_recorder.record_config_snapshot(
        name="ppo_training_start",
        paths=[
            "agent_ppo/conf/train_env_conf.toml",
            "agent_ppo/conf/conf.py",
            "conf/configure_app.toml",
            "conf/algo_conf_hok1v1.toml",
            "conf/app_conf_hok1v1.toml",
            "kaiwu.json",
        ],
        extra={
            "agent": "agent_ppo",
            "workflow": "agent_ppo/workflow/train_workflow.py",
        },
    )

    # Create environment configuration manager instance
    # 创建对局配置管理器实例
    env_conf_manager = EnvConfManager(
        config_path="agent_ppo/conf/train_env_conf.toml",
        logger=logger,
    )

    # Lineup iterator (112:Luban, 133:DiRenjie, 199:Arli)
    # 阵容迭代器 (112:鲁班, 133:狄仁杰, 199:公孙离)
    lineup_iterator = lineup_iterator_roundrobin_camp_heroes([112, 133, 199])

    # Create EpisodeRunner instance
    # 创建 EpisodeRunner 实例
    episode_runner = EpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
        lineup_iterator=lineup_iterator,
        training_recorder=training_recorder,
    )

    while True:
        # Run episodes and collect data
        # 运行对局并收集数据
        for g_data in episode_runner.run_episodes():
            for index, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                if d_learn and len(g_data[index]) > 0:
                    # The learner trains in a while true loop, here learn actually sends samples
                    # learner 采用 while true 训练，此处 learn 实际为发送样本
                    agent.send_sample_data(g_data[index])
            g_data.clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                training_recorder.record(
                    "model_save",
                    {
                        "episode_cnt": episode_runner.episode_cnt,
                        "save_interval_seconds": GameConfig.MODEL_SAVE_INTERVAL,
                    },
                )
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agents, logger, monitor, env_conf_manager, lineup_iterator, training_recorder):
        self.env = env
        self.agents = agents
        self.logger = logger
        self.monitor = monitor
        self.env_conf_manager = env_conf_manager
        self.lineup_iterator = lineup_iterator
        self.training_recorder = training_recorder
        self.agent_num = len(agents)
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.current_opponent_agent = None
        self.latest_training_metrics = {}

    def _call_init_config(self, usr_conf):
        """Call init_config on both agents to get summoner skill selections,
        then inject the results into usr_conf.
        调用双方 agent 的 init_config 获取召唤师技能选择，并注入 usr_conf。
        """
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)

        # camp_keys[i] is the camp key for agents[i] based on monitor_side
        # monitor_side 的 agent 对应 blue/red 取决于 monitor_side 配置
        monitor_side = self.env_conf_manager.get_monitor_side()
        camp_keys = ["blue_camp", "red_camp"]

        for agent_idx, agent in enumerate(self.agents):
            # Determine which camp this agent controls
            # 确定该 agent 控制哪个阵营
            if agent_idx == 0:
                my_hero_ids = blue_hero_ids
                opponent_hero_ids = red_hero_ids
                camp_key = camp_keys[0]
            else:
                my_hero_ids = red_hero_ids
                opponent_hero_ids = blue_hero_ids
                camp_key = camp_keys[1]

            config_data = {
                "my_camp": camp_key,
                "my_heroes": my_hero_ids,
                "opponent_heroes": opponent_hero_ids,
            }

            if camp_has_preset_skills(usr_conf, camp_key):
                self.logger.info(f"Agent[{agent_idx}] init_config skipped: camp={camp_key} preset skills found")
                continue

            select_skills = agent.init_config(config_data)
            EnvConfManager.inject_select_skills(usr_conf, camp_key, select_skills)
            self.logger.info(
                f"Agent[{agent_idx}] init_config: camp={camp_key}, select_skills={select_skills}"
            )

    def run_episodes(self):
        # Single environment process
        # 单局流程
        while True:
            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                self.latest_training_metrics = training_metrics
                self.training_recorder.record(
                    "metrics",
                    {
                        "episode_cnt": self.episode_cnt,
                        "training_metrics": training_metrics,
                    },
                )
                for key, value in training_metrics.items():
                    if key == "env":
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

            # Update environment configuration
            # Can use a list of length 2 to pass in the lineup id of the current game
            # 更新对局配置, 可以用长度为2的列表传入当前对局的阵容id
            lineup = next(self.lineup_iterator)
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)
            configured_opponent_agent = self.env_conf_manager.get_opponent_agent()
            current_opponent_agent = self._select_episode_opponent(configured_opponent_agent, is_eval)
            apply_opponent_agent(usr_conf, current_opponent_agent)
            self.current_opponent_agent = current_opponent_agent

            # Call init_config on agents to get summoner skill selections
            # 调用 agent 的 init_config 获取召唤师技能选择，注入 usr_conf
            self._call_init_config(usr_conf)
            blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)

            # Start a new environment
            # 启动新对局，返回初始环境状态

            env_obs = self.env.reset(usr_conf=usr_conf)
            # Disaster recovery
            # 容灾
            if handle_disaster_recovery(env_obs, self.logger):
                break

            observation = env_obs["observation"]
            extra_info = env_obs["extra_info"]

            # Reset agents
            # 重置智能体
            self.reset_agents(observation)

            # Reset environment frame collector
            # 重置环境帧收集器
            frame_collector = FrameCollector(self.agent_num)

            # Game variables
            # 对局变量
            self.episode_cnt += 1
            frame_no = 0
            reward_sum_list = [0] * self.agent_num
            reward_detail_sum_list = [dict() for _ in range(self.agent_num)]
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")
            self.training_recorder.record(
                "episode_start",
                {
                    "episode_cnt": self.episode_cnt,
                    "lineup": lineup,
                    "blue_hero_ids": blue_hero_ids,
                    "red_hero_ids": red_hero_ids,
                    "is_eval": is_eval,
                    "monitor_side": monitor_side,
                    "configured_opponent_agent": configured_opponent_agent,
                    "opponent_agent": current_opponent_agent,
                    "checkpoint": self._extract_checkpoint_snapshot(),
                    "usr_conf": usr_conf,
                },
            )

            # Reward initialization
            # 回报初始化
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += reward["reward_sum"]
                    self._accumulate_reward_detail(reward_detail_sum_list[i], reward)

            while True:
                # Initialize the default actions. If the agent does not make a decision, env.step uses the default action.
                # 初始化默认的actions，如果智能体不进行决策，则env.step使用默认action
                actions = [NONE_ACTION] * self.agent_num

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        if not is_eval:
                            actions[index] = agent.predict(observation[str(index)])
                        else:
                            actions[index] = agent.exploit(observation[str(index)])

                        # Only sample when do_sample=True and is_eval=False
                        # 评估对局数据不采样，不是训练中最新模型产生的数据不采样
                        if not is_eval and do_sample:
                            frame = build_frame(agent, observation[str(index)])
                            frame_collector.save_frame(frame, agent_id=index)

                # Step forward
                # 推进环境到下一帧，得到新的状态
                env_reward, env_obs = self.env.step(actions)
                # Disaster recovery
                # 容灾
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs["frame_no"]
                observation = env_obs["observation"]
                extra_info = env_obs["extra_info"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                # Reward generation
                # 计算回报，作为当前环境状态observation的一部分
                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += reward["reward_sum"]
                        self._accumulate_reward_detail(reward_detail_sum_list[i], reward)

                # Normal end or timeout exit, run train_test will exit early
                # 正常结束或超时退出，运行train_test时会提前退出
                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
                if is_gameover:
                    if not (is_train_test and frame_no >= 1000 and not terminated and not truncated):
                        for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                            if do_sample:
                                current_reward = observation[str(i)]["reward"]
                                terminal_reward = agent.reward_manager.terminal_reward(
                                    observation[str(i)]["frame_state"],
                                    win=observation[str(i)].get("win", 0),
                                    truncated=bool(truncated),
                                )
                                combined_reward = dict(current_reward)
                                combined_reward.update({key: value for key, value in terminal_reward.items() if key != "reward_sum"})
                                combined_reward["reward_sum"] = current_reward["reward_sum"] + terminal_reward["reward_sum"]
                                observation[str(i)]["reward"] = combined_reward
                                reward_sum_list[i] += terminal_reward["reward_sum"]
                                self._accumulate_reward_detail(reward_detail_sum_list[i], terminal_reward)

                    episode_record = self._build_episode_record(
                        observation=observation,
                        usr_conf=usr_conf,
                        lineup=lineup,
                        blue_hero_ids=blue_hero_ids,
                        red_hero_ids=red_hero_ids,
                        is_eval=is_eval,
                        monitor_side=monitor_side,
                        frame_no=frame_no,
                        terminated=terminated,
                        truncated=truncated,
                        reward_sum_list=reward_sum_list,
                        reward_detail_sum_list=reward_detail_sum_list,
                    )
                    self.training_recorder.record("episode_end", episode_record)
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, eval:{is_eval}, reward_sum:{reward_sum_list[monitor_side]}"
                    )
                    # Reward for saving the last state of the environment
                    # 保存环境最后状态的reward
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            frame_collector.save_last_frame(
                                agent_id=i,
                                reward=observation[str(i)]["reward"]["reward_sum"],
                            )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60:
                        monitor_data = {"episode_cnt": self.episode_cnt}
                        reward_detail = reward_detail_sum_list[monitor_side]
                        for key in (
                            "tower_hp_point",
                            "enemy_tower_hp_down",
                            "self_tower_hp_down",
                            "tower_destroy",
                            "hp_point",
                            "money",
                            "exp",
                            "kill",
                            "death",
                            "forward",
                            "push_window_tower_damage",
                            "unsafe_dive",
                            "push_window_active",
                            "unsafe_dive_active",
                            "win_result",
                            "timeout_tower_gap",
                        ):
                            if key in reward_detail:
                                monitor_data[f"reward_{key}"] = round(reward_detail[key], 4)
                        if self.monitor:
                            if is_eval:
                                monitor_data["reward"] = round(reward_sum_list[monitor_side], 2)
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                    # Sample process
                    # 进行样本处理，准备训练
                    if len(frame_collector) > 0 and not is_eval:
                        list_agents_samples = sample_process(frame_collector)
                        self.training_recorder.record(
                            "sample_batch",
                            {
                                "episode_cnt": self.episode_cnt,
                                "is_eval": is_eval,
                                "sample_counts": [len(agent_samples) for agent_samples in list_agents_samples],
                            },
                        )
                        yield list_agents_samples
                    break

    def reset_agents(self, observation):
        opponent_agent = self.env_conf_manager.get_opponent_agent()
        if self.current_opponent_agent is not None:
            opponent_agent = self.current_opponent_agent
        monitor_side = self.env_conf_manager.get_monitor_side()
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"

        # The 'do_predicts' specifies which agents are to perform model predictions.
        # do_predicts 指定哪些智能体要进行模型预测
        # The 'do_samples' specifies which agents are to perform training sampling.
        # do_samples 指定哪些智能体要进行训练采样
        self.do_predicts = [True, True]
        self.do_samples = [True, True]

        # Load model according to the configuration
        # 根据对局配置加载模型
        for i, agent in enumerate(self.agents):
            # Report the latest model in the training camp to the monitor
            # 训练中最新模型所在阵营上报监控
            if i == monitor_side:
                # monitor_side uses the latest model
                # monitor_side 使用最新模型
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    # common_ai does not need to load a model, no need to predict
                    # 如果对手是 common_ai 则不需要加载模型, 也不需要进行预测
                    self.do_predicts[i] = False
                    self.do_samples[i] = False
                elif opponent_agent == "selfplay":
                    # Training model, "latest" - latest model, "random" - random model from the model pool
                    # 加载训练过的模型，可以选择最新模型，也可以选择随机模型 "latest" - 最新模型, "random" - 模型池中随机模型
                    agent.load_model(id="latest")
                else:
                    # Opponent model, model_id is checked from kaiwu.json
                    # 选择kaiwu.json中设置的对手模型, model_id 即 opponent_agent，必须设置正确否则报错
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    else:
                        if is_train_test:
                            # Run train_test, cannot get opponent agent, so replace with latest model
                            # 运行 train_test 时, 无法获取到对手模型，因此将替换为最新模型
                            self.logger.info(f"Run train_test, cannot get opponent agent, so replace with latest model")
                            agent.load_model(id="latest")
                        else:
                            agent.load_opponent_agent(id=opponent_agent)
                        self.do_samples[i] = False
            # Reset agent
            # 重置agent
            agent.reset(observation[str(i)])

    def _accumulate_reward_detail(self, reward_detail, reward):
        for key, value in reward.items():
            if key == "reward_sum":
                continue
            reward_detail[key] = reward_detail.get(key, 0.0) + value

    def _build_episode_record(
        self,
        observation,
        usr_conf,
        lineup,
        blue_hero_ids,
        red_hero_ids,
        is_eval,
        monitor_side,
        frame_no,
        terminated,
        truncated,
        reward_sum_list,
        reward_detail_sum_list,
    ):
        return {
            "episode_cnt": self.episode_cnt,
            "lineup": lineup,
            "blue_hero_ids": blue_hero_ids,
            "red_hero_ids": red_hero_ids,
            "matchup": f"{blue_hero_ids[0] if blue_hero_ids else 'unknown'}_vs_{red_hero_ids[0] if red_hero_ids else 'unknown'}",
            "is_eval": is_eval,
            "monitor_side": monitor_side,
            "monitor_agent_index": monitor_side,
            "monitor_hero_id": self._first_or_none(blue_hero_ids if monitor_side == 0 else red_hero_ids),
            "opponent_hero_id": self._first_or_none(red_hero_ids if monitor_side == 0 else blue_hero_ids),
            "opponent_agent": self.current_opponent_agent or self.env_conf_manager.get_opponent_agent(),
            "checkpoint": self._extract_checkpoint_snapshot(),
            "usr_conf": usr_conf,
            "frame_no": frame_no,
            "terminated": terminated,
            "truncated": truncated,
            "reward_sum": reward_sum_list,
            "reward_detail": reward_detail_sum_list,
            "agents": [
                self._summarize_observation(index, observation.get(str(index), {}))
                for index in range(self.agent_num)
            ],
        }

    def _first_or_none(self, values):
        return values[0] if values else None

    def _select_episode_opponent(self, configured_opponent_agent, is_eval):
        if is_eval or configured_opponent_agent != "curriculum":
            return configured_opponent_agent
        return select_curriculum_opponent(
            model_pool=load_model_pool(),
            schedule=load_opponent_schedule(),
            rng=random,
        )

    def _extract_checkpoint_snapshot(self):
        metrics = self.latest_training_metrics or {}
        candidate_sources = [metrics]
        env_metrics = metrics.get("env")
        if isinstance(env_metrics, dict):
            candidate_sources.append(env_metrics)

        snapshot = {}
        for source in candidate_sources:
            if not isinstance(source, dict):
                continue
            for key in (
                "train_global_step",
                "actual_train_global_step",
                "global_step",
                "model_id",
                "checkpoint",
            ):
                if key in source:
                    snapshot[key] = source[key]
        if snapshot:
            snapshot["episode_cnt"] = self.episode_cnt
        return snapshot

    def _summarize_observation(self, agent_index, agent_observation):
        frame_state = agent_observation.get("frame_state", {})
        heroes = frame_state.get("hero_states", [])
        npcs = frame_state.get("npc_states", [])
        player_id = agent_observation.get("player_id")
        camp = agent_observation.get("camp", agent_observation.get("player_camp"))
        hero = self._find_main_hero(heroes, player_id, camp)
        enemy_hero = self._find_enemy_hero(heroes, camp)
        tower = self._find_tower(npcs, camp)
        enemy_tower = self._find_enemy_tower(npcs, camp)

        return {
            "agent_index": agent_index,
            "env_id": agent_observation.get("env_id"),
            "player_id": player_id,
            "camp": camp,
            "win": agent_observation.get("win"),
            "hero": self._summarize_hero(hero),
            "enemy_hero": self._summarize_hero(enemy_hero),
            "tower": self._summarize_unit(tower),
            "enemy_tower": self._summarize_unit(enemy_tower),
        }

    def _find_main_hero(self, heroes, player_id, camp):
        for hero in heroes:
            if player_id is not None and hero.get("runtime_id") == player_id:
                return hero
        for hero in heroes:
            if camp is not None and hero.get("camp") == camp:
                return hero
        return None

    def _find_enemy_hero(self, heroes, camp):
        for hero in heroes:
            if camp is not None and hero.get("camp") != camp:
                return hero
        return None

    def _find_tower(self, npcs, camp):
        for npc in npcs:
            if npc.get("sub_type") == 21 and npc.get("camp") == camp:
                return npc
        return None

    def _find_enemy_tower(self, npcs, camp):
        for npc in npcs:
            if npc.get("sub_type") == 21 and npc.get("camp") != camp:
                return npc
        return None

    def _summarize_hero(self, hero):
        if not hero:
            return None
        return {
            "runtime_id": hero.get("runtime_id"),
            "config_id": hero.get("config_id"),
            "camp": hero.get("camp"),
            "hp": hero.get("hp"),
            "max_hp": hero.get("max_hp"),
            "hp_rate": self._safe_rate(hero.get("hp"), hero.get("max_hp")),
            "level": hero.get("level"),
            "exp": hero.get("exp"),
            "money": hero.get("money"),
            "money_cnt": hero.get("money_cnt"),
            "kill_cnt": hero.get("kill_cnt"),
            "dead_cnt": hero.get("dead_cnt"),
            "total_hurt": hero.get("total_hurt"),
            "total_hurt_to_hero": hero.get("total_hurt_to_hero"),
            "total_be_hurt_by_hero": hero.get("total_be_hurt_by_hero"),
            "location": hero.get("location"),
        }

    def _summarize_unit(self, unit):
        if not unit:
            return None
        return {
            "runtime_id": unit.get("runtime_id"),
            "config_id": unit.get("config_id"),
            "camp": unit.get("camp"),
            "sub_type": unit.get("sub_type"),
            "hp": unit.get("hp"),
            "max_hp": unit.get("max_hp"),
            "hp_rate": self._safe_rate(unit.get("hp"), unit.get("max_hp")),
            "attack_target": unit.get("attack_target"),
            "location": unit.get("location"),
        }

    def _safe_rate(self, value, max_value):
        if value is None or not max_value:
            return None
        return value / max_value
