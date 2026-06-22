# Tencent Arena Training Log Collection Skill

Use this local project skill when collecting Tencent Arena training monitor data for this repository and writing factual step records under the log directory specified by the user.

## Scope

- Source page: Tencent Arena training monitor.
- Required browser surface: `chrome:control-chrome`, because the page depends on the user's logged-in Chrome session.
- Output directory: the user-specified directory, for example `logs/v1.1/` or `logs/v1.2/`.
- Output filename pattern: `step-<target-or-final-step>.md`.
- Output content: training parameters and metric values only. Do not include interpretation, recommendations, or analysis in step files.

## Output Directory Rule

- Always use the record directory stated by the user for the current training run.
- If the user says the current run is `v1.2`, write records under `logs/v1.2/`.
- Do not infer the output directory from previous runs such as `logs/v1.1/`.
- If the user gives a direct path, use that path exactly when it is inside the project workspace.
- If no output directory is stated, ask the user before collecting or writing records.

## Monitor URL

Current experiment URL:

```text
https://tencentarena.com/p/v5/exp/monitor?domain_id=2383&exp_id=15823&task_uuid=cd7b9eb6-7fe9-48b3-8a1b-471df5df5a7b&task_id=221342&platform=course
```

## Step Sampling Rule

- Record one file every 1000 target steps.
- Use the nearest real `train_global_step` available in the logs.
- Also record the final actual step as `step-<actual_train_global_step>.md`.
- In `## Sampling`, set:
  - `target_step`: requested target step.
  - `actual_train_global_step`: real step from `training_metrics basic`.
  - `step_delta`: `actual_train_global_step - target_step`.

## Chrome Workflow

1. Open or claim the monitor URL with `chrome:control-chrome`.
2. Confirm the page status and visible time range from the monitor header.
3. Use the time range picker to narrow the monitor window around the target step.
4. Switch to `训练日志`.
5. Search `training_metrics`.
6. Parse complete metric groups that share the same timestamp:
   - `aisrv training_metrics env common_ai is {...}`
   - `aisrv training_metrics env selfplay is {...}`
   - `aisrv training_metrics algorithm is {...}`
   - `aisrv training_metrics basic is {...}`
7. Select the record whose `basic.train_global_step` is nearest to the target step.
8. Switch to `监控总览`.
9. Narrow the same time window.
10. Scroll to `算法指标` and locate the `奖励分解` chart.
11. Hover along the chart x-axis and parse the tooltip nearest to the selected log timestamp.
12. Write or update the corresponding `step-*.md` file in the user-specified output directory.
13. Validate all required sections and key fields are present.

## Time Window Strategy

The log list is virtualized; broad scrolling may miss older records. Prefer time-window sampling:

- For each target step, estimate a timestamp from nearby checkpoint/log records.
- Set a narrow range, usually 3 to 10 minutes.
- Search `training_metrics` inside that range.
- If no complete group appears, widen the time range and retry.

Checkpoint lines such as `model.ckpt-14000.pkl successfully` are useful anchors for locating target windows, but the step record must use `training_metrics basic`, not checkpoint text.

## Metric Parsing

Parse Python-style dict strings into structured data by converting single quotes to JSON quotes and handling `True`, `False`, and `None` if present.

Only accept a timestamp record when all four metric groups are present:

```text
env common_ai
env selfplay
algorithm
basic
```

Required fields by group:

- `basic`: `train_global_step`, `sample_production_and_consumption_ratio`, `episode_cnt`, `sample_receive_cnt`, `predict_succ_cnt`, `load_model_succ_cnt`
- `common_ai`: `win_rate`, `self_tower_hp`, `enemy_tower_hp`, `frame`, `kill`, `death`, `money_per_frame`, `hurt_to_hero`, `hurt_by_hero`
- `selfplay`: same fields as `common_ai`
- `algorithm`: `reward`, `total_loss`, `policy_loss`, `value_loss`, `entropy_loss`

## Reward Detail Tooltip

Use the `奖励分解` chart under `算法指标`.

Hover multiple x-axis positions in the same narrowed time range. Parse the tooltip with:

- timestamp line, stored as `source_time`
- `运行时长`, stored as `runtime`
- `tower_hp`
- `tower_destroy`
- `hp`
- `money`
- `exp`
- `kill`
- `death`
- `forward`

Choose the tooltip whose `source_time` is closest to the selected log record timestamp. Do not invent missing values.

## Record File Format

Keep this exact section order:

```markdown
# Training Record

## Sampling

- target_step:
- actual_train_global_step:
- step_delta:

## Task

- experiment_name:
- status:
- task_id:
- experiment_version:
- algorithm:
- training_mode:
- latest_log_time:

## User Config

- opponent_agent:
- eval_interval:
- eval_opponent_type:
- blue_camp_hero_id:
- blue_camp_select_skill:
- red_camp_hero_id:
- red_camp_select_skill:

## Basic Metrics

- train_global_step:
- sample_production_and_consumption_ratio:
- episode_cnt:
- sample_receive_cnt:
- predict_succ_cnt:
- load_model_succ_cnt:

## Environment Metrics: Common AI

- win_rate:
- self_tower_hp:
- enemy_tower_hp:
- frame:
- kill:
- death:
- money_per_frame:
- hurt_to_hero:
- hurt_by_hero:

## Environment Metrics: Selfplay

- win_rate:
- self_tower_hp:
- enemy_tower_hp:
- frame:
- kill:
- death:
- money_per_frame:
- hurt_to_hero:
- hurt_by_hero:

## Algorithm Metrics

- reward:
- total_loss:
- policy_loss:
- value_loss:
- entropy_loss:

## Reward Detail Metrics

- source_time:
- runtime:
- tower_hp:
- tower_destroy:
- hp:
- money:
- exp:
- kill:
- death:
- forward:
```

## Status Values

Use factual page status values normalized for records:

- `running`: page shows `进行中`.
- `auto_released`: page shows `自动释放`.
- If another status appears, record the page status in lowercase snake case when possible.

## Current Fixed Task Metadata

For this experiment, use these values unless the page shows a later factual correction:

- `experiment_name`: `lmf-dev-v1_1`
- `task_id`: `221342`
- `experiment_version`: `V64.3.1`
- `algorithm`: `PPO`
- `training_mode`: `distributed`
- `opponent_agent`: `selfplay`
- `eval_interval`: `10`
- `eval_opponent_type`: `common_ai`
- `blue_camp_hero_id`: `199`
- `blue_camp_select_skill`: `80107`
- `red_camp_hero_id`: `133`
- `red_camp_select_skill`: `80110`

## Validation

After writing records, run checks similar to:

```sh
find <output-directory> -maxdepth 1 -type f -name 'step-*.md' | sort
rg -n "^## (Sampling|Basic Metrics|Environment Metrics: Common AI|Environment Metrics: Selfplay|Algorithm Metrics|Reward Detail Metrics)|^- train_global_step:|^- reward:|^- total_loss:|^- source_time:|^- forward:" <output-directory>/step-*.md
```

Confirm that each new file contains all required sections and that `Reward Detail Metrics` has all fields.
