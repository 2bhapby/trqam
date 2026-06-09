#!/bin/bash
#SBATCH --job-name=trqam-full-bc
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/trqam-full-bc-%j.out
#SBATCH --error=slurm_logs/trqam-full-bc-%j.err

set -euo pipefail

cd /mnt/lustre/slurm/users/sanghyeok/trqam
mkdir -p slurm_logs

export PYTHONPATH=/mnt/lustre/slurm/users/sanghyeok/trqam:${PYTHONPATH:-}
export WANDB_MODE=${WANDB_MODE:-disabled}
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHON=/home/sanghyeok/miniconda3/envs/trqam/bin/python

ENV_NAME=${ENV_NAME:-cube-triple-play-singletask-task2-v0}
SEED=${SEED:-10001}
SAVE_ROOT=${SAVE_ROOT:-exp_full}
RUN_GROUP=${RUN_GROUP:-bc_pretrain}
TAGS=${TAGS:-BC_FULL}
HORIZON_LENGTH=${HORIZON_LENGTH:-5}
WIDTH=${WIDTH:-512}
ACTOR_LAYER_NORM=${ACTOR_LAYER_NORM:-False}
FLOW_STEPS=${FLOW_STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_EPISODES=${EVAL_EPISODES:-50}
LOG_INTERVAL=${LOG_INTERVAL:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-50000}
SAVE_INTERVAL=${SAVE_INTERVAL:-50000}
DATASET_PROPORTION=${DATASET_PROPORTION:-1.0}

HIDDEN_DIMS="(${WIDTH},${WIDTH},${WIDTH},${WIDTH})"

DATASET_ARGS=()
if [[ -n "${OGBENCH_DATASET_DIR:-}" ]]; then
  DATASET_ARGS+=("--ogbench_dataset_dir=${OGBENCH_DATASET_DIR}")
  DATASET_ARGS+=("--dataset_replace_interval=${DATASET_REPLACE_INTERVAL:-1000}")
else
  DATASET_ARGS+=("--dataset_replace_interval=0")
fi

ARGS=(
  "--save_dir=${SAVE_ROOT%/}/"
  "--run_group=${RUN_GROUP}"
  "--agent=agents/trqam.py"
  "--tags=${TAGS}"
  "--seed=${SEED}"
  "--env_name=${ENV_NAME}"
  "--sparse=False"
  "--horizon_length=${HORIZON_LENGTH}"
  "--agent.action_chunking=True"
  "--bc_only=True"
  "--offline_steps=300000"
  "--online_steps=0"
  "--dataset_proportion=${DATASET_PROPORTION}"
  "--eval_interval=${EVAL_INTERVAL}"
  "--eval_episodes=${EVAL_EPISODES}"
  "--video_episodes=0"
  "--save_interval=${SAVE_INTERVAL}"
  "--log_interval=${LOG_INTERVAL}"
  "--agent.batch_size=${BATCH_SIZE}"
  "--agent.actor_hidden_dims=${HIDDEN_DIMS}"
  "--agent.value_hidden_dims=${HIDDEN_DIMS}"
  "--agent.actor_layer_norm=${ACTOR_LAYER_NORM}"
  "--agent.flow_steps=${FLOW_STEPS}"
  "--save_last_checkpoint=True"
)

"${PYTHON}" main.py "${ARGS[@]}" "${DATASET_ARGS[@]}"
"${PYTHON}" slurm/export_eval_report.py --latest-glob "${SAVE_ROOT%/}/*/${RUN_GROUP}/${ENV_NAME}/*"
