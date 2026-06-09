#!/usr/bin/env bash
#SBATCH --job-name=trqam-ogb-main
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd /mnt/lustre/slurm/users/sanghyeok/trqam

export WANDB_MODE=disabled
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/mnt/lustre/slurm/users/sanghyeok/trqam

/home/sanghyeok/miniconda3/envs/trqam/bin/python main.py --save_dir=exp_smoke/ --run_group=ogbench_smoke_clean --agent=agents/trqam.py --tags=OGB_SMOKE_CLEAN --seed=0 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True --bc_only=True --offline_steps=1 --online_steps=0 --dataset_proportion=0.001 --eval_interval=0 --eval_episodes=1 --video_episodes=0 --save_interval=0 --log_interval=1 --agent.batch_size=8 --agent.actor_hidden_dims="(64,64)" --agent.value_hidden_dims="(64,64)" --agent.flow_steps=2
