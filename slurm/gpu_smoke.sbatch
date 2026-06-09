#!/usr/bin/env bash
#SBATCH --job-name=trqam-gpu-smoke
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd /mnt/lustre/slurm/users/sanghyeok/trqam

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/mnt/lustre/slurm/users/sanghyeok/trqam:${PYTHONPATH:-}

/home/sanghyeok/miniconda3/envs/trqam/bin/python slurm/check_gpu_env.py
