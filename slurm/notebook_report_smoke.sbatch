#!/bin/bash
#SBATCH --job-name=trqam-nb-report
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/trqam-nb-report-%j.out
#SBATCH --error=slurm_logs/trqam-nb-report-%j.err

set -euo pipefail

cd /mnt/lustre/slurm/users/sanghyeok/trqam
mkdir -p slurm_logs

export PYTHONPATH=/mnt/lustre/slurm/users/sanghyeok/trqam:${PYTHONPATH:-}
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TRQAM_NOTEBOOK_PROFILE=smoke

/home/sanghyeok/miniconda3/envs/trqam/bin/python slurm/execute_notebook_report.py
