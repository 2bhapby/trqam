#!/usr/bin/env bash
#SBATCH --job-name=trqam-data-dl
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd /mnt/lustre/slurm/users/sanghyeok/trqam

export MUJOCO_GL=egl
export PYTHONPATH=/mnt/lustre/slurm/users/sanghyeok/trqam:${PYTHONPATH:-}

/home/sanghyeok/miniconda3/envs/trqam/bin/python slurm/download_ogbench_dataset.py cube-triple-play-singletask-task2-v0
