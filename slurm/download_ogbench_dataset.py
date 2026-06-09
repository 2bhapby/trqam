import os
import sys
from pathlib import Path

from envs.env_utils import make_env_and_datasets


def main():
    env_name = sys.argv[1] if len(sys.argv) > 1 else "cube-triple-play-singletask-task2-v0"
    print("python:", sys.executable)
    print("cwd:", Path.cwd())
    print("env_name:", env_name)
    print("HOME:", os.environ.get("HOME"))
    print("MUJOCO_GL:", os.environ.get("MUJOCO_GL"))

    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(env_name)
    print("train size:", train_dataset.size)
    print("train keys:", sorted(train_dataset.keys()))
    print("observation shape:", train_dataset["observations"].shape)
    print("action shape:", train_dataset["actions"].shape)
    if val_dataset is not None:
        print("val size:", val_dataset.size)
        print("val observation shape:", val_dataset["observations"].shape)
        print("val action shape:", val_dataset["actions"].shape)
    print("env observation shape:", env.observation_space.shape)
    print("env action shape:", env.action_space.shape)
    print("dataset download/load ok")


if __name__ == "__main__":
    main()
