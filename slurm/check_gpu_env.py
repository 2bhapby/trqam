import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from agents.trqam import TRQAMAgent, get_config
from envs.ogbench_utils import make_ogbench_env_and_datasets


def run(cmd):
    print(f"$ {' '.join(cmd)}", flush=True)
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"command failed: {exc}", flush=True)
        if hasattr(exc, "output") and exc.output:
            print(exc.output, flush=True)
        return
    print(out, flush=True)


def main():
    print("python:", sys.executable)
    print("cwd:", Path.cwd())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("XLA_PYTHON_CLIENT_PREALLOCATE:", os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"))

    if shutil.which("nvidia-smi"):
        run(["nvidia-smi"])
    else:
        print("nvidia-smi: not found")

    print("jax version:", jax.__version__)
    print("jax devices:", jax.devices())
    print("jax default backend:", jax.default_backend())

    x = jnp.ones((1024, 1024), dtype=jnp.float32)
    y = x @ x
    print("matmul sum:", float(jnp.sum(y)))

    env = make_ogbench_env_and_datasets(
        "cube-triple-play-singletask-task2-v0",
        env_only=True,
    )
    print("ogbench env observation shape:", env.observation_space.shape)
    print("ogbench env action shape:", env.action_space.shape)

    cfg = get_config()
    cfg["horizon_length"] = 5
    cfg["action_chunking"] = True
    cfg["bc_only"] = True
    agent = TRQAMAgent.create(
        0,
        jnp.zeros(env.observation_space.shape, dtype=jnp.float32),
        jnp.zeros(env.action_space.shape, dtype=jnp.float32),
        cfg,
    )
    print("TRQAM agent init ok, lambda:", float(agent.lam))


if __name__ == "__main__":
    main()
