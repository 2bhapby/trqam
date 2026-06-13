#!/usr/bin/env bash
# Launch JupyterLab inside a persistent tmux session so the kernel keeps running
# after VSCode/SSH disconnects. Re-running just reports the existing session.
#
# The orchestrator kernel runs on CPU; the 8 GPU training subprocesses it spawns
# are independent and resumable, so training survives disconnects and restarts.
#
# Override defaults via env vars: TRQAM_TMUX_SESSION, TRQAM_CONDA_ENV,
# TRQAM_JUPYTER_PORT, TRQAM_JUPYTER_TOKEN.
set -euo pipefail

SESSION="${TRQAM_TMUX_SESSION:-trqam-jupyter}"
ENV_NAME="${TRQAM_CONDA_ENV:-trqam}"
PORT="${TRQAM_JUPYTER_PORT:-9000}"
TOKEN="${TRQAM_JUPYTER_TOKEN:-trqam}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Install it (e.g. 'conda install -c conda-forge tmux' or 'apt-get install tmux')." >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' is already running."
  echo "Attach with:  tmux attach -t $SESSION"
  exit 0
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [ -d "$HOME/miniconda3" ]; then
  CONDA_BASE="$HOME/miniconda3"
else
  echo "conda not found. Run 'bash setup_env.sh' first." >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -c "$REPO_DIR" \
  "source '$CONDA_BASE/etc/profile.d/conda.sh'; conda activate '$ENV_NAME'; \
   jupyter lab --no-browser --ip=127.0.0.1 --port=$PORT --ServerApp.token=$TOKEN --ServerApp.password= ; \
   echo '[jupyter exited - press enter to close]'; read"

cat <<EOF
JupyterLab started in tmux session '$SESSION' (port $PORT, env '$ENV_NAME').

Forward the port from your laptop:
  ssh -N -L $PORT:127.0.0.1:$PORT <user>@<server>

Then connect:
  - Browser:  http://127.0.0.1:$PORT/lab?token=$TOKEN
  - VSCode:   run "Jupyter: Specify Jupyter Server for Connections" and enter
              http://127.0.0.1:$PORT/?token=$TOKEN   (keeps the kernel in tmux, not in VSCode)

Useful:
  tmux attach -t $SESSION     # view the JupyterLab server log
  tmux kill-session -t $SESSION   # stop the server (training subprocesses keep running)
EOF
