#!/usr/bin/env bash
# Create the TRQAM conda environment with GPU JAX and register the Jupyter kernel.
# Mirrors the install steps in README.md. Usage: bash setup_env.sh [env_name]
set -euo pipefail

ENV_NAME="${1:-trqam}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate conda; bootstrap Miniconda into ~/miniconda3 if it is not available.
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [ -d "$HOME/miniconda3" ]; then
  CONDA_BASE="$HOME/miniconda3"
else
  echo "Installing Miniconda into ~/miniconda3 ..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  CONDA_BASE="$HOME/miniconda3"
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Accept Anaconda channel Terms of Service (no-op on conda versions without `tos`).
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists; reusing it."
else
  conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"

cd "$REPO_DIR"
pip install -r requirements.txt
pip install "jax[cuda12]==0.6.2" jaxlib==0.6.2 jupyterlab ipykernel
python -m ipykernel install --user --name trqam-conda --display-name "TRQAM Conda (GPU)"

echo
echo "Environment '$ENV_NAME' ready."
echo "  - Kernel registered as: TRQAM Conda (GPU)"
echo "  - For Robomimic envs, also follow README installation steps 2-4."
echo "  - Verify GPU JAX:  python -c \"import os; os.environ.pop('JAX_PLATFORMS', None); import jax; print(jax.devices())\""
echo "  - Next: bash run_jupyter_tmux.sh"
