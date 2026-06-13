#!/usr/bin/env bash
# Terminal live status of all training runs. Reads only from disk, so it works
# regardless of the notebook front-end (browser/VSCode) and survives disconnects.
# Run in a separate tmux pane:  bash watch_runs.sh [interval_seconds]
INTERVAL="${1:-15}"
PY="${TRQAM_PY:-$HOME/miniconda3/envs/trqam/bin/python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

while true; do
  clear
  echo "TRQAM runs - $(date '+%Y-%m-%d %H:%M:%S')  (refresh ${INTERVAL}s, Ctrl-C to stop)"
  "$PY" - <<'EOF'
import glob, json, os
import pandas as pd
rows = []
for hb_path in sorted(glob.glob('exp_full/**/heartbeat.json', recursive=True)):
    run_dir = os.path.dirname(hb_path)
    try:
        hb = json.load(open(hb_path))
    except Exception:
        hb = {}
    sr = eval_step = None
    eval_csv = os.path.join(run_dir, 'eval.csv')
    if os.path.exists(eval_csv):
        try:
            e = pd.read_csv(eval_csv)
            if 'success' in e.columns and len(e):
                sr = round(float(pd.to_numeric(e['success'], errors='coerce').dropna().iloc[-1]), 3)
                eval_step = int(pd.to_numeric(e['step'], errors='coerce').dropna().iloc[-1])
        except Exception:
            pass
    ckpts = sorted(glob.glob(os.path.join(run_dir, 'params_*.pkl')))
    rows.append({
        'run': os.path.basename(run_dir),
        'stage': hb.get('stage'), 'step': hb.get('step'), 'total': hb.get('total'),
        'eval_step': eval_step, 'SR': sr,
        'ckpt': os.path.basename(ckpts[-1]) if ckpts else None,
        'done': os.path.exists(os.path.join(run_dir, 'done.tk')),
        'updated': hb.get('updated'),
    })
print(pd.DataFrame(rows).to_string(index=False) if rows else 'No runs have started yet.')
EOF
  sleep "$INTERVAL"
done
