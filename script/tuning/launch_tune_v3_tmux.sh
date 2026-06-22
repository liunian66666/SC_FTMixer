#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified"
SESSION="tune_v3"
cd "${ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists"
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n worker0 \
  "cd '${ROOT}' && CUDA_VISIBLE_DEVICES=0 bash script/tuning/tune_v3_confirm.sh 0 2; exec bash"
tmux new-window -t "${SESSION}" -n worker1 \
  "cd '${ROOT}' && CUDA_VISIBLE_DEVICES=0 bash script/tuning/tune_v3_confirm.sh 1 2; exec bash"
tmux new-window -t "${SESSION}" -n monitor \
  "cd '${ROOT}' && watch -n 10 'echo === GPU ===; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; echo; echo === TRAINING ===; pgrep -af \"main_sde.py.*TuneV3\" || true; echo; echo === STATUS ===; cat results/tuning/TuneV3/status/worker*.current 2>/dev/null || true; echo; python3 script/tuning/summarize_tuning_v3.py'; exec bash"
tmux select-window -t "${SESSION}:monitor"
echo "Started tmux session: ${SESSION}"
