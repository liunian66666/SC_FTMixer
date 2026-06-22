#!/bin/bash
set -e
cd /home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified

# =========================================================================
# SDE_Unified Ablation — Weather (10min, 21 vars, freq=t, phase=day_slot, cycle=144)
# =========================================================================

DATA="custom"
ROOT="./dataset/weather/"
DATA_PATH="weather.csv"
ENC_IN=21
BATCH=1024
FREQ="t"
PHASE_MODE="day_slot"
CYCLE_LEN=144
SLOTS=6
HIDDEN=256
GATE_INIT="2.0"
NUM_WORKERS=0

run_ablation() {
  local tag=$1; shift
  for pl in 96 720; do
    echo "=== [${tag}] Weather pred_len=${pl} ==="
    python3 -u main_sde.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "${ROOT}" \
      --data_path "${DATA_PATH}" \
      --model_id "Weather_96_${pl}_Abl_${tag}" \
      --model SC_FTMixer_SDE_Unified \
      --data "${DATA}" \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "${pl}" \
      --enc_in "${ENC_IN}" \
      --d_model 32 --d_ff 64 --e_layers 1 --n_heads 1 --dropout 0 \
      --batch_size "${BATCH}" \
      --learning_rate 0.005 --train_epochs 100 --patience 10 --lradj cosine_warmup \
      --freq "${FREQ}" \
      --des "Abl_${tag}" \
      --itr 1 \
      --use_sde 1 \
      --sde_phase_mode "${PHASE_MODE}" --sde_cycle_len "${CYCLE_LEN}" --sde_slots_per_hour "${SLOTS}" \
      --sde_hidden "${HIDDEN}" --sde_rec_weight 0.25 --sde_spectral_weight 0.75 \
      --num_workers "${NUM_WORKERS}" \
      "$@"
  done
}

# --- 1. Full ---
run_ablation "Full" \
  --sde_calendar_gate_init "${GATE_INIT}" \
  --fix_calendar_gate 0 \
  --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 1

# --- 2. No SDE ---
run_ablation "NoSDE" \
  --use_sde 0 \
  --sde_calendar_gate_init "${GATE_INIT}" \
  --fix_calendar_gate 0 \
  --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 1

# --- 3. Global Only ---
run_ablation "GlobalOnly" \
  --sde_calendar_gate_init "${GATE_INIT}" \
  --fix_calendar_gate 0 \
  --use_global_sde 1 --use_calendar_sde 0 --use_dynamic_filter 1

# --- 4. Calendar Only ---
run_ablation "CalendarOnly" \
  --sde_calendar_gate_init "${GATE_INIT}" \
  --fix_calendar_gate 0 \
  --use_global_sde 0 --use_calendar_sde 1 --use_dynamic_filter 1

# --- 5. No Gate (fix_calendar_gate=1, gate strictly fixed to 1.0) ---
run_ablation "NoGate" \
  --sde_calendar_gate_init "${GATE_INIT}" \
  --fix_calendar_gate 1 \
  --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 1

# --- 6. No Dynamic Filter ---
run_ablation "NoDynFilt" \
  --sde_calendar_gate_init "${GATE_INIT}" \
  --fix_calendar_gate 0 \
  --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 0

echo "=== Weather ablation complete ==="
