#!/bin/bash
set -euo pipefail
cd /home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified

ABL_VER="AblFixStatic"

DATA="custom"
ROOT="./dataset/traffic/"
DATA_PATH="traffic.csv"
ENC_IN=862; BATCH=32; FREQ="h"
PHASE_MODE="hour_week"; CYCLE_LEN=168; SLOTS=1
HIDDEN=256; GATE_INIT="2.0"; NUM_WORKERS=0

run_one() {
  local tag=$1 pl=$2; shift 2
  echo "=== [${tag}] Traffic pred_len=${pl} ==="
  python3 -u main_sde.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path "${ROOT}" --data_path "${DATA_PATH}" \
    --model_id "Traffic_96_${pl}_${ABL_VER}_${tag}_h${HIDDEN}" \
    --model SC_FTMixer_SDE_Unified --data "${DATA}" --features M \
    --seq_len 96 --label_len 48 --pred_len "${pl}" \
    --enc_in "${ENC_IN}" --d_model 32 --d_ff 64 --e_layers 1 --n_heads 1 --dropout 0 \
    --batch_size "${BATCH}" --learning_rate 0.005 --train_epochs 100 --patience 10 --lradj cosine_warmup \
    --freq "${FREQ}" --des "${ABL_VER}_${tag}_h${HIDDEN}" --itr 1 \
    --sde_phase_mode "${PHASE_MODE}" --sde_cycle_len "${CYCLE_LEN}" --sde_slots_per_hour "${SLOTS}" \
    --sde_hidden "${HIDDEN}" --sde_rec_weight 0.25 --sde_spectral_weight 0.75 \
    --num_workers "${NUM_WORKERS}" "$@"
}

run_ablation_set() {
  local pl=$1
  echo "" ; echo "========== Traffic pred_len=${pl} ==========" ; echo ""

  run_one "Full" "${pl}" \
    --use_sde 1 --sde_calendar_gate_init "${GATE_INIT}" --fix_calendar_gate 0 \
    --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 1

  run_one "NoStatic" "${pl}" \
    --use_sde 0 --sde_calendar_gate_init "${GATE_INIT}" --fix_calendar_gate 0 \
    --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 1

  run_one "GlobalOnly" "${pl}" \
    --use_sde 1 --sde_calendar_gate_init "${GATE_INIT}" --fix_calendar_gate 0 \
    --use_global_sde 1 --use_calendar_sde 0 --use_dynamic_filter 1

  run_one "CalendarOnly" "${pl}" \
    --use_sde 1 --sde_calendar_gate_init "${GATE_INIT}" --fix_calendar_gate 0 \
    --use_global_sde 0 --use_calendar_sde 1 --use_dynamic_filter 1

  run_one "NoGate" "${pl}" \
    --use_sde 1 --sde_calendar_gate_init "${GATE_INIT}" --fix_calendar_gate 1 \
    --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 1

  run_one "NoDynFilt" "${pl}" \
    --use_sde 1 --sde_calendar_gate_init "${GATE_INIT}" --fix_calendar_gate 0 \
    --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 0
}

run_ablation_set 96
run_ablation_set 720   # uncomment after 96 verified
echo "=== Traffic ablation complete ==="
