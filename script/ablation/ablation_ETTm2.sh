#!/bin/bash
set -e
# cd /home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified

DATA="ETTm2"
ROOT="./dataset/ETT-small/"
DATA_PATH="ETTm2.csv"
ENC_IN=7; BATCH=1024; FREQ="t"
PHASE_MODE="day_slot"; CYCLE_LEN=96; SLOTS=4
HIDDEN=256; GATE_INIT="2.0"; NUM_WORKERS=10

run_one() {
  local tag=$1 pl=$2; shift 2
  echo "=== [${tag}] ${DATA} pred_len=${pl} ==="
  python3 -u main_sde.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path "${ROOT}" --data_path "${DATA_PATH}" \
    --model_id "${DATA}_96_${pl}_Abl_${tag}" \
    --model SC_FTMixer_SDE_Unified --data "${DATA}" --features M \
    --seq_len 96 --label_len 48 --pred_len "${pl}" \
    --enc_in "${ENC_IN}" --d_model 32 --d_ff 64 --e_layers 1 --n_heads 1 --dropout 0 \
    --batch_size "${BATCH}" --learning_rate 0.005 --train_epochs 100 --patience 10 --lradj cosine_warmup \
    --freq "${FREQ}" --des "Abl_${tag}" --itr 1 --use_sde 1 \
    --sde_phase_mode "${PHASE_MODE}" --sde_cycle_len "${CYCLE_LEN}" --sde_slots_per_hour "${SLOTS}" \
    --sde_hidden "${HIDDEN}" --sde_rec_weight 0.25 --sde_spectral_weight 0.75 \
    --num_workers "${NUM_WORKERS}" "$@"
}

run_ablation_set() {
  local pl=$1
  echo "" ; echo "========== ${DATA} pred_len=${pl} ==========" ; echo ""
  run_one "Full"        "${pl}" --sde_calendar_gate_init "${GATE_INIT}" --fix_calendar_gate 0 --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 1
  run_one "NoSDE"       "${pl}" --use_sde 0 --sde_calendar_gate_init "${GATE_INIT}" --fix_calendar_gate 0 --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 1
  run_one "GlobalOnly"  "${pl}" --sde_calendar_gate_init "${GATE_INIT}" --fix_calendar_gate 0 --use_global_sde 1 --use_calendar_sde 0 --use_dynamic_filter 1
  run_one "CalendarOnly" "${pl}" --sde_calendar_gate_init "${GATE_INIT}" --fix_calendar_gate 0 --use_global_sde 0 --use_calendar_sde 1 --use_dynamic_filter 1
  run_one "NoGate"      "${pl}" --sde_calendar_gate_init "${GATE_INIT}" --fix_calendar_gate 1 --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 1
  run_one "NoDynFilt"   "${pl}" --sde_calendar_gate_init "${GATE_INIT}" --fix_calendar_gate 0 --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 0
}

run_ablation_set 96
run_ablation_set 720
echo "=== ETTm2 ablation complete ==="
