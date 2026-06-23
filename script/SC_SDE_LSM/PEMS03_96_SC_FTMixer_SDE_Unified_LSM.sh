#!/bin/bash
set -euo pipefail

DATA="PEMS"
ROOT="./dataset/PEMS/"
DATA_PATH="PEMS03.npz"
ENC_IN=358
BATCH=32
FREQ="t"

HIDDEN=256
GATE_INIT="2.0"
PHASE_MODE="position"
CYCLE_LEN=288
SLOTS=12
NUM_WORKERS=0

for pl in 12 96; do
  echo "=== Sanity PEMS03 pred_len=${pl} ==="
  python3 -u main_sde.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path "${ROOT}" --data_path "${DATA_PATH}" \
    --model_id "PEMS03_96_${pl}_LSM_Sanity_Off" \
    --model SC_FTMixer_SDE_Unified_LSM \
    --data "${DATA}" --features M \
    --seq_len 96 --label_len 48 --pred_len "${pl}" \
    --enc_in "${ENC_IN}" --d_model 32 --d_ff 64 --e_layers 1 --n_heads 1 --dropout 0 \
    --batch_size "${BATCH}" --learning_rate 0.005 --train_epochs 100 --patience 10 --lradj cosine_warmup \
    --freq "${FREQ}" --des "LSM_Sanity_Off_PEMS03" --itr 1 \
    --use_sde 1 \
    --sde_phase_mode "${PHASE_MODE}" --sde_cycle_len "${CYCLE_LEN}" --sde_slots_per_hour "${SLOTS}" \
    --sde_hidden "${HIDDEN}" --sde_rec_weight 0.25 --sde_spectral_weight 0.75 \
    --sde_calendar_gate_init "${GATE_INIT}" \
    --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 1 --fix_calendar_gate 0 \
    --use_sensor_mixer 0 \
    --num_workers "${NUM_WORKERS}"
done

DATA="PEMS"
ROOT="./dataset/PEMS/"
DATA_PATH="PEMS03.npz"
ENC_IN=358
BATCH=32
FREQ="t"

HIDDEN=256
GATE_INIT="1.0"
PHASE_MODE="position"
CYCLE_LEN=288
SLOTS=12
NUM_WORKERS=0

for pl in 12 96; do
  echo "=== LSM PEMS03 pred_len=${pl} ==="
  python3 -u main_sde.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path "${ROOT}" --data_path "${DATA_PATH}" \
    --model_id "PEMS03_96_${pl}_PemsTuneV1_P3_LSM_r4" \
    --model SC_FTMixer_SDE_Unified_LSM \
    --data "${DATA}" --features M \
    --seq_len 96 --label_len 48 --pred_len "${pl}" \
    --enc_in "${ENC_IN}" --d_model 32 --d_ff 64 --e_layers 1 --n_heads 1 --dropout 0 \
    --batch_size "${BATCH}" --learning_rate 0.005 --train_epochs 100 --patience 10 --lradj cosine_warmup \
    --freq "${FREQ}" --des "PemsTuneV1_P3_LSM_r4" --itr 1 \
    --use_sde 1 \
    --sde_phase_mode "${PHASE_MODE}" --sde_cycle_len "${CYCLE_LEN}" --sde_slots_per_hour "${SLOTS}" \
    --sde_hidden "${HIDDEN}" --sde_rec_weight 0.5 --sde_spectral_weight 0.5 \
    --sde_calendar_gate_init "${GATE_INIT}" \
    --use_global_sde 1 --use_calendar_sde 1 --use_dynamic_filter 1 --fix_calendar_gate 0 \
    --use_sensor_mixer 1 \
    --sensor_rank 4 \
    --sensor_alpha_init -6.0 \
    --sensor_max_scale 0.1 \
    --num_workers "${NUM_WORKERS}"
done

echo "=== PEMS03 LSM sanity check complete ==="