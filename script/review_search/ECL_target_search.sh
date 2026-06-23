#!/usr/bin/env bash
set -euo pipefail

# Copy this file to the project root or override ROOT explicitly:
# ROOT=/path/to/SC_FTMixer_SDE_Unified bash ECL_target_search.sh
# ROOT="${ROOT:-/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified}"
# cd "${ROOT}"

COMMON=(
  --task_name long_term_forecast
  --is_training 1
  --root_path ./dataset/electricity/
  --data_path electricity.csv
  --model SC_FTMixer_SDE_Unified
  --data custom
  --features M
  --seq_len 96
  --label_len 48
  --freq h
  --enc_in 321
  --d_model 32
  --d_ff 64
  --e_layers 1
  --n_heads 1
  --dropout 0
  --train_epochs 100
  --patience 10
  --lradj cosine_warmup
  --itr 1
  --num_workers 2
  --use_sde 1
  --sde_phase_mode hour_week
  --sde_cycle_len 168
  --sde_slots_per_hour 1
  --use_global_sde 1
  --use_calendar_sde 1
  --use_dynamic_filter 1
  --fix_calendar_gate 0
  --visualize 0
)

run_case() {
  local pred_len="$1"
  local tag="$2"
  local hidden="$3"
  local gate="$4"
  local rec="$5"
  local spectral="$6"
  local lr="$7"
  local batch="$8"

  python3 -u main_sde.py \
    "${COMMON[@]}" \
    --model_id "ECL_96_${pred_len}_TargetSearch_${tag}" \
    --pred_len "${pred_len}" \
    --batch_size "${batch}" \
    --learning_rate "${lr}" \
    --des "ECL_TargetSearch_${pred_len}_${tag}" \
    --sde_hidden "${hidden}" \
    --sde_calendar_gate_init "${gate}" \
    --sde_rec_weight "${rec}" \
    --sde_spectral_weight "${spectral}"
}

# ECL-96 target: MSE < 0.135 and MAE < 0.229
run_case 96 A1_H192_G1_R25S75_LR5 192 1.0 0.25 0.75 0.005 128
run_case 96 A2_H256_G2_R25S75_LR5 256 2.0 0.25 0.75 0.005 128
run_case 96 A3_H192_G2_R50S50_LR5 192 2.0 0.50 0.50 0.005 128
run_case 96 A4_H192_G2_R25S75_LR3 192 2.0 0.25 0.75 0.003 128

# ECL-192 target: MSE < 0.151 and MAE < 0.246
run_case 192 B1_H256_G2_R25S75_LR5 256 2.0 0.25 0.75 0.005 128
run_case 192 B2_H192_G3_R25S75_LR5 192 3.0 0.25 0.75 0.005 128
run_case 192 B3_H256_G3_R25S75_LR5 256 3.0 0.25 0.75 0.005 128
run_case 192 B4_H192_G2_R50S50_LR5 192 2.0 0.50 0.50 0.005 128
run_case 192 B5_H192_G2_R25S75_LR3 192 2.0 0.25 0.75 0.003 128

# ECL-720 target: MSE < 0.204 and MAE < 0.294
run_case 720 C1_H320_G3_R25S75_LR3 320 3.0 0.25 0.75 0.003 64
run_case 720 C2_H256_G3_R25S75_LR5 256 3.0 0.25 0.75 0.005 64
run_case 720 C3_H320_G2_R25S75_LR5 320 2.0 0.25 0.75 0.005 64
run_case 720 C4_H320_G3_R50S50_LR5 320 3.0 0.50 0.50 0.005 64
run_case 720 C5_H256_G3_R25S75_LR3 256 3.0 0.25 0.75 0.003 64
