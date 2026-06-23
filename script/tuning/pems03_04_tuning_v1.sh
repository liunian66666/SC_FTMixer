#!/usr/bin/env bash
set -euo pipefail

# cd /home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified
export TMPDIR=/tmp

# PEMS03/04 Full-model tuning.
#
# P1: R50S50, gate=2, lr=0.005, hidden=256
# P2: R25S75, gate=1, lr=0.005, hidden=256
# P3: R50S50, gate=1, lr=0.005, hidden=256
# P4: R50S50, gate=1, lr=0.003, hidden=192
#
# Search matrix:
#   PEMS03/PEMS04 x pred_len 12/96 x P1/P2/P3/P4 = 16 runs.
#
# Existing baseline:
#   PEMS03-12: MSE=0.065383, MAE=0.167103
#   PEMS03-96: MSE=0.210645, MAE=0.302546
#   PEMS04-12: MSE=0.073018, MAE=0.172891
#   PEMS04-96: MSE=0.191402, MAE=0.288138
#
# This script runs sequentially and does not use tmux.

run_one() {
  local dataset="$1"
  local enc_in="$2"
  local pred_len="$3"
  local tag="$4"
  local rec_weight="$5"
  local spectral_weight="$6"
  local gate_init="$7"
  local learning_rate="$8"
  local hidden="$9"

  python3 -u main_sde.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path "${dataset}.npz" \
    --model_id "${dataset}_96_${pred_len}_PemsTuneV1_${tag}" \
    --model SC_FTMixer_SDE_Unified \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "${pred_len}" \
    --freq t \
    --enc_in "${enc_in}" \
    --d_model 32 \
    --d_ff 64 \
    --e_layers 1 \
    --n_heads 1 \
    --dropout 0 \
    --batch_size 32 \
    --learning_rate "${learning_rate}" \
    --train_epochs 100 \
    --patience 10 \
    --lradj cosine_warmup \
    --des "PemsTuneV1_${tag}" \
    --itr 1 \
    --num_workers 2 \
    --use_sde 1 \
    --sde_phase_mode position \
    --sde_cycle_len 288 \
    --sde_slots_per_hour 12 \
    --sde_calendar_gate_init "${gate_init}" \
    --sde_hidden "${hidden}" \
    --sde_rec_weight "${rec_weight}" \
    --sde_spectral_weight "${spectral_weight}" \
    --use_global_sde 1 \
    --use_calendar_sde 1 \
    --use_dynamic_filter 1
}

run_set() {
  local dataset="$1"
  local enc_in="$2"
  local pred_len="$3"

  run_one "${dataset}" "${enc_in}" "${pred_len}" P1 0.50 0.50 2.0 0.005 256
  run_one "${dataset}" "${enc_in}" "${pred_len}" P2 0.25 0.75 1.0 0.005 256
  run_one "${dataset}" "${enc_in}" "${pred_len}" P3 0.50 0.50 1.0 0.005 256
  run_one "${dataset}" "${enc_in}" "${pred_len}" P4 0.50 0.50 1.0 0.003 192
}

run_set PEMS03 358 12
run_set PEMS03 358 96
run_set PEMS04 307 12
run_set PEMS04 307 96
