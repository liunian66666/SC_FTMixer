#!/usr/bin/env bash
set -euo pipefail

# cd /home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified
export TMPDIR=/tmp

# Dataset_PEMS derives a daily phase index from the absolute sample position.
# PEMS uses five-minute intervals, so one daily cycle contains 288 slots.
#
# Standard PEMS forecasting horizons: 12, 24, 48, 96.

run_pems() {
  local dataset="$1"
  local enc_in="$2"
  local batch_size="$3"
  local pred_len="$4"

  python3 -u main_sde.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path "${dataset}.npz" \
    --model_id "${dataset}_96_${pred_len}_SC_FTMixer_SDE_Unified" \
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
    --batch_size "${batch_size}" \
    --learning_rate 0.005 \
    --train_epochs 100 \
    --patience 10 \
    --lradj cosine_warmup \
    --des "SC_FTMixer_SDE_Unified_${dataset}" \
    --itr 1 \
    --num_workers 2 \
    --use_sde 1 \
    --sde_phase_mode position \
    --sde_cycle_len 288 \
    --sde_slots_per_hour 12 \
    --sde_calendar_gate_init 2.0 \
    --sde_hidden 256 \
    --sde_rec_weight 0.25 \
    --sde_spectral_weight 0.75 \
    --use_global_sde 1 \
    --use_calendar_sde 1 \
    --use_dynamic_filter 1
}

# PEMS03: 358 sensors
#dataset   enc_in  batch  pred_len
run_pems PEMS03 358 64 12
# run_pems PEMS03 358 32 24
# run_pems PEMS03 358 32 48
run_pems PEMS03 358 64 96

# PEMS04: 307 sensors
#dataset   enc_in  batch  pred_len
run_pems PEMS04 307 64 12
# run_pems PEMS04 307 32 24
# run_pems PEMS04 307 32 48
run_pems PEMS04 307 64 96

# # PEMS07: 883 sensors
# run_pems PEMS07 883 16 12
# run_pems PEMS07 883 16 24
# run_pems PEMS07 883 16 48
# run_pems PEMS07 883 16 96

# # PEMS08: 170 sensors
# #dataset   enc_in  batch  pred_len
# run_pems PEMS08 170 64 12
# run_pems PEMS08 170 64 24
# run_pems PEMS08 170 64 48
# run_pems PEMS08 170 64 96
