#!/usr/bin/env bash
set -euo pipefail

cd /home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified
export TMPDIR=/tmp

# PEMS03 diagnostic ablation
#
# Existing Full results:
# pred_len=12, MSE=0.065383, MAE=0.167103
# pred_len=96, MSE=0.210645, MAE=0.302546
#
# Interpretation:
# - GlobalOnly better than Full: calendar residual may be harmful.
# - NoDynFilt worse than Full: dynamic filter is effective.
#
# This script runs sequentially and does not use tmux.

run_one() {
  local pred_len="$1"
  local tag="$2"
  local use_calendar="$3"
  local use_dynamic_filter="$4"

  python3 -u main_sde.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id "PEMS03_96_${pred_len}_PemsDiagV1_${tag}" \
    --model SC_FTMixer_SDE_Unified \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "${pred_len}" \
    --freq t \
    --enc_in 358 \
    --d_model 32 \
    --d_ff 64 \
    --e_layers 1 \
    --n_heads 1 \
    --dropout 0 \
    --batch_size 32 \
    --learning_rate 0.005 \
    --train_epochs 100 \
    --patience 10 \
    --lradj cosine_warmup \
    --des "PemsDiagV1_${tag}" \
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
    --use_calendar_sde "${use_calendar}" \
    --use_dynamic_filter "${use_dynamic_filter}"
}

# GlobalOnly: remove calendar residual.
run_one 12 GlobalOnly 0 1
run_one 96 GlobalOnly 0 1

# NoDynFilt: retain global/calendar priors but remove dynamic filtering.
run_one 12 NoDynFilt 1 0
run_one 96 NoDynFilt 1 0
