#!/bin/bash
set -e
# cd /home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer
export TMPDIR=/tmp



for pred_len in 96 192 336 720; do
  python3 -u main_sde.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --model_id Solar_96_${pred_len}_SDE_Unified_pos144 \
    --model SC_FTMixer_SDE_Unified \
    --data SolarCalendar \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --enc_in 137 \
    --d_model 32 \
    --d_ff 64 \
    --e_layers 1 \
    --n_heads 1 \
    --dropout 0 \
    --batch_size 256 \
    --learning_rate 0.005 \
    --train_epochs 100 \
    --patience 10 \
    --lradj cosine_warmup \
    --freq t \
    --des SDE_Unified_pos144_h192 \
    --itr 1 \
    --use_sde 1 \
    --sde_phase_mode position \
    --sde_cycle_len 144 \
    --sde_slots_per_hour 6 \
    --sde_calendar_gate_init 2.0 \
    --sde_hidden 192 \
    --sde_rec_weight 0.25 \
    --sde_spectral_weight 0.75 \
    --num_workers 0
done
