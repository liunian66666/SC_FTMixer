#!/bin/bash
set -e
# cd /home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer
export TMPDIR=/tmp

# 96 192 336
for pred_len in  720 ; do
  python3 -u main_sde.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_${pred_len}_SDE_Unified_day144 \
    --model SC_FTMixer_SDE_Unified \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --enc_in 21 \
    --d_model 32 \
    --d_ff 64 \
    --e_layers 1 \
    --n_heads 1 \
    --dropout 0 \
    --batch_size 1024 \
    --learning_rate 0.005 \
    --train_epochs 100 \
    --patience 10 \
    --lradj cosine_warmup \
    --freq t \
    --des SDE_Unified_day144_h192 \
    --itr 1 \
    --use_sde 1 \
    --sde_phase_mode day_slot \
    --sde_cycle_len 144 \
    --sde_slots_per_hour 6 \
    --sde_calendar_gate_init 2.0 \
    --sde_hidden 256 \
    --sde_rec_weight 0.25 \
    --sde_spectral_weight 0.75
done
