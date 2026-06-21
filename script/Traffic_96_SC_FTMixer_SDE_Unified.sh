#!/bin/bash
set -e
cd /home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer
export TMPDIR=/tmp

PRED_LIST="${PRED_LIST:-}"
#  
for pred_len in 96 192 336 720 ; do
  python3 -u main_sde.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic_96_${pred_len}_SDE_Unified_week168 \
    --model SC_FTMixer_SDE_Unified \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --enc_in 862 \
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
    --freq h \
    --des SDE_Unified_week168_h192 \
    --itr 1 \
    --use_sde 1 \
    --sde_phase_mode hour_week \
    --sde_cycle_len 168 \
    --sde_slots_per_hour 1 \
    --sde_calendar_gate_init 2.0 \
    --sde_hidden 256 \
    --sde_rec_weight 0.25 \
    --sde_spectral_weight 0.75
done
