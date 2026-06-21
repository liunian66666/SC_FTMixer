#!/bin/bash
set -e

cd /home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer


# ETTh2 ETTm1 ETTm2 192 336 720


for dataset in  ETTh1 ; do
  for pred_len in 96 ; do
    echo "====================================="
    echo "Running $dataset | model=SC_FTMixer_SDE_Unified | pred_len=$pred_len"
    echo "====================================="
    python3 -u main_sde.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_96_SC_FTMixer_SDE_Unified \
      --model SC_FTMixer_SDE_Unified \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --enc_in 7 \
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
      --des SC_FTMixer_SDE_Unified \
      --itr 1 \
      --use_sde 1 \
      --sde_phase_mode hour \
      --sde_cycle_len 24 \
      --sde_slots_per_hour 1 \
      --sde_calendar_gate_init 2.0 \
      --sde_hidden 192 \
      --sde_rec_weight 0.25 \
      --sde_spectral_weight 0.75
  done
done