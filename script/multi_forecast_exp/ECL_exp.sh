#!/usr/bin/env bash
set -euo pipefail
# cd /home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified

# # pred_len=96, MSE=0.136225, MAE=0.229712
# python3 -u main_sde.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_96_96_ExpBest_B16 \
#   --model SC_FTMixer_SDE_Unified \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --freq h \
#   --enc_in 321 \
#   --d_model 32 \
#   --d_ff 64 \
#   --e_layers 1 \
#   --n_heads 1 \
#   --batch_size 16 \
#   --learning_rate 0.005 \
#   --train_epochs 100 \
#   --patience 10 \
#   --lradj cosine_warmup \
#   --des ExpBest_ECL_96_B16_H256_G2_R25S75 \
#   --itr 1 \
#   --num_workers 0 \
#   --use_sde 1 \
#   --sde_phase_mode hour_week \
#   --sde_cycle_len 168 \
#   --sde_slots_per_hour 1 \
#   --sde_calendar_gate_init 2.0 \
#   --sde_hidden 256 \
#   --sde_rec_weight 0.25 \
#   --sde_spectral_weight 0.75

# pred_len=192, MSE=0.154745, MAE=0.247229
python3 -u main_sde.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192_ExpBest \
  --model SC_FTMixer_SDE_Unified \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --freq h \
  --enc_in 321 \
  --d_model 32 \
  --d_ff 64 \
  --e_layers 1 \
  --n_heads 1 \
  --batch_size 16 \
  --learning_rate 0.005 \
  --train_epochs 100 \
  --patience 10 \
  --lradj cosine_warmup \
  --des ExpBest_ECL_192 \
  --itr 1 \
  --num_workers 2 \
  --use_sde 1 \
  --sde_phase_mode hour_week \
  --sde_cycle_len 168 \
  --sde_slots_per_hour 1 \
  --sde_calendar_gate_init 2.0 \
  --sde_hidden 256 \
  --sde_rec_weight 0.25 \
  --sde_spectral_weight 0.75

# pred_len=336, MSE=0.171729, MAE=0.264817
python3 -u main_sde.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336_ExpBest \
  --model SC_FTMixer_SDE_Unified \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --freq h \
  --enc_in 321 \
  --d_model 16 \
  --d_ff 64 \
  --e_layers 1 \
  --n_heads 1 \
  --batch_size 32  \
  --learning_rate 0.005 \
  --train_epochs 100 \
  --patience 10 \
  --lradj cosine_warmup \
  --des ExpBest_ECL_336 \
  --itr 1 \
  --num_workers 2 \
  --use_sde 1 \
  --sde_phase_mode hour_week \
  --sde_cycle_len 168 \
  --sde_slots_per_hour 1 \
  --sde_calendar_gate_init 2.0 \
  --sde_hidden 256 \
  --sde_rec_weight 0.25 \
  --sde_spectral_weight 0.75

# pred_len=720, MSE=0.209431, MAE=0.296847
python3 -u main_sde.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720_ExpBest \
  --model SC_FTMixer_SDE_Unified \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --freq h \
  --enc_in 321 \
  --d_model 32 \
  --d_ff 64 \
  --e_layers 1 \
  --n_heads 1 \
  --batch_size 16 \
  --learning_rate 0.005 \
  --train_epochs 100 \
  --patience 10 \
  --lradj cosine_warmup \
  --des ExpBest_ECL_720 \
  --itr 1 \
  --num_workers 2 \
  --use_sde 1 \
  --sde_phase_mode hour_week \
  --sde_cycle_len 168 \
  --sde_slots_per_hour 1 \
  --sde_calendar_gate_init 2.0 \
  --sde_hidden 320 \
  --sde_rec_weight 0.25 \
  --sde_spectral_weight 0.75
