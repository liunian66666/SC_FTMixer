#!/bin/bash
set -e

# cd /home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer
export TMPDIR=/tmp

MODEL=SC_FTMixer_SDE_Unified
COMMON="--task_name long_term_forecast --is_training 1 --model ${MODEL} --features M --seq_len 96 --label_len 48 --pred_len 96 --learning_rate 0.005 --train_epochs 100 --patience 10 --lradj cosine_warmup --dropout 0 --itr 1 --use_sde 0 --sde_hidden 192 --sde_rec_weight 0.25 --sde_spectral_weight 0.75 --sde_calendar_gate_init 2.0"

python3 -u main_sde.py ${COMMON} \
  --model_id Solar_96_96_SDE_Unified_off \
  --root_path ./dataset/Solar/ --data_path solar_AL.txt \
  --data SolarCalendar --enc_in 137 --batch_size 256 --num_workers 0 \
  --freq t --sde_phase_mode position --sde_cycle_len 144 \
  --sde_slots_per_hour 6 --des SDE_Unified_off_diagnostic

python3 -u main_sde.py ${COMMON} \
  --model_id Traffic_96_96_SDE_Unified_off \
  --root_path ./dataset/traffic/ --data_path traffic.csv \
  --data custom --enc_in 862 --batch_size 64 --num_workers 0 \
  --freq h --sde_phase_mode hour_week --sde_cycle_len 168 \
  --sde_slots_per_hour 1 --des SDE_Unified_off_diagnostic

python3 -u main_sde.py ${COMMON} \
  --model_id ETTh1_96_96_SDE_Unified_off \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --data ETTh1 --enc_in 7 --batch_size 1024 --num_workers 0 \
  --freq h --sde_phase_mode hour --sde_cycle_len 24 \
  --sde_slots_per_hour 1 --des SDE_Unified_off_diagnostic
