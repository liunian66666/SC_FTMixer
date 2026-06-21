#!/bin/bash
set -e

cd /home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer
#  
for dataset in ETTh1 ETTh2  ETTm1  ETTm2 ; do
  if [[ "$dataset" == ETTm* ]]; then
    phase_mode=day_slot
    cycle_len=96
    slots_per_hour=4
    freq=t
  else
    phase_mode=hour
    cycle_len=24
    slots_per_hour=1
    freq=h
  fi
   if [[ "$dataset" == ETTm2 ]]; then
    sde_hidden=256
  else
    sde_hidden=192
  fi
  #  sde_hiddenETTm2 sde_hidden=256其他192
  for pred_len in 96 192 336 720; do
    echo "====================================="
    echo "Running $dataset | model=SC_FTMixer_SDE_Unified | pred_len=$pred_len"
    echo "====================================="
    python3 -u main_sde.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ${dataset}.csv \
    --model_id ${dataset}_96_${pred_len}_SC_FTMixer_SDE_Unified \
    --model SC_FTMixer_SDE_Unified \
    --data ${dataset} \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --freq ${freq} \
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
    --sde_phase_mode ${phase_mode} \
    --sde_cycle_len ${cycle_len} \
    --sde_slots_per_hour ${slots_per_hour} \
    --sde_calendar_gate_init 2.0 \
    --sde_hidden ${sde_hidden} \
    --sde_rec_weight 0.25 \
    --sde_spectral_weight 0.75
  done
done
