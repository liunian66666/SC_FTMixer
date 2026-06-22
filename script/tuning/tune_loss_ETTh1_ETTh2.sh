#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified"
cd "${ROOT}"

WORKER_ID="${1:?usage: $0 WORKER_ID [NUM_WORKERS]}"
NUM_TUNE_WORKERS="${2:-2}"
RUN_TAG="TuneV1"
LOG_DIR="results/tuning/${RUN_TAG}/logs"
STATUS_DIR="results/tuning/${RUN_TAG}/status"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

jobs=(
  "ETTh1 96 0.50 0.50 R50S50"
  "ETTh1 96 0.75 0.25 R75S25"
  "ETTh1 96 1.00 0.00 R100S0"
  "ETTh1 720 0.50 0.50 R50S50"
  "ETTh1 720 0.75 0.25 R75S25"
  "ETTh1 720 1.00 0.00 R100S0"
  "ETTh2 96 0.50 0.50 R50S50"
  "ETTh2 96 0.75 0.25 R75S25"
  "ETTh2 96 1.00 0.00 R100S0"
  "ETTh2 720 0.50 0.50 R50S50"
  "ETTh2 720 0.75 0.25 R75S25"
  "ETTh2 720 1.00 0.00 R100S0"
)

run_job() {
  local dataset="$1"
  local pred_len="$2"
  local rec="$3"
  local spectral="$4"
  local ratio_tag="$5"
  local model_id="${dataset}_96_${pred_len}_${RUN_TAG}_${ratio_tag}_LR005_G2_H192"
  local des="${RUN_TAG}_${ratio_tag}_LR005_G2_H192"
  local setting="long_term_forecast_${model_id}_SC_FTMixer_SDE_Unified_sl96_ll48_pl${pred_len}_dm32_nh1_el1_${des}_0_dp0.0"
  local metrics="results/log_mse_mae/npy_results/${setting}/metrics.npy"
  local log_file="${LOG_DIR}/${model_id}.log"

  if [[ -f "${metrics}" ]]; then
    echo "[$(date '+%F %T')] SKIP completed ${model_id}" | tee -a "${log_file}"
    return
  fi

  echo "[$(date '+%F %T')] START worker=${WORKER_ID} ${model_id}" | tee "${log_file}"
  python3 -u main_sde.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path "${dataset}.csv" \
    --model_id "${model_id}" \
    --model SC_FTMixer_SDE_Unified \
    --data "${dataset}" \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "${pred_len}" \
    --freq h \
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
    --des "${des}" \
    --itr 1 \
    --num_workers 2 \
    --use_sde 1 \
    --sde_phase_mode hour \
    --sde_cycle_len 24 \
    --sde_slots_per_hour 1 \
    --sde_calendar_gate_init 2.0 \
    --sde_hidden 192 \
    --sde_rec_weight "${rec}" \
    --sde_spectral_weight "${spectral}" \
    2>&1 | tee -a "${log_file}"

  echo "[$(date '+%F %T')] DONE worker=${WORKER_ID} ${model_id}" | tee -a "${log_file}"
}

for i in "${!jobs[@]}"; do
  if (( i % NUM_TUNE_WORKERS != WORKER_ID )); then
    continue
  fi
  read -r dataset pred_len rec spectral ratio_tag <<< "${jobs[$i]}"
  echo "${dataset} ${pred_len} ${ratio_tag}" > "${STATUS_DIR}/worker${WORKER_ID}.current"
  run_job "${dataset}" "${pred_len}" "${rec}" "${spectral}" "${ratio_tag}"
done

echo "complete" > "${STATUS_DIR}/worker${WORKER_ID}.current"
echo "[$(date '+%F %T')] WORKER ${WORKER_ID} COMPLETE"
