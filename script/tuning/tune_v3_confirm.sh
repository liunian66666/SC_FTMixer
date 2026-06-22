#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified"
cd "${ROOT}"

WORKER_ID="${1:?usage: $0 WORKER_ID [NUM_TUNE_WORKERS]}"
NUM_TUNE_WORKERS="${2:-2}"
RUN_TAG="TuneV3"
LOG_DIR="results/tuning/${RUN_TAG}/logs"
STATUS_DIR="results/tuning/${RUN_TAG}/status"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

# ETTh1: confirm R50S50 at intermediate horizons.
# ETTh2: test whether gate=1 generalizes beyond horizon 720.
# dataset horizon rec spectral gate tag
jobs=(
  "ETTh1 192 0.50 0.50 2.0 MidConfirm"
  "ETTh1 336 0.50 0.50 2.0 MidConfirm"
  "ETTh2 96  0.25 0.75 1.0 Gate1Confirm"
  "ETTh2 192 0.25 0.75 1.0 Gate1Confirm"
  "ETTh2 336 0.25 0.75 1.0 Gate1Confirm"
)

run_job() {
  local dataset="$1" pred_len="$2" rec="$3" spectral="$4" gate="$5" tag="$6"
  local ratio gate_tag model_id des setting metrics log_file
  [[ "${rec}" == "0.50" ]] && ratio="R50S50" || ratio="R25S75"
  [[ "${gate}" == "1.0" ]] && gate_tag="G1" || gate_tag="G2"
  model_id="${dataset}_96_${pred_len}_${RUN_TAG}_${tag}_${ratio}_LR005_${gate_tag}_H192"
  des="${RUN_TAG}_${tag}_${ratio}_LR005_${gate_tag}_H192"
  setting="long_term_forecast_${model_id}_SC_FTMixer_SDE_Unified_sl96_ll48_pl${pred_len}_dm32_nh1_el1_${des}_0_dp0.0"
  metrics="results/log_mse_mae/npy_results/${setting}/metrics.npy"
  log_file="${LOG_DIR}/${model_id}.log"

  if [[ -f "${metrics}" ]]; then
    echo "[$(date '+%F %T')] SKIP completed ${model_id}" | tee -a "${log_file}"
    return
  fi

  echo "[$(date '+%F %T')] START worker=${WORKER_ID} ${model_id}" | tee "${log_file}"
  python3 -u main_sde.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/ETT-small/ --data_path "${dataset}.csv" \
    --model_id "${model_id}" --model SC_FTMixer_SDE_Unified \
    --data "${dataset}" --features M \
    --seq_len 96 --label_len 48 --pred_len "${pred_len}" --freq h \
    --enc_in 7 --d_model 32 --d_ff 64 --e_layers 1 --n_heads 1 --dropout 0 \
    --batch_size 1024 --learning_rate 0.005 \
    --train_epochs 100 --patience 10 --lradj cosine_warmup \
    --des "${des}" --itr 1 --num_workers 2 \
    --use_sde 1 --sde_phase_mode hour --sde_cycle_len 24 \
    --sde_slots_per_hour 1 --sde_calendar_gate_init "${gate}" \
    --sde_hidden 192 --sde_rec_weight "${rec}" \
    --sde_spectral_weight "${spectral}" \
    2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] DONE worker=${WORKER_ID} ${model_id}" | tee -a "${log_file}"
}

for i in "${!jobs[@]}"; do
  (( i % NUM_TUNE_WORKERS == WORKER_ID )) || continue
  read -r dataset pred_len rec spectral gate tag <<< "${jobs[$i]}"
  echo "${dataset} ${pred_len} ${tag}" > "${STATUS_DIR}/worker${WORKER_ID}.current"
  run_job "${dataset}" "${pred_len}" "${rec}" "${spectral}" "${gate}" "${tag}"
done

echo "complete" > "${STATUS_DIR}/worker${WORKER_ID}.current"
echo "[$(date '+%F %T')] WORKER ${WORKER_ID} COMPLETE"
