#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified"
cd "${ROOT}"

WORKER_ID="${1:?usage: $0 WORKER_ID [NUM_TUNE_WORKERS]}"
NUM_TUNE_WORKERS="${2:-2}"
RUN_TAG="TuneWE1"
LOG_DIR="results/tuning/${RUN_TAG}/logs"
STATUS_DIR="results/tuning/${RUN_TAG}/status"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

# dataset horizon tag rec spectral gate lr hidden
jobs=(
  "Weather 96  W1 0.50 0.50 2.0 0.005 256"
  "Weather 720 W1 0.50 0.50 2.0 0.005 256"
  "Weather 96  W2 0.25 0.75 1.0 0.005 256"
  "Weather 720 W2 0.25 0.75 1.0 0.005 256"
  "Weather 96  W3 0.25 0.75 3.0 0.005 256"
  "Weather 720 W3 0.25 0.75 3.0 0.005 256"
  "Weather 96  W4 0.25 0.75 2.0 0.003 256"
  "Weather 720 W4 0.25 0.75 2.0 0.003 256"
  "ECL 336 E1 0.25 0.75 2.0 0.005 256"
  "ECL 720 E1 0.25 0.75 2.0 0.005 256"
  "ECL 336 E2 0.25 0.75 3.0 0.005 192"
  "ECL 720 E2 0.25 0.75 3.0 0.005 192"
  "ECL 336 E3 0.25 0.75 3.0 0.005 256"
  "ECL 720 E3 0.25 0.75 3.0 0.005 256"
  "ECL 336 E4 0.50 0.50 2.0 0.005 192"
  "ECL 720 E4 0.50 0.50 2.0 0.005 192"
  "ECL 336 E5 0.25 0.75 2.0 0.003 192"
  "ECL 720 E5 0.25 0.75 2.0 0.003 192"
)

ratio_tag() {
  [[ "$1/$2" == "0.50/0.50" ]] && echo "R50S50" || echo "R25S75"
}

lr_tag() {
  [[ "$1" == "0.003" ]] && echo "LR003" || echo "LR005"
}

gate_tag() {
  case "$1" in
    1.0) echo "G1" ;;
    2.0) echo "G2" ;;
    3.0) echo "G3" ;;
  esac
}

run_job() {
  local dataset="$1" pred_len="$2" tag="$3" rec="$4" spectral="$5"
  local gate="$6" lr="$7" hidden="$8"
  local root data_path enc_in freq phase cycle slots batch
  local ratio lr_name gate_name model_id des setting metrics log_file

  if [[ "${dataset}" == "Weather" ]]; then
    root="./dataset/weather/"
    data_path="weather.csv"
    enc_in=21
    freq=t
    phase=day_slot
    cycle=144
    slots=6
    batch=1024
  else
    root="./dataset/electricity/"
    data_path="electricity.csv"
    enc_in=321
    freq=h
    phase=hour_week
    cycle=168
    slots=1
    if [[ "${pred_len}" == "720" ]]; then batch=64; else batch=128; fi
  fi

  ratio="$(ratio_tag "${rec}" "${spectral}")"
  lr_name="$(lr_tag "${lr}")"
  gate_name="$(gate_tag "${gate}")"
  model_id="${dataset}_96_${pred_len}_${RUN_TAG}_${tag}_${ratio}_${lr_name}_${gate_name}_H${hidden}"
  des="${RUN_TAG}_${tag}_${ratio}_${lr_name}_${gate_name}_H${hidden}"
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
    --root_path "${root}" --data_path "${data_path}" \
    --model_id "${model_id}" --model SC_FTMixer_SDE_Unified \
    --data custom --features M \
    --seq_len 96 --label_len 48 --pred_len "${pred_len}" \
    --freq "${freq}" --enc_in "${enc_in}" \
    --d_model 32 --d_ff 64 --e_layers 1 --n_heads 1 --dropout 0 \
    --batch_size "${batch}" --learning_rate "${lr}" \
    --train_epochs 100 --patience 10 --lradj cosine_warmup \
    --des "${des}" --itr 1 --num_workers 2 \
    --use_sde 1 --sde_phase_mode "${phase}" --sde_cycle_len "${cycle}" \
    --sde_slots_per_hour "${slots}" --sde_calendar_gate_init "${gate}" \
    --sde_hidden "${hidden}" --sde_rec_weight "${rec}" \
    --sde_spectral_weight "${spectral}" \
    2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] DONE worker=${WORKER_ID} ${model_id}" | tee -a "${log_file}"
}

for i in "${!jobs[@]}"; do
  (( i % NUM_TUNE_WORKERS == WORKER_ID )) || continue
  read -r dataset pred_len tag rec spectral gate lr hidden <<< "${jobs[$i]}"
  echo "${dataset} ${pred_len} ${tag}" > "${STATUS_DIR}/worker${WORKER_ID}.current"
  run_job "${dataset}" "${pred_len}" "${tag}" "${rec}" "${spectral}" \
    "${gate}" "${lr}" "${hidden}"
done

echo "complete" > "${STATUS_DIR}/worker${WORKER_ID}.current"
echo "[$(date '+%F %T')] WORKER ${WORKER_ID} COMPLETE"
