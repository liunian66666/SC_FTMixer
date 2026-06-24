#!/usr/bin/env bash
set -euo pipefail

# ECL local search for SC_FTMixer_SDE_Unified.
# Fixed: learning_rate=0.005, calendar_gate_init=2.0.
# The five configurations below exclude the combinations already present in ECL.txt.
#
# Optional overrides:
#   ROOT_DIR=/path/to/SC_FTMixer GPU=0 PROMOTE_MSE=0.136 bash ECL_target_search_lr005.sh

ROOT_DIR=$(pwd)
GPU="${GPU:-0}"
PROMOTE_MSE="${PROMOTE_MSE:-0.136}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/ecl_target_search_lr005}"

cd "${ROOT_DIR}"
mkdir -p "${RESULT_DIR}/logs"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TMPDIR="${TMPDIR:-/tmp}"

# name hidden rec_weight spectral_weight batch_size
#  "D1_H288_R25S75_B128 288 0.25 0.75 128"
# "D2_H320_R25S75_B128 320 0.25 0.75 128"
# "D3_H256_R15S85_B128 256 0.15 0.85 128"
# "D4_H256_R35S65_B128 256 0.35 0.65 128"
# "D6_H256_R25S75_B32  256 0.25 0.75 32"
# "D7_H256_R35S65_B32  256 0.35 0.65 32"
CONFIGS=(
  "D8_H256_R25S75_B8  256 0.25 0.75 8"
  
  
)

SUMMARY="${RESULT_DIR}/summary.csv"
echo "pred_len,name,hidden,rec_weight,spectral_weight,batch_size,mse,mae" > "${SUMMARY}"
LAST_MSE=""

run_one() {
  local pred_len="$1"
  local name="$2"
  local hidden="$3"
  local rec_weight="$4"
  local spectral_weight="$5"
  local batch_size="$6"
  local model_id="ECL_96_${pred_len}_TargetSearch_${name}_LR5"
  local des="ECL_TargetSearch_${pred_len}_${name}_LR5"
  local log_file="${RESULT_DIR}/logs/${model_id}.log"

  echo
  echo "============================================================"
  echo "Running ${model_id}"
  echo "hidden=${hidden}, rec=${rec_weight}, spectral=${spectral_weight}, batch=${batch_size}"
  echo "============================================================"

  python3 -u main_sde.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id "${model_id}" \
    --model SC_FTMixer_SDE_Unified \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "${pred_len}" \
    --enc_in 321 \
    --d_model 32 \
    --d_ff 64 \
    --e_layers 1 \
    --n_heads 1 \
    --batch_size "${batch_size}" \
    --learning_rate 0.005 \
    --train_epochs 100 \
    --patience 10 \
    --lradj cosine_warmup \
    --freq h \
    --des "${des}" \
    --itr 1 \
    --use_sde 1 \
    --sde_phase_mode hour_week \
    --sde_cycle_len 168 \
    --sde_slots_per_hour 1 \
    --sde_calendar_gate_init 2.0 \
    --sde_hidden "${hidden}" \
    --sde_rec_weight "${rec_weight}" \
    --sde_spectral_weight "${spectral_weight}" \
    2>&1 | tee "${log_file}"

  local metric_line
  local mse
  local mae
  metric_line="$(grep -E 'mse:[[:space:]]*[0-9.]+' "${log_file}" | tail -n 1 || true)"
  mse="$(printf '%s\n' "${metric_line}" | sed -nE 's/.*mse:[[:space:]]*([0-9.eE+-]+).*/\1/p')"
  mae="$(printf '%s\n' "${metric_line}" | sed -nE 's/.*mae:[[:space:]]*([0-9.eE+-]+).*/\1/p')"

  if [[ -z "${mse}" || -z "${mae}" ]]; then
    echo "ERROR: metrics were not found in ${log_file}" >&2
    return 1
  fi

  echo "${pred_len},${name},${hidden},${rec_weight},${spectral_weight},${batch_size},${mse},${mae}" \
    >> "${SUMMARY}"
  LAST_MSE="${mse}"
}

declare -a PROMOTED=()

echo "Stage 1: searching five new configurations on pred_len=96"
for config in "${CONFIGS[@]}"; do
  read -r name hidden rec_weight spectral_weight batch_size <<< "${config}"
  run_one 96 "${name}" "${hidden}" "${rec_weight}" "${spectral_weight}" "${batch_size}"
  mse="${LAST_MSE}"

  if awk -v mse="${mse}" -v threshold="${PROMOTE_MSE}" \
    'BEGIN { exit !(mse <= threshold) }'; then
    PROMOTED+=("${config}")
    echo "Promoted to pred_len=192: ${name}, MSE=${mse}"
  else
    echo "Not promoted: ${name}, MSE=${mse} > ${PROMOTE_MSE}"
  fi
done

if ((${#PROMOTED[@]} == 0)); then
  echo
  echo "No 96-step configuration reached MSE <= ${PROMOTE_MSE}."
  echo "Stopping before pred_len=192."
else
  echo
  echo "Stage 2: running promoted configurations on pred_len=192"
  for config in "${PROMOTED[@]}"; do
    read -r name hidden rec_weight spectral_weight batch_size <<< "${config}"
    run_one 192 "${name}" "${hidden}" "${rec_weight}" "${spectral_weight}" "${batch_size}"
  done
fi

echo
echo "Search complete."
echo "Summary: ${SUMMARY}"
echo "Logs: ${RESULT_DIR}/logs"
column -s, -t "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
