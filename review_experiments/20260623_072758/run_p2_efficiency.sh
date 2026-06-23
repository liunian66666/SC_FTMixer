#!/usr/bin/env bash
set -euo pipefail

REVIEW_ROOT="${1:?review root required}"
OUT="${REVIEW_ROOT}/p2_efficiency"
mkdir -p "${OUT}"

MODELS=("SC-FSD" "XLinear" "TEFN" "iTransformer" "PatchTST" "DLinear")
for model in "${MODELS[@]}"; do
  safe="${model//-/_}"
  log="${OUT}/${safe}.log"
  json="${OUT}/${safe}.json"
  if [[ -f "${json}" ]]; then
    echo "SKIP ${model}"
    continue
  fi
  echo "START ${model} $(date --iso-8601=seconds)" | tee -a "${OUT}/runner.log"
  python3 -u "${REVIEW_ROOT}/p2_efficiency_benchmark.py" \
    --model "${model}" \
    --output "${json}" \
    --batch-size 16 \
    --warmup 20 \
    --repeats 100 \
    --num-workers 2 > "${log}" 2>&1
  echo "DONE ${model} $(date --iso-8601=seconds)" | tee -a "${OUT}/runner.log"
done

python3 "${REVIEW_ROOT}/aggregate_p2.py" "${OUT}"
