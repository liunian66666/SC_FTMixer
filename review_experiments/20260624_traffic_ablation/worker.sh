#!/usr/bin/env bash
set -euo pipefail

REVIEW_ROOT="${REVIEW_ROOT:?REVIEW_ROOT required}"
WORKER_ID="${1:?worker id required}"
WORKER_COUNT="${2:?worker count required}"
QUEUE="${REVIEW_ROOT}/p0_jobs.tsv"
LOG="${REVIEW_ROOT}/worker${WORKER_ID}.log"

index=0
tail -n +2 "${QUEUE}" | while IFS=$'\t' read -r job dataset horizon variant seed status; do
  if (( index % WORKER_COUNT != WORKER_ID )); then
    index=$((index + 1))
    continue
  fi
  index=$((index + 1))
  if [[ -f "${REVIEW_ROOT}/jobs/${job}/metrics.json" ]]; then
    echo "SKIP ${job}" | tee -a "${LOG}"
    continue
  fi
  echo "START ${job} $(date --iso-8601=seconds)" | tee -a "${LOG}"
  if REVIEW_ROOT="${REVIEW_ROOT}" bash "${REVIEW_ROOT}/run_job.sh" \
      "${job}" "${dataset}" "${horizon}" "${variant}" "${seed}"; then
    echo "DONE ${job} $(date --iso-8601=seconds)" | tee -a "${LOG}"
  else
    echo "FAILED ${job} $(date --iso-8601=seconds)" | tee -a "${LOG}"
  fi
done
