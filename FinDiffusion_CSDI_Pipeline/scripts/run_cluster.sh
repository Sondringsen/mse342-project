#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PARTITION="${PARTITION:-gpu-turing}"
RUN_NAME="${RUN_NAME:-cluster_$(date +%Y%m%d_%H%M%S)}"
MODELS="${MODELS:-findiffusion csdi}"
GPUS_PER_MODEL="${GPUS_PER_MODEL:-1}"
CPUS_PER_MODEL="${CPUS_PER_MODEL:-8}"
LOG_DIR="${PROJECT_ROOT}/FinDiffusion_CSDI_Pipeline/outputs/${RUN_NAME}/logs"
mkdir -p "${LOG_DIR}"

echo "Run name: ${RUN_NAME}"
echo "Models: ${MODELS}"
echo "GPUs per model: ${GPUS_PER_MODEL}"
echo "CPUs per model: ${CPUS_PER_MODEL}"
echo "Logs: ${LOG_DIR}"

pids=()
for model in ${MODELS}; do
  echo "Launching ${model} on ${GPUS_PER_MODEL} GPU(s)"
  (
    export MODEL="${model}"
    export RUN_NAME="${RUN_NAME}"
    srun --partition="${PARTITION}" \
      --gres="gpu:${GPUS_PER_MODEL}" \
      --cpus-per-task="${CPUS_PER_MODEL}" \
      "${PROJECT_ROOT}/FinDiffusion_CSDI_Pipeline/scripts/run_one_gpu.sh" "$@"
  ) > "${LOG_DIR}/${model}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [ "${failed}" -ne 0 ]; then
  echo "At least one model job failed; skipping summary generation."
  exit "${failed}"
fi

"${PROJECT_ROOT}/FinDiffusion_CSDI_Pipeline/.venv/bin/python" \
  FinDiffusion_CSDI_Pipeline/pipeline/summarize_results.py \
  --run-dir "${PROJECT_ROOT}/FinDiffusion_CSDI_Pipeline/outputs/${RUN_NAME}" || failed=1

exit "${failed}"
