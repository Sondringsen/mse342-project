#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if command -v ml >/dev/null 2>&1; then
  ml course/cme213/nvhpc/24.1
elif [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
  module load course/cme213/nvhpc/24.1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mse342_matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PYTHON="${PROJECT_ROOT}/FinDiffusion_CSDI_Pipeline/.venv/bin/python"
MODEL="${MODEL:-both}"
RUN_NAME="${RUN_NAME:-cluster_$(date +%Y%m%d_%H%M%S)}"

exec "${PYTHON}" FinDiffusion_CSDI_Pipeline/run_pipeline.py \
  --model "${MODEL}" \
  --run-name "${RUN_NAME}" \
  "$@"
