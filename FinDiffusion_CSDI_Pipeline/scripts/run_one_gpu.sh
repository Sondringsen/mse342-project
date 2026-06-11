#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

load_course_cuda_module() {
  if ml --terse avail course/cme213/nvhpc/24.1 2>&1 | grep -q "course/cme213/nvhpc/24.1"; then
    ml course/cme213/nvhpc/24.1
  fi
}

if command -v ml >/dev/null 2>&1; then
  load_course_cuda_module
elif [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
  load_course_cuda_module
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mse342_matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PARENT_VENV_PYTHON="${PROJECT_ROOT}/../venv/bin/python"
PIPELINE_VENV_PYTHON="${PROJECT_ROOT}/FinDiffusion_CSDI_Pipeline/.venv/bin/python"
if [ -z "${PYTHON:-}" ]; then
  if [ -x "${PARENT_VENV_PYTHON}" ]; then
    PYTHON="${PARENT_VENV_PYTHON}"
  else
    PYTHON="${PIPELINE_VENV_PYTHON}"
  fi
fi
MODEL="${MODEL:-both}"
RUN_NAME="${RUN_NAME:-cluster_$(date +%Y%m%d_%H%M%S)}"

exec "${PYTHON}" FinDiffusion_CSDI_Pipeline/run_pipeline.py \
  --model "${MODEL}" \
  --run-name "${RUN_NAME}" \
  "$@"
