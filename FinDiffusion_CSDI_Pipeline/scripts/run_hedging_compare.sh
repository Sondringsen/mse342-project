#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mse342_matplotlib}"
THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"

python_is_usable() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

PARENT_VENV_PYTHON="${PROJECT_ROOT}/../venv/bin/python"
PIPELINE_VENV_PYTHON="${PROJECT_ROOT}/FinDiffusion_CSDI_Pipeline/.venv/bin/python"
if [ -z "${PYTHON:-}" ]; then
  if [ -x "${PIPELINE_VENV_PYTHON}" ] && python_is_usable "${PIPELINE_VENV_PYTHON}"; then
    PYTHON="${PIPELINE_VENV_PYTHON}"
  elif [ -x "${PARENT_VENV_PYTHON}" ] && python_is_usable "${PARENT_VENV_PYTHON}"; then
    PYTHON="${PARENT_VENV_PYTHON}"
  else
    PYTHON="python3"
  fi
fi

exec "${PYTHON}" FinDiffusion/scripts/hedging_from_predictions.py "$@"
