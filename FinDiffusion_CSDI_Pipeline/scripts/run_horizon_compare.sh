#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

PARENT_VENV_PYTHON="${PROJECT_ROOT}/../venv/bin/python"
PIPELINE_VENV_PYTHON="${PROJECT_ROOT}/FinDiffusion_CSDI_Pipeline/.venv/bin/python"
if [ -z "${PYTHON:-}" ]; then
  if [ -x "${PARENT_VENV_PYTHON}" ]; then
    PYTHON="${PARENT_VENV_PYTHON}"
  else
    PYTHON="${PIPELINE_VENV_PYTHON}"
  fi
fi

exec "${PYTHON}" FinDiffusion_CSDI_Pipeline/scripts/compare_horizon_runs.py "$@"
