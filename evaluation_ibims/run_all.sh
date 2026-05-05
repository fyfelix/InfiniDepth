#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
        PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

usage() {
    cat <<'EOF'
Usage:
  ./evaluation_ibims/run_all.sh [model_path=checkpoints/depth/infinidepth_depthsensor.ckpt] [extra run_all.py args...]

Environment overrides:
  PYTHON_BIN    Python executable. Default: .venv/bin/python if present, otherwise python3

This wrapper is fixed to InfiniDepth_DepthSensor for iBims metric depth evaluation.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 0 && "${1}" != --* ]]; then
    model_path="${1}"
    shift
else
    model_path="checkpoints/depth/infinidepth_depthsensor.ckpt"
fi

cd "${PROJECT_ROOT}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/infinidepth-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_all.py" \
    --model-path "${model_path}" \
    "$@"
