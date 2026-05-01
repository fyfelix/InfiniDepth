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
  ./evaluation/run_eval.sh [model_path=ckpts/infinidepth_depthsensor.ckpt] [raw_type=d435] [encoder=vitl16] [cleanup_npy=false]

Environment overrides:
  DATASET_PATH          HAMMER JSONL path. Default: data/HAMMER/test.jsonl
  OUTPUT_DIR            Base output directory. Each run writes to OUTPUT_DIR/YYYY-mm-dd_HH-MM-SS. Default: evaluation/output
  INPUT_SIZE            InfiniDepth input size as HxW. Default: 768x1024
  BATCH_SIZE            Recorded for compatibility; adapter runs sample-by-sample. Default: 1
  NUM_WORKERS           Recorded for compatibility; adapter uses a single-process loop. Default: 0
  MAX_SAMPLES           Maximum samples to infer/evaluate. 0 means all samples. Default: 0
  SAVE_VIS              Save visualizations to visualizations/. Default: false
  ENABLE_NOISE_FILTER   Apply strict filtering before sampling raw-depth prompts. Default: false
  PYTHON_BIN            Python executable. Default: ./.venv/bin/python when present

This wrapper is fixed to InfiniDepth_DepthSensor for HAMMER metric depth evaluation.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

model_path="${1:-ckpts/infinidepth_depthsensor.ckpt}"
raw_type="${2:-d435}"
encoder="${3:-vitl16}"
cleanup_npy="${4:-false}"

dataset_path="${DATASET_PATH:-data/HAMMER/test.jsonl}"
output_base_dir="${OUTPUT_DIR:-evaluation/output}"
input_size="${INPUT_SIZE:-768x1024}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
max_samples="${MAX_SAMPLES:-0}"
save_vis="${SAVE_VIS:-false}"
enable_noise_filter="${ENABLE_NOISE_FILTER:-false}"
timestamp="$(date '+%Y-%m-%d_%H-%M-%S')"
run_output_dir="${output_base_dir}/${timestamp}"
prediction_dir="${run_output_dir}/predictions"
visualization_dir="${run_output_dir}/visualizations"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/infinidepth-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

cd "${PROJECT_ROOT}"

echo "python: ${PYTHON_BIN}"
echo "model path: ${model_path}"
echo "fixed model class: InfiniDepth_DepthSensor"
echo "encoder: ${encoder}"
echo "dataset path: ${dataset_path}"
echo "raw type: ${raw_type}"
echo "input size: ${input_size}"
echo "output base dir: ${output_base_dir}"
echo "run output dir: ${run_output_dir}"
echo "prediction dir: ${prediction_dir}"
echo "visualization dir: ${visualization_dir}"
echo "max samples: ${max_samples}"
echo "save vis: ${save_vis}"
echo "enable noise filter: ${enable_noise_filter}"
echo "cleanup npy: ${cleanup_npy}"

mkdir -p "${run_output_dir}" "${prediction_dir}" "${visualization_dir}"

infer_args=(
    "${SCRIPT_DIR}/infer.py"
    --model-path "${model_path}"
    --dataset "${dataset_path}"
    --raw-type "${raw_type}"
    --encoder "${encoder}"
    --input-size "${input_size}"
    --output "${run_output_dir}"
    --prediction-dir "${prediction_dir}"
    --visualization-dir "${visualization_dir}"
    --batch-size "${batch_size}"
    --num-workers "${num_workers}"
    --max-samples "${max_samples}"
)

if [[ "${save_vis}" == "true" ]]; then
    infer_args+=(--save-vis)
fi

if [[ "${enable_noise_filter}" == "true" ]]; then
    infer_args+=(--enable-noise-filter)
fi

"${PYTHON_BIN}" "${infer_args[@]}"

echo "evaluating predictions"
time "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py" \
    --encoder "${encoder}" \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --output "${run_output_dir}" \
    --prediction-dir "${prediction_dir}" \
    --raw-type "${raw_type}" \
    --max-samples "${max_samples}"

if [[ "${cleanup_npy}" == "true" ]]; then
    echo "cleanup_npy is enabled, removing generated .npy files under ${prediction_dir}"
    find "${prediction_dir}" -maxdepth 1 -type f -name '*.npy' -delete
fi
