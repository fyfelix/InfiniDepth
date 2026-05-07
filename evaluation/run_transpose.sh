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
  bash evaluation/run_transpose.sh [model_path=ckpts/infinidepth_depthsensor.ckpt] [encoder=vitl16] [cleanup_npy=false]

Arguments:
  model_path            InfiniDepth_DepthSensor checkpoint path.
  encoder               InfiniDepth encoder: vitl16 or vith16plus.
  cleanup_npy           Delete predictions/*.npy after evaluation when true.

Environment overrides:
  DATASET_PATH          TRansPose JSONL path. Default: data/TRansPose/sequences/dc_testset.jsonl
  OUTPUT_DIR            Prediction/evaluation output directory. Default: <checkpoint_dir>/transpose_<jsonl_stub>_<checkpoint_stub>_data_l515
  INPUT_SIZE            InfiniDepth input size as HxW. Default: 768x1024
  BATCH_SIZE            Recorded for compatibility; adapter runs sample-by-sample. Default: 1
  NUM_WORKERS           Recorded for compatibility; adapter uses a single-process loop. Default: 0
  MAX_SAMPLES           Maximum samples to infer/evaluate. 0 means all samples. Default: 0
  SAVE_VIS              Save visualizations when true. Default: true
  ENABLE_NOISE_FILTER   Apply strict filtering before sampling raw-depth prompts. Default: false
  PROMPT_SAMPLES        Maximum valid raw-depth prompt pixels. Default: 1500
  PYTHON_BIN            Python executable. Default: ./.venv/bin/python when present

TRansPose is fixed to raw-type=l515 and uses uint16 PNG depth in millimeters.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

model_path="${1:-ckpts/infinidepth_depthsensor.ckpt}"
encoder="${2:-vitl16}"
cleanup_npy="${3:-false}"
camera_type="l515"

dataset_path="${DATASET_PATH:-data/TRansPose/sequences/dc_testset.jsonl}"
input_size="${INPUT_SIZE:-768x1024}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
max_samples="${MAX_SAMPLES:-0}"
save_vis="${SAVE_VIS:-true}"
enable_noise_filter="${ENABLE_NOISE_FILTER:-false}"
prompt_samples="${PROMPT_SAMPLES:-1500}"

model_name="$(basename "${model_path}")"
model_stub="${model_name%%.*}"
model_dir="$(dirname "${model_path}")"
dataset_name="$(basename "${dataset_path}")"
dataset_stub="${dataset_name%%.*}"
output_dir="${OUTPUT_DIR:-${model_dir}/transpose_${dataset_stub}_${model_stub}_data_${camera_type}}"

save_vis_arg=()
case "${save_vis}" in
    [Tt][Rr][Uu][Ee]) save_vis_arg=(--save-vis) ;;
    [Ff][Aa][Ll][Ss][Ee]) save_vis_arg=(--no-save-vis) ;;
    *)
        echo "SAVE_VIS must be true or false, got: ${save_vis}" >&2
        exit 2
        ;;
esac

noise_filter_arg=()
if [[ "${enable_noise_filter}" == "true" ]]; then
    noise_filter_arg=(--enable-noise-filter)
fi

cd "${PROJECT_ROOT}"

echo "python: ${PYTHON_BIN}"
echo "model path: ${model_path}"
echo "fixed model class: InfiniDepth_DepthSensor"
echo "encoder: ${encoder}"
echo "dataset path: ${dataset_path}"
echo "camera type: ${camera_type}"
echo "input size: ${input_size}"
echo "output dir: ${output_dir}"
echo "max samples: ${max_samples}"
echo "save vis: ${save_vis}"
echo "enable noise filter: ${enable_noise_filter}"
echo "cleanup npy: ${cleanup_npy}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py" \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --raw-type "${camera_type}" \
    --encoder "${encoder}" \
    --input-size "${input_size}" \
    --output "${output_dir}" \
    --batch-size "${batch_size}" \
    --num-workers "${num_workers}" \
    --max-samples "${max_samples}" \
    --prompt-samples "${prompt_samples}" \
    "${save_vis_arg[@]}" \
    "${noise_filter_arg[@]}"

echo "evaluating the model on TRansPose"
time "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py" \
    --encoder "${encoder}" \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --output "${output_dir}" \
    --raw-type "${camera_type}" \
    --max-samples "${max_samples}"

if [[ "${cleanup_npy}" == "true" ]]; then
    echo "cleanup_npy is enabled, removing generated .npy files under ${output_dir}/predictions"
    if [[ -d "${output_dir}/predictions" ]]; then
        find "${output_dir}/predictions" -maxdepth 1 -type f -name '*.npy' -delete
    fi
fi
