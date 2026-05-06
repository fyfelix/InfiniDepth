#!/usr/bin/env bash

set -euo pipefail

export OPENCV_IO_ENABLE_OPENEXR=1

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
  bash evaluation/run_dreds.sh [model_path=ckpts/infinidepth_depthsensor.ckpt] [encoder=vitl16] [variant=all] [cleanup_npy=false]

Arguments:
  model_path            InfiniDepth_DepthSensor checkpoint path.
  encoder               InfiniDepth encoder: vitl16 or vith16plus.
  variant               catknown, catnovel, or all.
  cleanup_npy           Delete predictions/*.npy after evaluation when true.

Environment overrides:
  DREDS_KNOWN_JSONL     DREDS catknown JSONL. Default: data/DREDS/test_std_catknown.jsonl
  DREDS_NOVEL_JSONL     DREDS catnovel JSONL. Default: data/DREDS/test_std_catnovel.jsonl
  OUTPUT_DIR            Prediction/evaluation output directory for a single variant.
  OUTPUT_ROOT           Root directory for default per-variant outputs. Default: checkpoint directory
  INPUT_SIZE            InfiniDepth input size as HxW. Default: 768x1024
  BATCH_SIZE            Recorded for compatibility; adapter runs sample-by-sample. Default: 1
  NUM_WORKERS           Recorded for compatibility; adapter uses a single-process loop. Default: 0
  MAX_SAMPLES           Maximum samples to infer/evaluate. 0 means all samples. Default: 0
  SAVE_VIS              Save visualizations when true. Default: true
  ENABLE_NOISE_FILTER   Apply strict filtering before sampling raw-depth prompts. Default: false
  PROMPT_SAMPLES        Maximum valid raw-depth prompt pixels. Default: 1500
  PYTHON_BIN            Python executable. Default: ./.venv/bin/python when present

DREDS uses EXR floating-point depth in meters. raw-type is passed as d435 only
to satisfy the shared Python CLI and is ignored by the DREDS dataset loader.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

model_path="${1:-ckpts/infinidepth_depthsensor.ckpt}"
encoder="${2:-vitl16}"
variant="${3:-all}"
cleanup_npy="${4:-false}"
camera_type="d435"

dreds_known_jsonl="${DREDS_KNOWN_JSONL:-data/DREDS/test_std_catknown.jsonl}"
dreds_novel_jsonl="${DREDS_NOVEL_JSONL:-data/DREDS/test_std_catnovel.jsonl}"
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
output_root="${OUTPUT_ROOT:-${model_dir}}"

if [[ "${variant}" == "all" && -n "${OUTPUT_DIR:-}" ]]; then
    echo "OUTPUT_DIR can only be used with variant=catknown or variant=catnovel; use OUTPUT_ROOT for variant=all." >&2
    exit 2
fi

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

run_one_variant() {
    local label="$1"
    local jsonl_path="$2"
    local output_dir="${OUTPUT_DIR:-${output_root}/dreds_${label}_${model_stub}}"

    echo "[${label}] python: ${PYTHON_BIN}"
    echo "[${label}] model path: ${model_path}"
    echo "[${label}] fixed model class: InfiniDepth_DepthSensor"
    echo "[${label}] encoder: ${encoder}"
    echo "[${label}] dataset path: ${jsonl_path}"
    echo "[${label}] input size: ${input_size}"
    echo "[${label}] output dir: ${output_dir}"
    echo "[${label}] max samples: ${max_samples}"
    echo "[${label}] save vis: ${save_vis}"
    echo "[${label}] enable noise filter: ${enable_noise_filter}"
    echo "[${label}] cleanup npy: ${cleanup_npy}"

    "${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py" \
        --model-path "${model_path}" \
        --dataset "${jsonl_path}" \
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

    echo "[${label}] evaluating the model on DREDS"
    time "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py" \
        --encoder "${encoder}" \
        --model-path "${model_path}" \
        --dataset "${jsonl_path}" \
        --output "${output_dir}" \
        --raw-type "${camera_type}" \
        --max-samples "${max_samples}"

    if [[ "${cleanup_npy}" == "true" ]]; then
        echo "[${label}] cleanup_npy is enabled, removing generated .npy files under ${output_dir}/predictions"
        if [[ -d "${output_dir}/predictions" ]]; then
            find "${output_dir}/predictions" -maxdepth 1 -type f -name '*.npy' -delete
        fi
    fi
}

case "${variant}" in
    catknown)
        run_one_variant catknown "${dreds_known_jsonl}"
        ;;
    catnovel)
        run_one_variant catnovel "${dreds_novel_jsonl}"
        ;;
    all)
        run_one_variant catknown "${dreds_known_jsonl}"
        run_one_variant catnovel "${dreds_novel_jsonl}"
        ;;
    *)
        echo "unknown DREDS variant: ${variant} (expected: catknown | catnovel | all)" >&2
        exit 1
        ;;
esac
