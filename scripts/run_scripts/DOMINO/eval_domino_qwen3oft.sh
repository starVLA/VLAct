#!/usr/bin/env bash

set -euo pipefail
trap 'echo "[ERROR] line $LINENO, exit $?" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DOMINO_PATH="${DOMINO_PATH:-}"
EVAL_FILES_PATH="${STARVLA_ROOT}/examples/DOMINO/eval_files"

STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
DOMINO_PYTHON="${DOMINO_PYTHON:-python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

DEFAULT_CKPT="./results/Checkpoints/vlact_domino_qwen3oft/checkpoints/steps_100000_pytorch_model.pt"

CKPT_PATH="${1:-${DEFAULT_CKPT}}"
BASE_PORT="${2:-7700}"
NUM_GPUS="${3:-8}"
TASK_CONFIG="${4:-${DOMINO_TASK_CONFIG:-demo_clean_dynamic}}"
SEED="${5:-0}"
CKPT_SETTING="${6:-}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-${DOMINO_ACTION_CHUNK_SIZE:-32}}"

TASK_ARGS=()
if (( $# > 6 )); then
    TASK_ARGS=("${@:7}")
else
    TASK_ARGS=(all)
fi

resolve_path() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        echo "${path}"
    else
        echo "${STARVLA_ROOT}/${path}"
    fi
}

show_usage() {
    cat <<EOF
Usage:
  bash ${SCRIPT_DIR}/eval_domino_qwen3oft.sh [CKPT_PATH] [BASE_PORT] [NUM_GPUS] [TASK_CONFIG] [SEED] [CKPT_SETTING] [tasks...]

Defaults:
  CKPT_PATH     ${DEFAULT_CKPT}
  BASE_PORT     7700
  NUM_GPUS      8
  TASK_CONFIG   demo_clean_dynamic
  SEED          0
  tasks         all

TASK_CONFIG aliases:
  clean         -> demo_clean_dynamic
  random        -> demo_random_dynamic
  all           -> run both demo_clean_dynamic and demo_random_dynamic

Examples:
  bash ${SCRIPT_DIR}/eval_domino_qwen3oft.sh
  bash ${SCRIPT_DIR}/eval_domino_qwen3oft.sh ./results/Checkpoints/vlact_domino_clean_wrap_qwen3OFT/checkpoints/steps_50000_pytorch_model.pt 7700 8 clean 0
  bash ${SCRIPT_DIR}/eval_domino_qwen3oft.sh ./results/Checkpoints/vlact_domino_clean_qwen3OFT/checkpoints/steps_50000_pytorch_model.pt 7800 8 random 0 vlact_eval adjust_bottle
EOF
}

select_gpus() {
    local -a visible_gpus=()
    local gpu_id=""

    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
    else
        mapfile -t visible_gpus < <(seq 0 "$((NUM_GPUS - 1))")
    fi

    if (( ${#visible_gpus[@]} < NUM_GPUS )); then
        echo "[ERROR] Requested ${NUM_GPUS} GPUs but CUDA_VISIBLE_DEVICES only exposes ${#visible_gpus[@]}: ${visible_gpus[*]}" >&2
        exit 1
    fi

    GPU_IDS=()
    for gpu_id in "${visible_gpus[@]:0:${NUM_GPUS}}"; do
        GPU_IDS+=("${gpu_id//[[:space:]]/}")
    done

    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_IDS[*]}")"
}

normalize_task_config() {
    case "$1" in
        clean|demo_clean_dynamic)
            echo "demo_clean_dynamic"
            ;;
        random|demo_random_dynamic)
            echo "demo_random_dynamic"
            ;;
        all|both)
            echo "all"
            ;;
        *)
            echo "[ERROR] Unsupported TASK_CONFIG: $1 (expected clean, random, all, demo_clean_dynamic, demo_random_dynamic)" >&2
            exit 1
            ;;
    esac
}

run_one_mode() {
    local mode="$1"
    local mode_base_port="$2"
    local mode_ckpt_setting="${CKPT_SETTING}"

    if [[ -z "${mode_ckpt_setting}" ]]; then
        mode_ckpt_setting="${MODEL_NAME}_${mode}_${MODEL_VARIANT}_seed${SEED}"
    elif [[ "${TASK_CONFIG}" == "all" ]]; then
        mode_ckpt_setting="${mode_ckpt_setting}_${mode}"
    fi

    export DOMINO_PATH
    export STARVLA_PYTHON
    export DOMINO_PYTHON
    export DOMINO_ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE}"
    export DOMINO_LOG_ROOT="${output_eval_dir}"
    export DOMINO_VIDEO_ROOT="${VIDEO_DIR}/${MODEL_NAME}"
    unset PYTHONPATH

    echo "============================================================"
    echo "DOMINO eval mode : ${mode}"
    echo "Checkpoint       : ${CKPT_PATH}"
    echo "Run name         : ${mode_ckpt_setting}"
    echo "GPUs             : ${CUDA_VISIBLE_DEVICES}"
    echo "Base port        : ${mode_base_port}"
    echo "Seed             : ${SEED}"
    echo "Chunk size       : ${ACTION_CHUNK_SIZE}"
    echo "StarVLA python   : ${STARVLA_PYTHON}"
    echo "DOMINO python    : ${DOMINO_PYTHON}"
    echo "Video dir        : ${DOMINO_VIDEO_ROOT}"
    echo "Test num         : ${DOMINO_TEST_NUM:-default}"
    echo "Expert check     : ${DOMINO_EXPERT_CHECK:-default}"
    echo "Resume by seed   : ${DOMINO_RESUME_BY_SEED:-0}"
    echo "Start seed       : ${DOMINO_START_SEED:-default}"
    echo "Tasks            : ${TASK_ARGS[*]}"
    echo "Logs             : ${DOMINO_LOG_ROOT}"
    echo "============================================================"

    bash "${EVAL_FILES_PATH}/start_eval.sh" \
        --mode "${mode}" \
        --name "${mode_ckpt_setting}" \
        --ckpt "${CKPT_PATH}" \
        --seed "${SEED}" \
        --jobs-per-gpu "${DOMINO_JOBS_PER_GPU:-1}" \
        --base-port "${mode_base_port}" \
        --server-timeout "${DOMINO_SERVER_TIMEOUT:-600}" \
        --action-chunk-size "${ACTION_CHUNK_SIZE}" \
        "${TASK_ARGS[@]}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_usage
    exit 0
fi

cd "${STARVLA_ROOT}"
CKPT_PATH="$(resolve_path "${CKPT_PATH}")"
TASK_CONFIG="$(normalize_task_config "${TASK_CONFIG}")"

if [[ ! -d "${DOMINO_PATH}" ]]; then
    echo "[ERROR] DOMINO_PATH does not exist: ${DOMINO_PATH}" >&2
    exit 1
fi

if [[ ! -x "${STARVLA_PYTHON}" ]]; then
    echo "[ERROR] STARVLA_PYTHON is not executable: ${STARVLA_PYTHON}" >&2
    exit 1
fi

if [[ ! -x "${DOMINO_PYTHON}" ]]; then
    echo "[ERROR] DOMINO_PYTHON is not executable: ${DOMINO_PYTHON}" >&2
    exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[ERROR] Checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
fi

select_gpus

ckpt_dir="$(dirname "${CKPT_PATH}")"
ckpt_base="$(basename "${CKPT_PATH}")"
ckpt_name="${ckpt_base%.*}"
MODEL_NAME="$(basename "$(dirname "${ckpt_dir}")")"
if [[ "${MODEL_NAME,,}" == *wrap* ]]; then
    MODEL_VARIANT="wrap"
else
    MODEL_VARIANT="plain"
fi
output_eval_dir="${ckpt_dir}/output_eval"
VIDEO_DIR="${STARVLA_ROOT}/results/DOMINO"
mkdir -p "${output_eval_dir}" "${VIDEO_DIR}/${MODEL_NAME}"

echo "Script copy      : ${output_eval_dir}/$(basename "$0")"
cp "$0" "${output_eval_dir}/"

if [[ "${TASK_CONFIG}" == "all" ]]; then
    run_one_mode "demo_clean_dynamic" "${BASE_PORT}"
    run_one_mode "demo_random_dynamic" "$((BASE_PORT + NUM_GPUS * ${DOMINO_JOBS_PER_GPU:-1} + 32))"
else
    run_one_mode "${TASK_CONFIG}" "${BASE_PORT}"
fi
