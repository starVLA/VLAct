#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:?Please set LIBERO_HOME to the LIBERO repository root}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"

CKPT_PATH="${1:-./results/Checkpoints/vlact_libero_all_qwen3pi/checkpoints/steps_50000_pytorch_model.pt}"
TASK_SUITE_NAME="${2:-libero_goal}"
GPU_ID="${3:-0}"
PORT="${4:-5694}"
NUM_TRIALS="${5:-50}"

if [[ "${CKPT_PATH}" != /* ]]; then
    CKPT_PATH="${STARVLA_ROOT}/${CKPT_PATH}"
fi

export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_ROOT}:${PYTHONPATH:-}"

model_root="$(dirname "$(dirname "${CKPT_PATH}")")"
ckpt_name="$(basename "${CKPT_PATH}")"
video_out_path="${model_root}/videos/${TASK_SUITE_NAME}/${ckpt_name%.*}"
log_dir="${model_root}/logs/${TASK_SUITE_NAME}"
mkdir -p "${video_out_path}" "${log_dir}"

server_pid=""
cleanup() {
    if [[ -n "${server_pid}" ]]; then
        kill "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

cd "${STARVLA_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port "${PORT}" \
    --use_bf16 &
server_pid=$!
sleep "${SERVER_STARTUP_WAIT:-30}"

"${LIBERO_PYTHON}" examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT_PATH}" \
    --args.host 127.0.0.1 \
    --args.port "${PORT}" \
    --args.task-suite-name "${TASK_SUITE_NAME}" \
    --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.video-out-path "${video_out_path}" \
    2>&1 | tee "${log_dir}/${ckpt_name%.*}.log"
