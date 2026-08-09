#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash examples/DOMINO/eval_files/run_policy_server.sh <ckpt_path> [gpu_id] [port]" >&2
    exit 1
fi

your_ckpt="$1"
gpu_id="${2:-${DOMINO_SERVER_GPU:-0}}"
port="${3:-${DOMINO_SERVER_PORT:-5694}}"
star_vla_python="${STARVLA_PYTHON:-${star_vla_python:-python}}"
idle_timeout="${DOMINO_POLICY_IDLE_TIMEOUT:--1}"
action_chunk_size="${DOMINO_ACTION_CHUNK_SIZE:-${ACTION_CHUNK_SIZE:-32}}"

use_bf16_flag=()
if [[ "${DOMINO_USE_BF16:-1}" != "0" ]]; then
    use_bf16_flag+=(--use_bf16)
fi

echo "[INFO] Starting DOMINO policy server"
echo "[INFO] python: ${star_vla_python}"
echo "[INFO] checkpoint: ${your_ckpt}"
echo "[INFO] gpu: ${gpu_id}"
echo "[INFO] port: ${port}"
echo "[INFO] idle_timeout: ${idle_timeout}"
echo "[INFO] action_chunk_size: ${action_chunk_size}"

CUDA_VISIBLE_DEVICES="${gpu_id}" \
PYTHONPATH="${REPO_ROOT}${STARVLA_EXTRA_PYTHONPATH:+:${STARVLA_EXTRA_PYTHONPATH}}" \
"${star_vla_python}" "${REPO_ROOT}/deployment/model_server/server_policy.py" \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    --idle_timeout "${idle_timeout}" \
    --action_chunk_size "${action_chunk_size}" \
    "${use_bf16_flag[@]}"
