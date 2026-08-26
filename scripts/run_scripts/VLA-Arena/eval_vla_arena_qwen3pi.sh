#!/bin/bash

set -euo pipefail
trap 'echo "[ERROR] line $LINENO, exit $?" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
EVAL_SCRIPT_PATH="${STARVLA_ROOT}/examples/VLA-Arena/eval_files/eval_vla_arena.py"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"

VLA_ARENA_PROJECT="${VLA_ARENA_PROJECT:-}"
VLA_ARENA_HOME="${VLA_ARENA_HOME:-}"

CKPT_PATH="${1:-./results/Checkpoints/vlact_vla_arena_qwen3pi/checkpoints/steps_50000_pytorch_model.pt}"
BASE_PORT="${2:-10090}"
NUM_GPUS="${3:-8}"
NUM_TRIALS="${4:-10}"
SEED="${5:-7}"

# Save all rollout videos by default.
SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"
USE_TWO_VIEWS="${USE_TWO_VIEWS:-true}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-240}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

TASK_SUITES=(
    "safety_static_obstacles"
    "safety_cautious_grasp"
    "safety_hazard_avoidance"
    "safety_state_preservation"
    "safety_dynamic_obstacles"
    "distractor_static_distractors"
    "distractor_dynamic_distractors"
    "extrapolation_preposition_combinations"
    "extrapolation_task_workflows"
    "extrapolation_unseen_objects"
    "long_horizon"
)
TASK_LEVELS=(0 1 2)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

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
  bash ${SCRIPT_DIR}/eval_vla_arena_qwen3pi.sh [CKPT_PATH] [BASE_PORT] [NUM_GPUS] [NUM_TRIALS] [SEED]

Environment variables:
  STARVLA_ROOT         starVLA root directory
  STARVLA_PYTHON       python used to launch server_policy.py
  VLA_ARENA_PROJECT    uv project directory for VLA-Arena eval environment (required)
  VLA_ARENA_HOME       VLA-Arena repository root (required)
  SAVE_VIDEO_MODE      none | first_success_failure | all   (default: all)
  USE_TWO_VIEWS        true | false                         (default: true)
  SERVER_STARTUP_WAIT  seconds to wait for each server      (default: 240)
  SKIP_EXISTING        true | false                         (default: true)
  Video output         ${STARVLA_ROOT}/results/VLA-Arena/<run>/<checkpoint>/<suite>_L<level>

Examples:
  VLA_ARENA_PROJECT=/path/to/VLA-Arena/envs/openpi \\
  VLA_ARENA_HOME=/path/to/VLA-Arena \\
  bash ${SCRIPT_DIR}/eval_vla_arena_qwen3pi.sh \\
    ./results/Checkpoints/your_run/checkpoints/steps_50000_pytorch_model.pt 10090 8 10 7
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_usage
    exit 0
fi

if [[ -z "${VLA_ARENA_PROJECT}" || -z "${VLA_ARENA_HOME}" ]]; then
    print_error "Please set both VLA_ARENA_PROJECT and VLA_ARENA_HOME before running."
    show_usage
    exit 1
fi

if [[ ! -d "${VLA_ARENA_PROJECT}" ]]; then
    print_error "VLA_ARENA_PROJECT does not exist: ${VLA_ARENA_PROJECT}"
    exit 1
fi

if [[ ! -d "${VLA_ARENA_HOME}" ]]; then
    print_error "VLA_ARENA_HOME does not exist: ${VLA_ARENA_HOME}"
    exit 1
fi

if [[ ! -x "${STARVLA_PYTHON}" ]]; then
    print_error "STARVLA_PYTHON is not executable: ${STARVLA_PYTHON}"
    exit 1
fi

CKPT_PATH="$(resolve_path "${CKPT_PATH}")"
if [[ ! -f "${CKPT_PATH}" ]]; then
    print_error "Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
    GPU_IDS=($(seq 0 $((NUM_GPUS - 1))))
fi

if (( ${#GPU_IDS[@]} < NUM_GPUS )); then
    print_error "Requested ${NUM_GPUS} GPUs but only got ${#GPU_IDS[@]} from CUDA_VISIBLE_DEVICES."
    exit 1
fi
GPU_IDS=("${GPU_IDS[@]:0:${NUM_GPUS}}")

ckpt_dir="$(dirname "${CKPT_PATH}")"
ckpt_base="$(basename "${CKPT_PATH}")"
ckpt_name="${ckpt_base%.*}"
run_name="$(basename "$(dirname "${ckpt_dir}")")"

output_server_dir="${ckpt_dir}/output_server"
output_eval_dir="${ckpt_dir}/output_eval"
output_video_dir="${STARVLA_ROOT}/results/VLA-Arena/${run_name}/${ckpt_name}"
mkdir -p "${output_server_dir}" "${output_eval_dir}" "${output_video_dir}"

summary_file="${output_eval_dir}/${ckpt_name}_vla_arena_summary.csv"
summary_lock="${summary_file}.lock"
queue_idx_file="${output_eval_dir}/${ckpt_name}.queue_idx"
queue_lock_file="${output_eval_dir}/${ckpt_name}.queue_lock"
job_sep=$'\t'

server_pids=()
worker_pids=()
ready_indices=()
ALL_JOBS=()

cleanup() {
    for pid in "${worker_pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${server_pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}

trap cleanup EXIT

port_ready() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(:|\\])${port}$"
        return $?
    fi
    if command -v nc >/dev/null 2>&1; then
        nc -z 127.0.0.1 "${port}" >/dev/null 2>&1
        return $?
    fi
    return 1
}

wait_for_server_slot() {
    local idx="$1"
    local gpu_id="${GPU_IDS[$idx]}"
    local port=$((BASE_PORT + idx))
    local log_file="${output_server_dir}/${ckpt_name}_server_gpu${gpu_id}_port${port}.log"
    local elapsed=0

    while (( elapsed < SERVER_STARTUP_WAIT )); do
        if ! kill -0 "${server_pids[$idx]}" 2>/dev/null; then
            print_error "Server on GPU ${gpu_id} exited early. Log: ${log_file}"
            [[ -f "${log_file}" ]] && tail -30 "${log_file}" || true
            return 1
        fi

        if [[ -f "${log_file}" ]] && grep -Eq "server listening on .*:${port}" "${log_file}"; then
            print_success "Server ready: gpu=${gpu_id} port=${port}"
            return 0
        fi

        if port_ready "${port}"; then
            print_success "Server ready: gpu=${gpu_id} port=${port}"
            return 0
        fi

        sleep 5
        elapsed=$((elapsed + 5))
        print_info "[${elapsed}s] waiting for gpu=${gpu_id} port=${port}"
    done

    print_error "Timeout waiting for server on gpu=${gpu_id} port=${port}"
    return 1
}

launch_server_for_slot() {
    local idx="$1"
    local gpu_id="${GPU_IDS[$idx]}"
    local port=$((BASE_PORT + idx))
    local log_file="${output_server_dir}/${ckpt_name}_server_gpu${gpu_id}_port${port}.log"

    print_info "Launching server: gpu=${gpu_id} port=${port}"
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${STARVLA_PYTHON}" \
        "${STARVLA_ROOT}/deployment/model_server/server_policy.py" \
        --ckpt_path "${CKPT_PATH}" \
        --port "${port}" \
        --use_bf16 \
        > "${log_file}" 2>&1 &

    server_pids[$idx]=$!
}

restart_server_for_slot() {
    local idx="$1"
    local old_pid="${server_pids[$idx]:-}"
    if [[ -n "${old_pid}" ]]; then
        kill "${old_pid}" 2>/dev/null || true
        wait "${old_pid}" 2>/dev/null || true
    fi
    launch_server_for_slot "${idx}"
    wait_for_server_slot "${idx}"
}

extract_success_rate() {
    local log_file="$1"
    grep -i "Final SR:" "${log_file}" | tail -1 | sed 's/.*Final SR: //' | awk '{print $1}'
}

extract_success_pair() {
    local log_file="$1"
    grep -i "Final SR:" "${log_file}" | tail -1 | grep -o '([0-9]*/[0-9]*)' | tr -d '()'
}

extract_avg_cost() {
    local log_file="$1"
    grep -i "avg_cost=" "${log_file}" | tail -1 | sed 's/.*avg_cost=//' | awk '{print $1}'
}

append_summary_row() {
    local row="$1"
    exec 8>"${summary_lock}"
    flock 8
    echo "${row}" >> "${summary_file}"
    flock -u 8
    exec 8>&-
}

claim_next_job() {
    CLAIMED_JOB=""
    exec 9>"${queue_lock_file}"
    flock 9
    local next_idx
    next_idx=$(<"${queue_idx_file}")
    if (( next_idx < ${#ALL_JOBS[@]} )); then
        CLAIMED_JOB="${ALL_JOBS[$next_idx]}"
        printf "%s\n" "$((next_idx + 1))" > "${queue_idx_file}"
    fi
    flock -u 9
    exec 9>&-
    [[ -n "${CLAIMED_JOB}" ]]
}

run_eval_job() {
    local idx="$1"
    local suite="$2"
    local level="$3"
    local gpu_id="${GPU_IDS[$idx]}"
    local port=$((BASE_PORT + idx))
    local log_file="${output_eval_dir}/${ckpt_name}_${suite}_L${level}.log"
    local video_out="${output_video_dir}/${suite}_L${level}"
    local two_view_args=()

    mkdir -p "${video_out}"

    if [[ "${SKIP_EXISTING}" == "true" && -f "${log_file}" ]] && grep -q "Final SR:" "${log_file}"; then
        print_warning "[gpu=${gpu_id}] Skip ${suite} L${level} (existing log)"
        return 0
    fi

    if [[ "${USE_TWO_VIEWS}" == "true" ]]; then
        two_view_args+=(--args.use-two-views)
    else
        two_view_args+=(--args.no-use-two-views)
    fi

    print_info "[gpu=${gpu_id}] Eval ${suite} L${level} on port ${port}"
    if ! PYTHONUNBUFFERED=1 \
        PYTHONPATH="${VLA_ARENA_HOME}/vla_arena:${STARVLA_ROOT}:${PYTHONPATH:-}" \
        uv run --project "${VLA_ARENA_PROJECT}" \
        python "${EVAL_SCRIPT_PATH}" \
            --args.pretrained-path "${CKPT_PATH}" \
            --args.host "127.0.0.1" \
            --args.port "${port}" \
            --args.task-suite-name "${suite}" \
            --args.task-level "${level}" \
            --args.num-trials-per-task "${NUM_TRIALS}" \
            --args.seed "${SEED}" \
            "${two_view_args[@]}" \
            --args.video-out-path "${video_out}" \
            --args.save-video-mode "${SAVE_VIDEO_MODE}" \
            > "${log_file}" 2>&1; then
        print_error "[gpu=${gpu_id}] Failed ${suite} L${level}, restarting server"
        append_summary_row "${suite},L${level},FAILED,N/A,N/A,N/A,${log_file}"
        restart_server_for_slot "${idx}"
        return 1
    fi

    local sr pair succ total avg_cost
    sr="$(extract_success_rate "${log_file}")"
    pair="$(extract_success_pair "${log_file}")"
    succ="${pair%%/*}"
    total="${pair##*/}"
    avg_cost="$(extract_avg_cost "${log_file}")"
    append_summary_row "${suite},L${level},${sr},${succ},${total},${avg_cost},${log_file}"
    print_success "[gpu=${gpu_id}] Done ${suite} L${level}: SR=${sr} (${succ}/${total}) avg_cost=${avg_cost}"
    return 0
}

worker_loop() {
    local idx="$1"
    local suite level
    while claim_next_job; do
        IFS="${job_sep}" read -r suite level <<< "${CLAIMED_JOB}"
        run_eval_job "${idx}" "${suite}" "${level}"
    done
}

printf "Task Suite,Level,Success Rate,Successes,Total Episodes,Avg Cost,Log File\n" > "${summary_file}"
printf "0\n" > "${queue_idx_file}"
: > "${queue_lock_file}"

for suite in "${TASK_SUITES[@]}"; do
    for level in "${TASK_LEVELS[@]}"; do
        ALL_JOBS+=("${suite}${job_sep}${level}")
    done
done

print_info "Checkpoint : ${CKPT_PATH}"
print_info "GPUs       : ${GPU_IDS[*]}"
print_info "Trials     : ${NUM_TRIALS}"
print_info "Seed       : ${SEED}"
print_info "Two views  : ${USE_TWO_VIEWS}"
print_info "Jobs       : ${#ALL_JOBS[@]} (11 suites x 3 levels)"
print_info "Summary    : ${summary_file}"

cd "${STARVLA_ROOT}"
for i in "${!GPU_IDS[@]}"; do
    launch_server_for_slot "${i}"
done

for i in "${!GPU_IDS[@]}"; do
    if wait_for_server_slot "${i}"; then
        ready_indices+=("${i}")
    fi
done

if (( ${#ready_indices[@]} == 0 )); then
    print_error "No policy server became ready."
    exit 1
fi

print_success "Ready workers: ${#ready_indices[@]} / ${#GPU_IDS[@]}"

for idx in "${ready_indices[@]}"; do
    (
        worker_loop "${idx}"
    ) &
    worker_pids+=($!)
done

failed_workers=0
for i in "${!worker_pids[@]}"; do
    if wait "${worker_pids[$i]}"; then
        print_success "Worker ${i} finished"
    else
        print_error "Worker ${i} failed"
        failed_workers=$((failed_workers + 1))
    fi
done

print_info "Finished. failed_workers=${failed_workers}"
print_info "Server logs: ${output_server_dir}"
print_info "Eval logs  : ${output_eval_dir}"
print_success "Summary CSV: ${summary_file}"
