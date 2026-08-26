#!/bin/bash

set -euo pipefail
trap 'echo "[ERROR] line $LINENO, exit $?" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:?Please set ROBOTWIN_PATH to the RoboTwin repository root}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-python}"
EVAL_FILES_PATH=${STARVLA_ROOT}/examples/Robotwin/eval_files
VIDEO_DIR=${STARVLA_ROOT}/results/Robotwin

CKPT_PATH=${1:-./results/Checkpoints/vlact_robotwin_all_qwen3pi/checkpoints/steps_100000_pytorch_model.pt}
BASE_PORT=${2:-6700}
NUM_GPUS=${3:-8}
TASK_CONFIG=${4:-demo_clean}
SEED=${5:-0}
CKPT_SETTING=${6:-starvla_eval}
TARGET_EPISODES=100

declare -a ALL_TASKS=(
    adjust_bottle
    beat_block_hammer
    blocks_ranking_rgb
    blocks_ranking_size
    click_alarmclock
    click_bell
    dump_bin_bigbin
    grab_roller
    handover_block
    handover_mic
    hanging_mug
    lift_pot
    move_can_pot
    move_pillbottle_pad
    move_playingcard_away
    move_stapler_pad
    open_laptop
    open_microwave
    pick_diverse_bottles
    pick_dual_bottles
    place_a2b_left
    place_a2b_right
    place_bread_basket
    place_bread_skillet
    place_burger_fries
    place_can_basket
    place_cans_plasticbox
    place_container_plate
    place_dual_shoes
    place_empty_cup
    place_fan
    place_mouse_pad
    place_object_basket
    place_object_scale
    place_object_stand
    place_phone_stand
    place_shoe
    press_stapler
    put_bottles_dustbin
    put_object_cabinet
    rotate_qrcode
    scan_object
    shake_bottle_horizontally
    shake_bottle
    stack_blocks_three
    stack_blocks_two
    stack_bowls_three
    stack_bowls_two
    stamp_seal
    turn_switch
)

server_pids=()
worker_pids=()
ready_indices=()
MAX_WORKER_RESTARTS=${MAX_WORKER_RESTARTS:-5}
MAX_TASK_FAILURES=${MAX_TASK_FAILURES:-3}
WORKER_RESTART_DELAY=${WORKER_RESTART_DELAY:-10}

cleanup() {
    for pid in "${worker_pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${server_pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}

resolve_path() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        echo "${path}"
    else
        echo "${STARVLA_ROOT}/${path}"
    fi
}

wait_for_servers() {
    local max_wait=240
    local wait_interval=5
    local elapsed=0

    while (( elapsed < max_wait )); do
        sleep "${wait_interval}"
        elapsed=$((elapsed + wait_interval))

        local ready=0
        for i in "${!GPU_IDS[@]}"; do
            local port=$((BASE_PORT + i))
            if nc -z localhost "${port}" 2>/dev/null; then
                ready=$((ready + 1))
            fi
        done

        echo "[${elapsed}s] ${ready}/${#GPU_IDS[@]} servers ready"
        (( ready == ${#GPU_IDS[@]} )) && break
    done

    ready_indices=()
    for i in "${!GPU_IDS[@]}"; do
        local port=$((BASE_PORT + i))
        if nc -z localhost "${port}" 2>/dev/null; then
            ready_indices+=("${i}")
            echo "[OK] gpu=${GPU_IDS[$i]} port=${port}"
        else
            echo "[FAIL] gpu=${GPU_IDS[$i]} port=${port}"
        fi
    done

    if (( ${#ready_indices[@]} == 0 )); then
        echo "No server is ready." >&2
        exit 1
    fi
}

wait_for_server_slot() {
    local idx="$1"
    local port=$((BASE_PORT + idx))
    local max_wait=240
    local wait_interval=5
    local elapsed=0

    while (( elapsed < max_wait )); do
        if nc -z localhost "${port}" 2>/dev/null; then
            echo "[OK] gpu=${GPU_IDS[$idx]} port=${port}"
            return 0
        fi

        sleep "${wait_interval}"
        elapsed=$((elapsed + wait_interval))
        echo "[${elapsed}s] waiting for gpu=${GPU_IDS[$idx]} port=${port}"
    done

    echo "[FAIL] gpu=${GPU_IDS[$idx]} port=${port}" >&2
    return 1
}

launch_server_for_slot() {
    local idx="$1"
    local gpu_id="${GPU_IDS[$idx]}"
    local port=$((BASE_PORT + idx))
    local log_file="${output_server_dir}/${ckpt_name}_policy_server_gpu${gpu_id}_port${port}.log"

    echo "Launch server: gpu=${gpu_id} port=${port}"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${gpu_id}" "${STARVLA_PYTHON}" "${STARVLA_ROOT}/deployment/model_server/server_policy.py" \
        --ckpt_path "${CKPT_PATH}" \
        --port "${port}" \
        --use_bf16 \
        > "${log_file}" 2>&1 &

    server_pids[$idx]=$!
}

restart_server_for_slot() {
    local idx="$1"
    local old_pid="${server_pids[$idx]:-}"
    local port=$((BASE_PORT + idx))

    if [[ -n "${old_pid}" ]]; then
        kill "${old_pid}" 2>/dev/null || true
        wait "${old_pid}" 2>/dev/null || true
    fi

    launch_server_for_slot "${idx}"
    wait_for_server_slot "${idx}"
    write_config "${port}"
}

write_config() {
    local port="$1"
    local config_file="${output_config_dir}/deploy_policy_port${port}.yml"

    cat > "${config_file}" <<EOF
policy_name: starVLA
task_name: null
task_config: null
ckpt_setting: null
seed: null
instruction_type: unseen
host: "127.0.0.1"
port: ${port}
policy_ckpt_path: "${CKPT_PATH}"
unnorm_key: "robotwin"
logging_dir: "${VIDEO_DIR}/${MODEL_NAME}"
EOF
}

summarize_results() {
    summary_file="${output_eval_dir}/${ckpt_name}_summary_${TASK_CONFIG}_seed${SEED}.txt"
    {
        echo "Task Config: ${TASK_CONFIG}, Seed: ${SEED}, Checkpoint: ${ckpt_name}"
        echo "=========================================================="
        printf "%-35s %s\n" "Task" "Success Rate"
        echo "----------------------------------------------------------"
    } > "${summary_file}"

    for task_name in "${ALL_TASKS[@]}"; do
        task_log="${output_eval_dir}/${ckpt_name}_${task_name}_${TASK_CONFIG}_seed${SEED}.log"
        if [[ -f "${task_log}" ]]; then
            rate=$(grep -i -oP '(?:success[_ ]?rate|SR)[:\s]*\K[\d.]+' "${task_log}" 2>/dev/null | tail -1)
            if [[ -n "${rate}" ]]; then
                printf "%-35s %s\n" "${task_name}" "${rate}" | tee -a "${summary_file}"
            else
                printf "%-35s %s\n" "${task_name}" "(no result found)" | tee -a "${summary_file}"
            fi
        else
            printf "%-35s %s\n" "${task_name}" "(log missing)" | tee -a "${summary_file}"
        fi
    done

    echo "Summary: ${summary_file}"
}

count_saved_episodes() {
    local folder="$1"
    local max_count=0
    local current_count=0
    local nullglob_enabled=0

    if shopt -q nullglob; then
        nullglob_enabled=1
    fi
    shopt -s nullglob

    local matches=("${folder}"/episode*.mp4)
    current_count=${#matches[@]}
    (( current_count > max_count )) && max_count=${current_count}

    matches=("${folder}"/video/episode*.mp4)
    current_count=${#matches[@]}
    (( current_count > max_count )) && max_count=${current_count}

    matches=("${folder}"/data/episode*.hdf5)
    current_count=${#matches[@]}
    (( current_count > max_count )) && max_count=${current_count}

    matches=("${folder}"/_traj_data/episode*.pkl)
    current_count=${#matches[@]}
    (( current_count > max_count )) && max_count=${current_count}

    if (( nullglob_enabled == 0 )); then
        shopt -u nullglob
    fi

    echo "${max_count}"
}

find_latest_task_dir() {
    local task_name="$1"
    local task_root="${VIDEO_DIR}/${MODEL_NAME}/${task_name}/${policy_name}/${TASK_CONFIG}/${CKPT_SETTING}"
    local latest_dir=""

    if [[ -d "${task_root}" ]]; then
        latest_dir=$(ls -1dt "${task_root}"/* 2>/dev/null | head -n 1 || true)
    fi

    echo "${latest_dir}"
}

parse_resume_progress() {
    local task_log="$1"

    "${ROBOTWIN_PYTHON}" - "${task_log}" <<'PY'
import os
import re
import sys

task_log = sys.argv[1]
tested = 0
success = 0
next_seed = ""

if os.path.isfile(task_log):
    with open(task_log, "r", errors="ignore") as f:
        content = f.read()
    content = re.sub(r"\x1b\[[0-9;]*m", "", content)
    matches = re.findall(r"Success rate:\s*(\d+)/(\d+)\s*=>.*?current seed:\s*(\d+)", content)
    if matches:
        success, tested, current_seed = matches[-1]
        next_seed = str(int(current_seed) + 1)

print(f"{tested} {success} {next_seed}")
PY
}

prepare_pending_tasks() {
    PENDING_TASKS=()

    for task_name in "${ALL_TASKS[@]}"; do
        existing_dir=$(find_latest_task_dir "${task_name}")
        if [[ -n "${existing_dir}" ]]; then
            existing_count=$(count_saved_episodes "${existing_dir}")
            if (( existing_count >= TARGET_EPISODES )); then
                echo "[SKIP] ${task_name}: existing folder already has ${existing_count} episodes (${existing_dir})"
                continue
            fi
        fi

        PENDING_TASKS+=("${task_name}")
    done
}

claim_next_task() {
    local next_idx
    local retry_tasks=()
    CLAIMED_TASK=""

    exec 8>"${task_retry_lock_file}"
    flock 8
    if [[ -s "${task_retry_file}" ]]; then
        mapfile -t retry_tasks < "${task_retry_file}"
        CLAIMED_TASK="${retry_tasks[0]}"
        : > "${task_retry_file}"
        for ((i = 1; i < ${#retry_tasks[@]}; i++)); do
            printf "%s
" "${retry_tasks[$i]}" >> "${task_retry_file}"
        done
        flock -u 8
        exec 8>&-
        [[ -n "${CLAIMED_TASK}" ]]
        return
    fi
    flock -u 8
    exec 8>&-

    exec 9>"${task_queue_lock_file}"
    flock 9
    next_idx=$(<"${task_queue_index_file}")
    if (( next_idx < ${#PENDING_TASKS[@]} )); then
        CLAIMED_TASK="${PENDING_TASKS[$next_idx]}"
        printf "%s
" "$((next_idx + 1))" > "${task_queue_index_file}"
    fi
    flock -u 9
    exec 9>&-

    [[ -n "${CLAIMED_TASK}" ]]
}

increment_task_failure_count() {
    local task_name="$1"
    local next_count=1
    local current_task=""
    local current_count=""

    exec 8>"${task_retry_lock_file}"
    flock 8

    if [[ -f "${task_failure_count_file}" ]]; then
        while IFS=$'	' read -r current_task current_count; do
            [[ "${current_task}" == "${task_name}" ]] || continue
            next_count=$((current_count + 1))
        done < "${task_failure_count_file}"
    fi

    local tmp_file="${task_failure_count_file}.tmp"
    : > "${tmp_file}"
    local found=0
    if [[ -f "${task_failure_count_file}" ]]; then
        while IFS=$'	' read -r current_task current_count; do
            [[ -n "${current_task}" ]] || continue
            if [[ "${current_task}" == "${task_name}" ]]; then
                printf "%s	%s
" "${task_name}" "${next_count}" >> "${tmp_file}"
                found=1
            else
                printf "%s	%s
" "${current_task}" "${current_count}" >> "${tmp_file}"
            fi
        done < "${task_failure_count_file}"
    fi
    if (( found == 0 )); then
        printf "%s	%s
" "${task_name}" "${next_count}" >> "${tmp_file}"
    fi
    mv "${tmp_file}" "${task_failure_count_file}"

    flock -u 8
    exec 8>&-
    echo "${next_count}"
}

requeue_task() {
    local task_name="$1"
    local failure_count

    failure_count=$(increment_task_failure_count "${task_name}")
    if (( failure_count > MAX_TASK_FAILURES )); then
        echo "[DROP] ${task_name}: exceeded retry limit (${failure_count}/${MAX_TASK_FAILURES})"
        return 1
    fi

    exec 8>"${task_retry_lock_file}"
    flock 8
    printf "%s
" "${task_name}" >> "${task_retry_file}"
    flock -u 8
    exec 8>&-

    echo "[REQUEUE] ${task_name}: retry ${failure_count}/${MAX_TASK_FAILURES}"
}

run_eval_task() {
    local gpu_id="$1"
    local config_file="$2"
    local srv_pid="$3"
    local task_name="$4"
    local task_log="${output_eval_dir}/${ckpt_name}_${task_name}_${TASK_CONFIG}_seed${SEED}.log"
    local resume_dir=""
    local resume_test_num=0
    local resume_success_num=0
    local resume_seed=""

    echo "[GPU ${gpu_id}] ${task_name}"

    existing_dir=$(find_latest_task_dir "${task_name}")
    if [[ -n "${existing_dir}" ]]; then
        existing_count=$(count_saved_episodes "${existing_dir}")
        if (( existing_count >= TARGET_EPISODES )); then
            echo "[GPU ${gpu_id}] SKIP: ${task_name}, existing folder already has ${existing_count} episodes (${existing_dir})"
            return 0
        fi

        if (( existing_count > 0 )); then
            read -r parsed_tested parsed_success parsed_next_seed < <(parse_resume_progress "${task_log}")
            resume_dir="${existing_dir}"
            resume_test_num=${existing_count}
            resume_success_num=${parsed_success:-0}
            if (( resume_success_num > resume_test_num )); then
                resume_success_num=${resume_test_num}
            fi
            if [[ -n "${parsed_next_seed:-}" ]]; then
                resume_seed=${parsed_next_seed}
            else
                resume_seed=$((100000 * (1 + SEED) + resume_test_num))
            fi
            echo "[GPU ${gpu_id}] RESUME: ${task_name}, folder=${resume_dir}, tested=${resume_test_num}, success=${resume_success_num}, next_seed=${resume_seed}"
        fi
    fi

    extra_overrides=()
    if [[ -n "${resume_dir}" ]]; then
        extra_overrides=(
            --resume_dir "${resume_dir}"
            --resume_test_num "${resume_test_num}"
            --resume_success_num "${resume_success_num}"
            --resume_seed "${resume_seed}"
        )
    fi

    if [[ -n "${resume_dir}" ]]; then
        if ! PYTHONUNBUFFERED=1 \
            CUDA_LAUNCH_BLOCKING=1 \
            CUDA_VISIBLE_DEVICES="${gpu_id}" \
            PYTHONWARNINGS=ignore::UserWarning \
            "${ROBOTWIN_PYTHON}" script/eval_policy.py \
                --config "${config_file}" \
                --overrides \
                --task_name "${task_name}" \
                --task_config "${TASK_CONFIG}" \
                --ckpt_setting "${CKPT_SETTING}" \
                --seed "${SEED}" \
                --policy_name "${policy_name}" \
                --logging_dir "${VIDEO_DIR}/${MODEL_NAME}" \
                "${extra_overrides[@]}" \
                >> "${task_log}" 2>&1; then
            echo "[GPU ${gpu_id}] FAIL: ${task_name}, log=${task_log}"
            kill "${srv_pid}" 2>/dev/null || true
            return 1
        fi
    elif ! PYTHONUNBUFFERED=1 \
        CUDA_LAUNCH_BLOCKING=1 \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        PYTHONWARNINGS=ignore::UserWarning \
        "${ROBOTWIN_PYTHON}" script/eval_policy.py \
            --config "${config_file}" \
            --overrides \
            --task_name "${task_name}" \
            --task_config "${TASK_CONFIG}" \
            --ckpt_setting "${CKPT_SETTING}" \
            --seed "${SEED}" \
            --policy_name "${policy_name}" \
            --logging_dir "${VIDEO_DIR}/${MODEL_NAME}" \
            > "${task_log}" 2>&1; then
        echo "[GPU ${gpu_id}] FAIL: ${task_name}, log=${task_log}"
        kill "${srv_pid}" 2>/dev/null || true
        return 1
    fi
}

worker_loop() {
    local idx="$1"
    local gpu_id="${GPU_IDS[$idx]}"
    local port=$((BASE_PORT + idx))
    local srv_pid="${server_pids[$idx]}"
    local config_file="${output_config_dir}/deploy_policy_port${port}.yml"
    local task_name=""

    cd "${ROBOTWIN_PATH}"
    while claim_next_task; do
        task_name="${CLAIMED_TASK}"
        srv_pid="${server_pids[$idx]}"
        if ! run_eval_task "${gpu_id}" "${config_file}" "${srv_pid}" "${task_name}"; then
            requeue_task "${task_name}" || true
            return 1
        fi
    done
}

trap cleanup EXIT

cd "${STARVLA_ROOT}"
CKPT_PATH=$(resolve_path "${CKPT_PATH}")

ckpt_dir=$(dirname "${CKPT_PATH}")
ckpt_base=$(basename "${CKPT_PATH}")
ckpt_name="${ckpt_base%.*}"
MODEL_NAME=$(basename "$(dirname "${ckpt_dir}")")

output_server_dir="${ckpt_dir}/output_server"
output_eval_dir="${ckpt_dir}/output_eval"
output_config_dir="${ckpt_dir}/output_config"
mkdir -p "${output_server_dir}" "${output_eval_dir}" "${output_config_dir}" "${VIDEO_DIR}/${MODEL_NAME}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
    GPU_IDS=($(seq 0 $((NUM_GPUS - 1))))
fi

echo "Checkpoint : ${CKPT_PATH}"
echo "Task config: ${TASK_CONFIG}"
echo "Seed       : ${SEED}"
echo "GPUs       : ${GPU_IDS[*]}"
echo "Tasks      : ${#ALL_TASKS[@]}"
echo "Eval dir   : ${output_eval_dir}"
echo

export PYTHONPATH=${STARVLA_ROOT}:${PYTHONPATH:-}

for i in "${!GPU_IDS[@]}"; do
    launch_server_for_slot "${i}"
done

printf "%s\n" "${server_pids[*]}" > "${output_server_dir}/server_pids.txt"

wait_for_servers

if (( ${#ready_indices[@]} < ${#GPU_IDS[@]} )); then
    echo "Warning: only ${#ready_indices[@]}/${#GPU_IDS[@]} servers are ready."
fi

for idx in "${ready_indices[@]}"; do
    write_config "$((BASE_PORT + idx))"
done

export PYTHONPATH=${ROBOTWIN_PATH}:${STARVLA_ROOT}:${EVAL_FILES_PATH}:${PYTHONPATH}

policy_name="model2robotwin_interface"
active_workers=${#ready_indices[@]}
prepare_pending_tasks

task_queue_index_file="${output_eval_dir}/${ckpt_name}_${TASK_CONFIG}_seed${SEED}.queue_idx"
task_queue_lock_file="${output_eval_dir}/${ckpt_name}_${TASK_CONFIG}_seed${SEED}.queue_lock"
task_retry_file="${output_eval_dir}/${ckpt_name}_${TASK_CONFIG}_seed${SEED}.retry_queue"
task_retry_lock_file="${output_eval_dir}/${ckpt_name}_${TASK_CONFIG}_seed${SEED}.retry_lock"
task_failure_count_file="${output_eval_dir}/${ckpt_name}_${TASK_CONFIG}_seed${SEED}.failure_counts"
printf "0
" > "${task_queue_index_file}"
: > "${task_queue_lock_file}"
: > "${task_retry_file}"
: > "${task_retry_lock_file}"
: > "${task_failure_count_file}"

echo "Pending tasks: ${#PENDING_TASKS[@]}/${#ALL_TASKS[@]}"

if (( ${#PENDING_TASKS[@]} == 0 )); then
    echo "All task folders are already complete."
    echo "Logs: ${output_eval_dir}"
    summarize_results
    exit 0
fi

for i in "${!ready_indices[@]}"; do
    idx=${ready_indices[$i]}
    gpu_id=${GPU_IDS[$idx]}
    port=$((BASE_PORT + idx))

    echo "Worker ${i}: gpu=${gpu_id} port=${port} queue=dynamic restart_limit=${MAX_WORKER_RESTARTS}"

    (
        restart_count=0
        while true; do
            if worker_loop "${idx}"; then
                exit 0
            fi

            restart_count=$((restart_count + 1))
            if (( restart_count > MAX_WORKER_RESTARTS )); then
                echo "[FAIL] worker gpu=${gpu_id} exceeded restart limit (${restart_count}/${MAX_WORKER_RESTARTS})"
                exit 1
            fi

            echo "[RESTART] worker gpu=${gpu_id} attempt ${restart_count}/${MAX_WORKER_RESTARTS}"
            sleep "${WORKER_RESTART_DELAY}"
            if ! restart_server_for_slot "${idx}"; then
                echo "[FAIL] worker gpu=${gpu_id} could not restart server"
                exit 1
            fi
        done
    ) &

    worker_pids+=($!)
done

failed_count=0
for i in "${!worker_pids[@]}"; do
    if wait "${worker_pids[$i]}"; then
        echo "[DONE] worker ${i}"
    else
        echo "[FAIL] worker ${i}"
        failed_count=$((failed_count + 1))
    fi
done

echo
echo "Finished. failed_workers=${failed_count}"
echo "Logs: ${output_eval_dir}"
summarize_results
