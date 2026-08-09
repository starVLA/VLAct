#!/bin/bash
#SBATCH --job-name=droid_cotrain_coco
#SBATCH --partition=vonneumann
#SBATCH --account=vonneumann1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=24
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/droid_cotrain_coco_%j.out
#SBATCH --error=logs/droid_cotrain_coco_%j.err

# ============================================================
#  DROID + COCO co-training on a single 8-GPU node
#  Submit as a normal batch job:
#    sbatch examples/DROID/train_files/run_droid_cotrain_coco.sh
#
#  Reuse an existing allocation on a reserved node:
#    srun --overlap --jobid <alloc_jobid> -w <node> \
#      bash examples/DROID/train_files/run_droid_cotrain_coco.sh
# ============================================================

set -eo pipefail

source /project/vonneumann1/sqyang/.bashrc
conda activate starVLA

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000

SCRIPT_REL_PATH=examples/DROID/train_files/run_droid_cotrain_coco.sh
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/starVLA" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
fi

if [[ ! -d "${REPO_ROOT}/starVLA" ]]; then
  echo "Resolved REPO_ROOT is invalid: ${REPO_ROOT}" >&2
  exit 1
fi

SCRIPT_SOURCE="${REPO_ROOT}/${SCRIPT_REL_PATH}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p logs

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

NNODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=8
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

Framework_name=QwenOFT
base_vlm="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
config_yaml="${REPO_ROOT}/examples/DROID/train_files/starvla_droid_cotrain_coco.yaml"
droid_data_root=/project/vonneumann1/datasets
vlm_data=sharegpt4v_coco
run_root_dir="${REPO_ROOT}/results/Checkpoints"
run_id=0404_droid_cotrain_coco_qwen3OFT

freeze_module_list=qwen_vl_interface.model.visual
for layer_idx in $(seq 0 17); do
  freeze_module_list+=",qwen_vl_interface.model.language_model.layers.${layer_idx}"
done

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
if [[ -f "${SCRIPT_SOURCE}" ]]; then
  cp "${SCRIPT_SOURCE}" "${output_dir}/"
else
  cp "$0" "${output_dir}/"
fi

echo "============================================"
echo "Job ID                : ${SLURM_JOB_ID:-manual}"
echo "Nodes                 : ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "NNODES                : ${NNODES}"
echo "GPUS/NODE             : ${GPUS_PER_NODE}"
echo "TOTAL GPUS            : ${TOTAL_GPUS}"
echo "MASTER_ADDR           : ${MASTER_ADDR}"
echo "MASTER_PORT           : ${MASTER_PORT}"
echo "Freeze modules        : ${freeze_module_list}"
echo "DROID root            : ${droid_data_root}"
echo "VLM dataset           : ${vlm_data}"
echo "============================================"

launch_cmd=(
  accelerate launch
  --config_file "${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2.yaml"
  --main_process_ip "${MASTER_ADDR}"
  --main_process_port "${MASTER_PORT}"
  --machine_rank 0
  --num_machines "${NNODES}"
  --num_processes "${TOTAL_GPUS}"
  "${REPO_ROOT}/starVLA/training/train_starvla_cotrain.py"
  --config_yaml "${config_yaml}"
  --framework.name "${Framework_name}"
  --framework.qwenvl.base_vlm "${base_vlm}"
  --datasets.vla_data.data_root_dir "${droid_data_root}"
  --datasets.vla_data.data_mix droid
  --datasets.vla_data.per_device_batch_size 24
  --datasets.vla_data.sequential_step_sampling False
  --datasets.vlm_data.dataset_use "${vlm_data}"
  --datasets.vlm_data.eval_dataset "${vlm_data}"
  --datasets.vlm_data.per_device_batch_size 8
  --trainer.freeze_modules "${freeze_module_list}"
  --trainer.max_train_steps 50000
  --trainer.save_interval 10000
  --trainer.logging_frequency 100
  --trainer.eval_interval 1000
  --run_root_dir "${run_root_dir}"
  --run_id "${run_id}"
  --wandb_project EM-LLaVA
  --wandb_entity yangsenqiao
)

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --kill-on-bad-exit=1 "${launch_cmd[@]}"
else
  "${launch_cmd[@]}"
fi