#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600  # timeout set to 1 hour (unit: seconds)
export DEEPSPEED_TIMEOUT=720  # collective wait timeout, unit: minutes (rank0 cold-cache filtering takes ~40min)



###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenHybrid_xrobot_padding
freeze_module_list=qwen_vl_interface.model.visual
for layer_idx in $(seq 0 17); do
  freeze_module_list+=" ,qwen_vl_interface.model.language_model.layers.${layer_idx}"
done
freeze_module_list=${freeze_module_list// ,/,}
base_vlm=StarVLA/Qwen3-VL-4B-Instruct-Action
config_yaml=./examples/InternA1/train_files/starvla_cotrain_agilex_franka_50_padding.yaml
data_root_dir=./playground/Datasets
run_root_dir=./results/Checkpoints
data_mix=agilex_franka_5data_manualvel_balance33_66_50
vlm_coco_data=${VLM_COCO_DATA:-./playground/Datasets/LLaVA-OneVision-COCO/llava_jsons/sharegpt4v_coco.json::./playground/Datasets/LLaVA-OneVision-COCO/images}
vlm_cc3m_data=${VLM_CC3M_DATA:-./playground/Datasets/LLaVA-ReCap-CC3M/data}
vlm_coco_repeat=5
vlm_data=${vlm_coco_data}
for _ in $(seq 2 ${vlm_coco_repeat}); do
  vlm_data+=,${vlm_coco_data}
done
vlm_data+=,${vlm_cc3m_data}
image_size_buckets='[[320,180],[280,210]]'
heads=oft,gr00t,pi
head_loss_weights=oft:1,gr00t:1,pi:1
run_id=qwen3_pretrain
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla_cotrain.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.disjoint_action_layout True \
  --framework.mask_padded_action_dims True \
  --framework.single_arm_loss_weight 8.0 \
  --framework.dual_arm_loss_weight 1.0 \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.heads ${heads} \
  --framework.head_loss_weights ${head_loss_weights} \
  --framework.action_model.repeated_diffusion_steps 2 \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.balance_dataset_weights True \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.video_backend pyav \
  --datasets.vla_data.sequential_step_sampling False \
  --datasets.vla_data.image_size_buckets "${image_size_buckets}" \
  --datasets.vla_data.prompt_prefix_embodiment True \
  --datasets.vlm_data.dataset_use "${vlm_data}" \
  --datasets.vlm_data.eval_dataset "" \
  --datasets.vlm_data.per_device_batch_size 2 \
  --trainer.shortest_angular_joint_loss True \
  --trainer.shortest_angular_joint_loss_diff True \
  --trainer.endpoint_wrap_loss_weight 0.5 \
  --trainer.loss_scale.vlm 0.2 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.learning_rate.base 1e-04 \
  --trainer.learning_rate.qwen_vl_interface 1e-05 \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Agilex_Franka \
  --wandb_entity "${WANDB_ENTITY:-your_name}"
