# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)



###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenPI_v4
freeze_module_list=''
base_vlm=StarVLA/Qwen3-VL-4B-Instruct-Action
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs_embodiment_prompt.yaml
data_root_dir=./playground/Datasets/RoboTwin-All
run_root_dir=./results/Checkpoints
data_mix=robotwin_all_wrap_32
action_chunk_size=32
future_action_window_size=31
image_size_buckets='[[320,180],[280,210]]'
run_id=vlact_robotwin_all_qwen3pi
pretrained_ckpt=./results/Checkpoints/qwen3_pretrain/checkpoints/steps_100000_pytorch_model.pt
# === End of environment variable configuration ===
###########################################################################################



export WANDB_MODE=${WANDB_MODE:-disabled}
output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.future_action_window_size ${future_action_window_size} \
  --framework.action_model.action_horizon ${action_chunk_size} \
  --framework.action_model.repeated_diffusion_steps 4 \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.balance_dataset_weights True \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.video_backend pyav \
  --datasets.vla_data.sequential_step_sampling False \
  --datasets.vla_data.image_size_buckets "${image_size_buckets}" \
  --trainer.shortest_angular_joint_loss True \
  --trainer.shortest_angular_joint_loss_diff True \
  --trainer.endpoint_wrap_loss_weight 0.5 \
  --trainer.pretrained_checkpoint ${pretrained_ckpt} \
  --trainer.random_init_action_model True \
  --trainer.reset_steps True \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.learning_rate.base 1e-4 \
  --trainer.learning_rate.qwen_vl_interface 1e-5 \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 20000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Robotwin \
  --wandb_entity "${WANDB_ENTITY:-your_name}"
