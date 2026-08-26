# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)



###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=StarVLA/Qwen3-VL-4B-Instruct
config_yaml=./examples/DOMINO/train_files/starvla_train_domino.yaml
data_root_dir=./playground/Datasets/DOMINO
run_root_dir=./results/Checkpoints
data_mix=domino_clean_wrap
action_chunk_size=32
future_action_window_size=31
image_size_buckets='[[320,180],[280,210]]'
run_id=vlact_domino_qwen3oft
pretrained_ckpt=./results/Checkpoints/vlact_qwen3_pretrain/checkpoints/steps_100000_pytorch_model.pt
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
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.video_backend pyav \
  --datasets.vla_data.sequential_step_sampling False \
  --datasets.vla_data.image_size_buckets "${image_size_buckets}" \
  --trainer.shortest_angular_joint_loss True \
  --trainer.pretrained_checkpoint ${pretrained_ckpt} \
  --trainer.random_init_action_model True \
  --trainer.reset_steps True \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.learning_rate.base 1e-4 \
  --trainer.learning_rate.qwen_vl_interface 1e-5 \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_DOMINO \
  --wandb_entity "${WANDB_ENTITY:-your_name}"
