# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)



###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenPI_v4
freeze_module_list=''
base_vlm=StarVLA/Qwen3-VL-4B-Instruct-Action
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero_embodiment_prompt.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=vlact_libero_all_qwen3pi
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
  --framework.action_model.repeated_diffusion_steps 2 \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.video_backend pyav \
  --datasets.vla_data.sequential_step_sampling False \
  --trainer.pretrained_checkpoint ${pretrained_ckpt} \
  --trainer.random_init_action_model True \
  --trainer.reset_steps True \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.learning_rate.base 1e-4 \
  --trainer.learning_rate.qwen_vl_interface 1e-5 \
  --trainer.max_train_steps 50000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity "${WANDB_ENTITY:-your_name}"
