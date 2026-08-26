# Qwen3 pre-training

Pre-training jointly uses VLA robot data and VLM image-text data. Run all commands from the repository root.

## 1. Download the base model

```bash
huggingface-cli download StarVLA/Qwen3-VL-4B-Instruct \
  --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

Set `base_vlm` in the training script to this local directory if offline training is required.

## 2. Prepare VLM data

```bash
huggingface-cli download StarVLA/LLaVA-OneVision-COCO \
  --repo-type dataset \
  --local-dir playground/Datasets/LLaVA-OneVision-COCO
unzip playground/Datasets/LLaVA-OneVision-COCO/sharegpt4v_coco.zip \
  -d playground/Datasets/LLaVA-OneVision-COCO
```

The expected files are:

```text
playground/Datasets/LLaVA-OneVision-COCO/
├── images/
└── llava_jsons/sharegpt4v_coco.json
```

Prepare CC3M as Qwen/LLaVA JSON, JSONL, or Parquet data under
`playground/Datasets/LLaVA-ReCap-CC3M/data`. Override both paths when needed:

```bash
export VLM_COCO_DATA=/path/to/coco.json::/path/to/coco/images
export VLM_CC3M_DATA=/path/to/processed/cc3m
```

## 3. Prepare VLA data

Place the downloaded LeRobot v2.1 datasets under:

```text
playground/Datasets/
├── InternData-A1/
├── RoboCOIN/
├── DROID/
├── DROID_100/
└── MolmoAct-Dataset/
```

Each dataset must contain `data/`, `videos/`, `meta/info.json`, and
`meta/modality.json`. The selected mixture is
`agilex_franka_5data_manualvel_balance_50`.

Before training, calculate each dataset's sampling index and statistics. These
commands produce `meta/steps_data_index*.pkl` and
`meta/stats_gr00t*.json`; `--skip-existing` makes reruns safe.

```bash
# InternData-A1
python examples/InternA1/preprocess/compute_caches_split_aloha.py \
  --data-root playground/Datasets/InternData-A1 \
  --task-list playground/Datasets/InternData-A1/split_aloha_tasks.txt \
  --robot-type split_aloha_flip_wrap50 --skip-existing

# RoboCOIN
python examples/RoboCoin/compute_stats_gr00t.py \
  --data-root playground/Datasets/RoboCOIN \
  --robot-type ROBOCOIN.AgileX_flip_wrap --skip-existing

# DROID: both camera configurations
for robot_type in \
  oxe_droid_exterior1_wrist_manualvel_strict_50 \
  oxe_droid_exterior2_wrist_manualvel_strict_50; do
  python examples/DROID/compute_filtered_caches_droid_action_delta.py \
    --data-root playground/Datasets --data-name DROID \
    --robot-type "${robot_type}" --filter-delta-eef-steps --skip-existing
done

# DROID_100
python examples/DROID/compute_filtered_caches_droid_action_delta.py \
  --data-root playground/Datasets --data-name DROID_100 \
  --robot-type oxe_droid_exterior1_wrist_manualvel_strict_50 \
  --filter-delta-eef-steps --skip-existing

# MolmoAct: both camera configurations
python examples/MolmoAct/compute_caches_molmoact_delta_eef.py \
  --data-root playground/Datasets/MolmoAct-Dataset --mode filtered \
  --robot-types \
    molmoact_franka_exterior1_wrist_manualvel_strict_50 \
    molmoact_franka_exterior2_wrist_manualvel_strict_50 \
  --filter-delta-eef-steps --skip-existing
```

## 4. Start training

```bash
# Eight GPUs on one node
bash scripts/run_scripts/Pretrain/pretrain_qwen3_single_node.sh

# Multi-node Slurm
sbatch scripts/run_scripts/Pretrain/pretrain_qwen3_slurm.sh
```

Check `data_root_dir`, batch size, GPU count, and Slurm resources before launch.
