#!/usr/bin/env python3
"""Compute complete stats_gr00t.json files for all split_aloha datasets.

This script mirrors the training-time statistics generation and writes all
three action modes expected by the dataloader: abs, delta, and rel.
"""
import json
import sys
from glob import glob
from pathlib import Path
from multiprocessing import Pool

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.datasets import (
    calculate_dataset_statistics,
    calculate_delta_action_statistics,
    calculate_rel_action_statistics,
)
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata

DATA_ROOT = "/project/vonneumann1/datasets/InternData-A1"
TASK_LIST = "/project/vonneumann1/wcy/code/starVLA-dev/playground/Datasets/InternData-A1/split_aloha_tasks.txt"
NUM_WORKERS = 64
ROBOT_TYPE = "split_aloha"


def has_complete_modes(stats_path: Path) -> bool:
    if not stats_path.exists():
        return False
    try:
        with open(stats_path) as f:
            stats = json.load(f)
    except Exception:
        return False
    return all(mode in stats for mode in ("abs", "delta", "rel"))


def compute_one(task: str) -> str:
    dataset_dir = Path(DATA_ROOT) / task
    out_path = dataset_dir / "meta" / "stats_gr00t.json"
    if has_complete_modes(out_path):
        return f"[SKIP] {task}"

    parquet_paths = sorted(
        Path(p) for p in glob(str(dataset_dir / "data" / "*" / "*.parquet"))
        if "episode_033675.parquet" not in p
    )
    if not parquet_paths:
        return f"[NO DATA] {task}"

    modality_path = dataset_dir / "meta" / "modality.json"
    with open(modality_path) as f:
        lerobot_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))

    modality_configs = ROBOT_TYPE_CONFIG_MAP[ROBOT_TYPE].modality_config()
    action_keys_full = list(modality_configs["action"].modality_keys)
    state_keys_full = list(modality_configs["state"].modality_keys)
    action_indices = list(modality_configs["action"].delta_indices)
    state_indices = list(modality_configs["state"].delta_indices)

    abs_stats = calculate_dataset_statistics(parquet_paths)
    delta_stats = calculate_delta_action_statistics(
        parquet_paths=parquet_paths,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        state_keys_full=state_keys_full,
        action_indices=action_indices,
        state_indices=state_indices,
        action_mode_apply_keys=action_keys_full,
        action_mode_state_map={},
        base_stats=abs_stats,
    )
    rel_stats = calculate_rel_action_statistics(
        parquet_paths=parquet_paths,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        state_keys_full=state_keys_full,
        action_indices=action_indices,
        state_indices=state_indices,
        action_mode_apply_keys=action_keys_full,
        action_mode_state_map={},
        base_stats=abs_stats,
    )

    stats = {
        "abs": abs_stats,
        "delta": delta_stats,
        "rel": rel_stats,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return f"[DONE] {task}"


if __name__ == "__main__":
    with open(TASK_LIST) as f:
        tasks = [l.strip() for l in f if l.strip()]
    print(f"Total tasks: {len(tasks)}")

    with Pool(NUM_WORKERS) as pool:
        for i, msg in enumerate(pool.imap_unordered(compute_one, tasks)):
            print(f"[{i+1}/{len(tasks)}] {msg}")
    print("All done.")
