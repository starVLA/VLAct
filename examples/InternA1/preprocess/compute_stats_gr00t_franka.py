#!/usr/bin/env python3
"""Compute stats_gr00t.json for all InternA1 Franka datasets."""

import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.datasets import (
    calculate_action_only_delta_statistics,
    calculate_action_only_rel_statistics,
    calculate_dataset_statistics,
)
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata

DATA_ROOT = Path("/project/vonneumann1/datasets/InternData-A1")
TASK_LIST = (
    Path("/project/vonneumann1/wcy/code/starVLA-dev/playground/Datasets/InternData-A1")
    / "franka_tasks.txt"
)
NUM_WORKERS = 64
ROBOT_TYPE = "interna1_franka_q99"
ACTION_MODE_APPLY_KEYS = ["action.eef_position", "action.eef_rotation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--task-list", type=Path, default=TASK_LIST)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def has_complete_modes(stats_path: Path) -> bool:
    if not stats_path.exists():
        return False
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception:
        return False
    return all(mode in stats for mode in ("abs", "delta", "rel"))


def list_parquet_paths(dataset_dir: Path) -> list[Path]:
    data_dir = dataset_dir / "data"
    return sorted(
        p
        for p in data_dir.rglob("*.parquet")
        if p.is_file() and p.name != "episode_033675.parquet"
    )


def compute_one(task: str, data_root: Path, overwrite: bool) -> str:
    dataset_dir = data_root / task
    stats_path = dataset_dir / "meta" / "stats_gr00t.json"
    modality_path = dataset_dir / "meta" / "modality.json"
    if stats_path.exists() and not overwrite and has_complete_modes(stats_path):
        return f"[SKIP] {task}"
    if not modality_path.exists():
        return f"[NO MODALITY] {task}"

    parquet_paths = list_parquet_paths(dataset_dir)
    if not parquet_paths:
        return f"[NO DATA] {task}"

    with open(modality_path, "r", encoding="utf-8") as f:
        lerobot_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))

    modality_configs = ROBOT_TYPE_CONFIG_MAP[ROBOT_TYPE].modality_config()
    action_keys_full = list(modality_configs["action"].modality_keys)
    action_indices = list(modality_configs["action"].delta_indices)

    abs_stats = calculate_dataset_statistics(parquet_paths)
    delta_stats = calculate_action_only_delta_statistics(
        parquet_paths=parquet_paths,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        action_indices=action_indices,
        action_mode_apply_keys=ACTION_MODE_APPLY_KEYS,
        base_stats=abs_stats,
    )
    rel_stats = calculate_action_only_rel_statistics(
        parquet_paths=parquet_paths,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        action_indices=action_indices,
        action_mode_apply_keys=ACTION_MODE_APPLY_KEYS,
        base_stats=abs_stats,
    )

    stats = {
        "abs": abs_stats,
        "delta": delta_stats,
        "rel": rel_stats,
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return f"[DONE] {task}"


def main() -> None:
    args = parse_args()
    with open(args.task_list, "r", encoding="utf-8") as f:
        tasks = [line.strip() for line in f if line.strip()]

    print(f"Total tasks: {len(tasks)}")
    worker = partial(compute_one, data_root=args.data_root, overwrite=args.overwrite)
    with Pool(args.num_workers) as pool:
        for idx, msg in enumerate(pool.imap_unordered(worker, tasks), start=1):
            print(f"[{idx}/{len(tasks)}] {msg}")
    print("All done.")


if __name__ == "__main__":
    main()
