#!/usr/bin/env python3
"""Compute DROID `stats_gr00t.json` with multi-process abs-stat parallelism.

This script mirrors the StarVLA cache format:
  {
    "abs": ...,
    "delta": ...,
    "rel": ...
  }

If `meta/modality.json` is missing, it will be generated automatically for the
current DROID dataset layout.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.datasets import (
    build_dataset_cache_key,
    calculate_action_only_delta_statistics,
    calculate_action_only_rel_statistics,
    is_invalid_droid_task_text,
)
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata


ALL_LOW_DIM_DATA: pd.DataFrame | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/project/vonneumann1/wcy/code/starVLA-dev/playground/Datasets/DROID"),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <dataset-dir>/meta/stats_gr00t.json",
    )
    parser.add_argument(
        "--overwrite-modality",
        action="store_true",
        help="Overwrite auto-generated modality.json even if it already exists.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip when output already contains abs/delta/rel.",
    )
    parser.add_argument(
        "--filter-invalid-droid-task",
        action="store_true",
        help="Exclude trajectories whose task text is unknown_task or N/A.",
    )
    return parser.parse_args()


def has_complete_modes(stats_path: Path) -> bool:
    if not stats_path.exists():
        return False
    try:
        with open(stats_path) as f:
            stats = json.load(f)
    except Exception:
        return False
    return all(mode in stats for mode in ("abs", "delta", "rel"))


def load_existing_stats(stats_path: Path) -> dict:
    if not stats_path.exists():
        return {}
    try:
        with open(stats_path) as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def list_parquet_paths(dataset_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in dataset_dir.glob("data/*/*.parquet")
        if "episode_033675.parquet" not in p.name
    )


def load_invalid_task_indices(dataset_dir: Path) -> set[int]:
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    with open(tasks_path, "r") as f:
        tasks = [json.loads(line) for line in f]
    invalid_indices = set()
    for task in tasks:
        for key in ("task_name", "task"):
            if key in task and is_invalid_droid_task_text(task[key]):
                invalid_indices.add(int(task["task_index"]))
                break
    return invalid_indices


def filter_parquet_paths_by_task(parquet_paths: list[Path], dataset_dir: Path) -> list[Path]:
    invalid_task_indices = load_invalid_task_indices(dataset_dir)
    if not invalid_task_indices:
        return parquet_paths

    filtered_paths: list[Path] = []
    for parquet_path in tqdm(parquet_paths, desc="Filtering invalid-task trajectories"):
        task_series = pd.read_parquet(parquet_path, columns=["task_index"])["task_index"]
        if task_series.empty:
            continue
        task_index_value = task_series.iloc[0]
        task_index = int(task_index_value if isinstance(task_index_value, (int, float)) else task_index_value.item())
        if task_index in invalid_task_indices:
            continue
        filtered_paths.append(parquet_path)
    return filtered_paths


def resolve_output_path(dataset_dir: Path, explicit_output: Path | None, filter_invalid_droid_task: bool) -> Path:
    if explicit_output is not None:
        return explicit_output
    if not filter_invalid_droid_task:
        return dataset_dir / "meta" / "stats_gr00t.json"
    data_cfg = {"filter_invalid_droid_task": True}
    cache_key = build_dataset_cache_key(
        dataset_name=dataset_dir.name,
        filter_outlier_trajectory=False,
        outlier_abs_limit=float(np.pi),
        embodiment_tag="oxe_droid",
        data_cfg=data_cfg,
    )
    return dataset_dir / "meta" / f"stats_gr00t_filtered_{cache_key}.json"


def _init_worker(all_low_dim_data: pd.DataFrame) -> None:
    global ALL_LOW_DIM_DATA
    ALL_LOW_DIM_DATA = all_low_dim_data


def _compute_one_modality(le_modality: str) -> tuple[str, dict] | None:
    global ALL_LOW_DIM_DATA
    if ALL_LOW_DIM_DATA is None:
        raise RuntimeError("Worker dataframe is not initialized.")
    if "task_info" in le_modality:
        return None

    try:
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in ALL_LOW_DIM_DATA[le_modality]]
        )
    except Exception as e:
        print(f"Warning: Failed to process modality {le_modality} due to error: {e}")
        return None

    stats = {
        "mean": np.mean(np_data, axis=0).tolist(),
        "std": np.std(np_data, axis=0).tolist(),
        "min": np.min(np_data, axis=0).tolist(),
        "max": np.max(np_data, axis=0).tolist(),
        "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
        "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
    }
    return le_modality, stats


def calculate_dataset_statistics_mp(parquet_paths: list[Path], num_workers: int = 8) -> dict:
    all_low_dim_data_list = []
    for parquet_path in tqdm(sorted(parquet_paths), desc="Collecting all parquet files..."):
        all_low_dim_data_list.append(pd.read_parquet(parquet_path))

    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    modalities = [col for col in all_low_dim_data.columns if "task_info" not in col]

    ctx = mp.get_context("fork")
    dataset_statistics = {}
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(all_low_dim_data,),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_compute_one_modality, modalities),
            total=len(modalities),
            desc=f"Processing modalities ({num_workers} proc)",
        ):
            if result is None:
                continue
            le_modality, stats = result
            dataset_statistics[le_modality] = stats

    return dataset_statistics


def build_droid_modality(info: dict) -> dict:
    features = info["features"]

    required_video_keys = {
        "observation.images.exterior_1",
        "observation.images.exterior_2",
        "observation.images.wrist",
    }
    missing_video_keys = sorted(required_video_keys - set(features))
    if missing_video_keys:
        raise ValueError(f"Missing DROID video features: {missing_video_keys}")

    state_dim = int(features["observation.state"]["shape"][0])
    action_dim = int(features["action"]["shape"][0])
    if state_dim != 8:
        raise ValueError(f"Unexpected DROID state dim: {state_dim}, expected 8")
    if action_dim != 7:
        raise ValueError(f"Unexpected DROID action dim: {action_dim}, expected 7")

    modality = {
        "state": {
            "joint_position": {
                "start": 0,
                "end": 7,
                "dtype": "float32",
                "original_key": "observation.state",
            },
            "gripper_position": {
                "start": 7,
                "end": 8,
                "dtype": "float32",
                "original_key": "observation.state",
            },
        },
        "action": {
            "eef_position": {
                "start": 0,
                "end": 3,
                "dtype": "float32",
                "original_key": "action",
            },
            "eef_rotation": {
                "start": 3,
                "end": 6,
                "dtype": "float32",
                "rotation_type": "euler_angles_rpy",
                "original_key": "action",
            },
            "gripper_position": {
                "start": 6,
                "end": 7,
                "dtype": "float32",
                "original_key": "action",
            },
        },
        "video": {
            "exterior_image_1": {"original_key": "observation.images.exterior_1"},
            "exterior_image_2": {"original_key": "observation.images.exterior_2"},
            "wrist_image": {"original_key": "observation.images.wrist"},
        },
        "annotation": {
            "language.language_instruction": {"original_key": "task_index"},
        },
    }
    return modality


def ensure_modality_json(dataset_dir: Path, overwrite: bool = False) -> Path:
    modality_path = dataset_dir / "meta" / "modality.json"
    if modality_path.exists() and not overwrite:
        return modality_path

    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info file: {info_path}")

    with open(info_path) as f:
        info = json.load(f)

    modality = build_droid_modality(info)
    modality_path.parent.mkdir(parents=True, exist_ok=True)
    with open(modality_path, "w") as f:
        json.dump(modality, f, indent=2)
    print(f"Wrote modality metadata to {modality_path}")
    return modality_path


def build_action_metadata(modality_path: Path) -> tuple[LeRobotModalityMetadata, list[str], list[int]]:
    with open(modality_path) as f:
        lerobot_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))

    modality_configs = ROBOT_TYPE_CONFIG_MAP["oxe_droid"].modality_config()
    action_keys_full = list(modality_configs["action"].modality_keys)
    action_indices = list(modality_configs["action"].delta_indices)
    return lerobot_modality_meta, action_keys_full, action_indices


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_path = resolve_output_path(dataset_dir, args.output, args.filter_invalid_droid_task)

    if args.skip_existing and has_complete_modes(output_path):
        print(f"[SKIP] Existing complete stats found: {output_path}")
        return

    parquet_paths = list_parquet_paths(dataset_dir)
    if args.filter_invalid_droid_task:
        parquet_paths = filter_parquet_paths_by_task(parquet_paths, dataset_dir)
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}")

    modality_path = ensure_modality_json(dataset_dir, overwrite=args.overwrite_modality)
    lerobot_modality_meta, action_keys_full, action_indices = build_action_metadata(modality_path)
    delta_apply_keys = ["action.eef_position", "action.eef_rotation"]
    existing_stats = load_existing_stats(output_path)

    abs_stats = existing_stats.get("abs")
    if abs_stats is None:
        abs_stats = calculate_dataset_statistics_mp(parquet_paths, num_workers=args.num_workers)
    else:
        print(f"Reusing existing abs stats from {output_path}")
    delta_stats = calculate_action_only_delta_statistics(
        parquet_paths=parquet_paths,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        action_indices=action_indices,
        action_mode_apply_keys=delta_apply_keys,
        base_stats=abs_stats,
    )
    rel_stats = calculate_action_only_rel_statistics(
        parquet_paths=parquet_paths,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        action_indices=action_indices,
        action_mode_apply_keys=delta_apply_keys,
        base_stats=abs_stats,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"abs": abs_stats, "delta": delta_stats, "rel": rel_stats}, f, indent=2)

    print(f"Saved stats to {output_path}")


if __name__ == "__main__":
    main()
