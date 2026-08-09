#!/usr/bin/env python3
"""Compute `stats_gr00t.json` with 8-process abs-stat parallelism.

This script keeps the original `delta` / `rel` implementations unchanged and
only parallelizes the expensive `abs` modality-stat computation for one dataset.

Example:
python examples/RoboCoin/compute_stats_gr00t_mp.py \
  --dataset-dir ./playground/Datasets/RoboCOIN/Cobot_Magic_clean_blackboard \
  --num-workers 8
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

from starVLA.dataloader.gr00t_lerobot.datasets import (
    calculate_delta_action_statistics,
    calculate_rel_action_statistics,
)
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata


ALL_LOW_DIM_DATA: pd.DataFrame | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <dataset-dir>/meta/stats_gr00t.json",
    )
    return parser.parse_args()


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


def list_parquet_paths(dataset_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in dataset_dir.glob("data/*/*.parquet")
        if "episode_033675.parquet" not in p.name
    )


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


def build_action_metadata(modality_path: Path) -> tuple[LeRobotModalityMetadata, list[str], list[str], dict[str, str]]:
    with open(modality_path) as f:
        lerobot_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))

    action_keys_full = [f"action.{key}" for key in lerobot_modality_meta.action.keys()]
    state_keys_full = [f"state.{key}" for key in lerobot_modality_meta.state.keys()]
    action_mode_state_map = {
        f"action.{key}": f"state.{key}"
        for key in lerobot_modality_meta.action.keys()
        if key in lerobot_modality_meta.state
    }
    return lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_state_map


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_path = args.output or (dataset_dir / "meta" / "stats_gr00t.json")
    modality_path = dataset_dir / "meta" / "modality.json"
    parquet_paths = list_parquet_paths(dataset_dir)

    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}")
    if not modality_path.exists():
        raise FileNotFoundError(f"Missing modality file: {modality_path}")

    lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_state_map = build_action_metadata(
        modality_path
    )

    action_indices = list(range(16))
    state_indices = [0]

    abs_stats = calculate_dataset_statistics_mp(parquet_paths, num_workers=args.num_workers)
    delta_stats = calculate_delta_action_statistics(
        parquet_paths=parquet_paths,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        state_keys_full=state_keys_full,
        action_indices=action_indices,
        state_indices=state_indices,
        action_mode_apply_keys=action_keys_full,
        action_mode_state_map=action_mode_state_map,
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
        action_mode_state_map=action_mode_state_map,
        base_stats=abs_stats,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"abs": abs_stats, "delta": delta_stats, "rel": rel_stats}, f, indent=2)

    print(f"Saved stats to {output_path}")


if __name__ == "__main__":
    main()
