#!/usr/bin/env python3
"""Prepare the `DROID_100` dataset by mirroring the DROID preprocessing flow.

This script is intended for the RoboInter-annotated DROID subset used in this
workspace. It does three lightweight setup steps:

1. create/update `playground/Datasets/DROID_100`
2. reuse heavy `data/` and `videos/` via symlinks
3. write DROID-like `meta/info.json` and `meta/modality.json`

Optionally, it can also compute a fresh `meta/stats_gr00t.json` that matches
the custom `DROID_100` modality layout.

Example:
    python examples/DROID/prepare_droid_100.py --compute-stats --skip-existing
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
    calculate_action_only_delta_statistics,
    calculate_action_only_rel_statistics,
)
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata

DEFAULT_SOURCE_DIR = Path(
    "/project/vonneumann1/datasets/RoboInter-Data/Annotation_with_action_lerobotv21/lerobot_droid_anno"
)
DEFAULT_DATASET_DIR = REPO_ROOT / "playground" / "Datasets" / "DROID_100"
DEFAULT_REPO_ID = "local/droid_100_v21"
DEFAULT_ROBOT_TYPE = "oxe_droid_exterior1_wrist_manualvel_50"

ALL_LOW_DIM_DATA: pd.DataFrame | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--robot-type", type=str, default=DEFAULT_ROBOT_TYPE)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        help="Compute meta/stats_gr00t.json after preparing the dataset wrapper.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stats generation when meta/stats_gr00t.json already contains abs/delta/rel.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_symlink(target: Path, source: Path) -> None:
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        target.unlink()
    elif target.exists():
        raise FileExistsError(f"{target} already exists and is not a symlink.")
    target.symlink_to(source, target_is_directory=source.is_dir())


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def list_parquet_paths(dataset_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in (dataset_dir / "data").glob("chunk-*/*.parquet")
        if "episode_033675.parquet" not in p.name
    )


def has_complete_modes(stats_path: Path) -> bool:
    if not stats_path.exists():
        return False
    try:
        stats = load_json(stats_path)
    except Exception:
        return False
    return all(mode in stats for mode in ("abs", "delta", "rel"))


def build_droid_100_info(source_info: dict, repo_id: str) -> dict:
    info = dict(source_info)
    info["repo_id"] = repo_id
    return info


def build_droid_100_modality(info: dict) -> dict:
    features = info["features"]
    required_keys = {
        "action",
        "other_information.observation_joint_position",
        "other_information.observation_gripper_position",
        "observation.images.primary",
        "observation.images.wrist",
        "task_index",
    }
    missing = sorted(required_keys - set(features))
    if missing:
        raise ValueError(f"Missing required DROID_100 features: {missing}")

    action_dim = int(features["action"]["shape"][0])
    joint_dim = int(features["other_information.observation_joint_position"]["shape"][0])
    gripper_dim = int(features["other_information.observation_gripper_position"]["shape"][0])
    if action_dim != 7:
        raise ValueError(f"Unexpected action dim: {action_dim}, expected 7")
    if joint_dim != 7:
        raise ValueError(f"Unexpected observation joint dim: {joint_dim}, expected 7")
    if gripper_dim != 1:
        raise ValueError(f"Unexpected observation gripper dim: {gripper_dim}, expected 1")

    return {
        "state": {
            "joint_position": {
                "start": 0,
                "end": 7,
                "dtype": "float64",
                "original_key": "other_information.observation_joint_position",
            },
            "gripper_position": {
                "start": 0,
                "end": 1,
                "dtype": "float64",
                "original_key": "other_information.observation_gripper_position",
            },
        },
        "action": {
            "eef_position": {
                "start": 0,
                "end": 3,
                "dtype": "float64",
                "original_key": "action",
            },
            "eef_rotation": {
                "start": 3,
                "end": 6,
                "dtype": "float64",
                "rotation_type": "euler_angles_rpy",
                "original_key": "action",
            },
            "gripper_position": {
                "start": 6,
                "end": 7,
                "dtype": "float64",
                "original_key": "action",
            },
        },
        "video": {
            "exterior_image_1": {"original_key": "observation.images.primary"},
            "wrist_image": {"original_key": "observation.images.wrist"},
        },
        "annotation": {
            "language.language_instruction": {"original_key": "task_index"},
        },
    }


def prepare_dataset_layout(source_dir: Path, dataset_dir: Path, repo_id: str) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source dataset: {source_dir}")

    ensure_dir(dataset_dir)
    ensure_dir(dataset_dir / "meta")
    ensure_symlink(dataset_dir / "data", source_dir / "data")
    ensure_symlink(dataset_dir / "videos", source_dir / "videos")

    for name in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
        ensure_symlink(dataset_dir / "meta" / name, source_dir / "meta" / name)

    source_info = load_json(source_dir / "meta" / "info.json")
    write_json(dataset_dir / "meta" / "info.json", build_droid_100_info(source_info, repo_id))
    write_json(dataset_dir / "meta" / "modality.json", build_droid_100_modality(source_info))


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


def build_action_metadata(
    modality_path: Path, robot_type: str
) -> tuple[LeRobotModalityMetadata, list[str], list[int]]:
    with modality_path.open("r", encoding="utf-8") as f:
        lerobot_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))

    modality_configs = ROBOT_TYPE_CONFIG_MAP[robot_type].modality_config()
    action_keys_full = list(modality_configs["action"].modality_keys)
    action_indices = list(modality_configs["action"].delta_indices)
    return lerobot_modality_meta, action_keys_full, action_indices


def compute_stats(dataset_dir: Path, robot_type: str, num_workers: int, skip_existing: bool) -> Path:
    stats_path = dataset_dir / "meta" / "stats_gr00t.json"
    if skip_existing and has_complete_modes(stats_path):
        print(f"[SKIP] Existing complete stats found: {stats_path}")
        return stats_path

    parquet_paths = list_parquet_paths(dataset_dir)
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}")

    modality_path = dataset_dir / "meta" / "modality.json"
    lerobot_modality_meta, action_keys_full, action_indices = build_action_metadata(
        modality_path, robot_type
    )

    abs_stats = calculate_dataset_statistics_mp(parquet_paths, num_workers=num_workers)
    delta_stats = calculate_action_only_delta_statistics(
        parquet_paths=parquet_paths,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        action_indices=action_indices,
        action_mode_apply_keys=["action.eef_position", "action.eef_rotation"],
        base_stats=abs_stats,
    )
    rel_stats = calculate_action_only_rel_statistics(
        parquet_paths=parquet_paths,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        action_indices=action_indices,
        action_mode_apply_keys=["action.eef_position", "action.eef_rotation"],
        base_stats=abs_stats,
    )

    write_json(stats_path, {"abs": abs_stats, "delta": delta_stats, "rel": rel_stats})
    return stats_path


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()

    print(f"Source dataset: {source_dir}")
    print(f"Target dataset: {dataset_dir}")
    prepare_dataset_layout(source_dir, dataset_dir, repo_id=args.repo_id)
    print("[DONE] Prepared dataset layout and metadata.")

    if args.compute_stats:
        stats_path = compute_stats(
            dataset_dir=dataset_dir,
            robot_type=args.robot_type,
            num_workers=args.num_workers,
            skip_existing=args.skip_existing,
        )
        print(f"[DONE] Saved stats to {stats_path}")
    else:
        print("[INFO] Stats generation skipped. Use --compute-stats to build meta/stats_gr00t.json.")


if __name__ == "__main__":
    main()
