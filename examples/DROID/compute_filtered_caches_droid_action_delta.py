#!/usr/bin/env python3
"""Precompute filtered DROID caches (`json + pkl`) for training.

This script mirrors the training-time cache generation path but overrides the
delta-eef trajectory filter for DROID to use adjacent action deltas:

    delta_action[t] = action[t + 1] - action[t]

instead of the Franka-only `action_pose - state_pose` logic in the shared
dataset code. The generated cache file names still follow the exact
`LeRobotSingleDataset` naming scheme, so training can reuse them directly.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotSingleDataset,
    build_dataset_cache_key,
    _wrap_rotation_delta,
)
from starVLA.dataloader.lerobot_datasets import (
    _merge_robot_type_data_cfg_defaults,
    make_LeRobotSingleDataset,
)


DATA_ROOT = Path("/project/vonneumann1/wcy/copy/starVLA-VLAct/playground/Datasets")
DATA_NAME = "DROID"
ROBOT_TYPE = "oxe_droid_exterior1_wrist_manualvel_50"
DELETE_PAUSE_FRAME = False
DEFAULT_POSITION_LIMIT = 0.5
DEFAULT_ROTATION_LIMIT = 1.0
DEFAULT_VALID_RATIO = 0.5
DEFAULT_FILTER_INVALID_DROID_TASK = True
DEFAULT_NUM_WORKERS = 8

ACTION_KEY = "action"
POS_DIMS = slice(0, 3)
ROT_DIMS = slice(3, 6)

ORIGINAL_TRAJECTORY_FILTER = LeRobotSingleDataset._trajectory_has_large_delta_eef
ORIGINAL_GET_ALL_STEPS = LeRobotSingleDataset._get_all_steps
ORIGINAL_GET_METADATA = LeRobotSingleDataset._get_metadata
ORIGINAL_SET_TRANSFORMS_METADATA = LeRobotSingleDataset.set_transforms_metadata

WORKER_DATASET: LeRobotSingleDataset | None = None
WORKER_HAS_LANGUAGE_MODALITY = False
WORKER_LANGUAGE_KEY: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--data-name", type=str, default=DATA_NAME)
    parser.add_argument(
        "--robot-type",
        type=str,
        default=ROBOT_TYPE,
        choices=[
            "oxe_droid",
            "oxe_droid_exterior1_wrist",
            "oxe_droid_exterior2_wrist",
            "oxe_droid_exterior1_wrist_50",
            "oxe_droid_exterior2_wrist_50",
            "oxe_droid_q99",
            "oxe_droid_exterior1_wrist_q99",
            "oxe_droid_exterior2_wrist_q99",
            "oxe_droid_exterior1_wrist_q99_32",
            "oxe_droid_exterior2_wrist_q99_32",
            "oxe_droid_exterior1_wrist_q99_50",
            "oxe_droid_exterior2_wrist_q99_50",
            "oxe_droid_exterior1_wrist_manualvel_50",
            "oxe_droid_exterior2_wrist_manualvel_50",
            "oxe_droid_exterior1_wrist_manualvel_strict_50",
            "oxe_droid_exterior2_wrist_manualvel_strict_50",
        ],
        help="DROID robot config used to instantiate the dataset while building caches.",
    )
    parser.add_argument(
        "--delta-eef-position-abs-limit",
        type=float,
        default=DEFAULT_POSITION_LIMIT,
    )
    parser.add_argument(
        "--delta-eef-rotation-abs-limit",
        type=float,
        default=DEFAULT_ROTATION_LIMIT,
    )
    parser.add_argument(
        "--filter-delta-eef-steps",
        action="store_true",
        help="Filter individual steps whose action chunk contains delta-eef outliers.",
    )
    parser.add_argument(
        "--delta-eef-valid-ratio",
        type=float,
        default=DEFAULT_VALID_RATIO,
        help="Keep action chunks whose valid-step ratio is at least this value.",
    )
    parser.add_argument(
        "--filter-invalid-droid-task",
        action="store_true",
        default=DEFAULT_FILTER_INVALID_DROID_TASK,
        help="Match training-time invalid-task filtering for DROID.",
    )
    parser.add_argument(
        "--no-filter-invalid-droid-task",
        action="store_false",
        dest="filter_invalid_droid_task",
        help="Disable invalid-task filtering when building caches.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if both target cache files already exist and stats contain abs/delta/rel.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Parallel workers used to precompute the filtered steps cache.",
    )
    return parser.parse_args()


def build_data_cfg(
    data_root: Path,
    position_limit: float,
    rotation_limit: float,
    filter_invalid_droid_task: bool,
    filter_delta_eef_steps: bool = False,
    valid_ratio: float = DEFAULT_VALID_RATIO,
) -> dict:
    return {
        "data_root_dir": str(data_root),
        "data_mix": "droid_manualvel_50",
        "action_mode": "delta",
        "action_type": "delta_eef",
        "action_mode_reference": "action",
        "action_target_mode": "delta_eef_velocity",
        "prompt_prefix_fps": True,
        "video_backend": "pyav",
        "sequential_step_sampling": False,
        "filter_delta_eef_steps": bool(filter_delta_eef_steps),
        "filter_delta_eef_trajectory": not bool(filter_delta_eef_steps),
        "delta_eef_position_abs_limit": float(position_limit),
        "delta_eef_rotation_abs_limit": float(rotation_limit),
        "delta_eef_valid_ratio": float(valid_ratio),
        "filter_invalid_droid_task": bool(filter_invalid_droid_task),
        "delete_pause_frame": DELETE_PAUSE_FRAME,
    }


def build_effective_data_cfg(
    data_root: Path,
    robot_type: str,
    position_limit: float,
    rotation_limit: float,
    filter_invalid_droid_task: bool,
    filter_delta_eef_steps: bool = False,
    valid_ratio: float = DEFAULT_VALID_RATIO,
) -> dict:
    requested_cfg = build_data_cfg(
        data_root=data_root,
        position_limit=position_limit,
        rotation_limit=rotation_limit,
        filter_invalid_droid_task=filter_invalid_droid_task,
        filter_delta_eef_steps=filter_delta_eef_steps,
        valid_ratio=valid_ratio,
    )
    return _merge_robot_type_data_cfg_defaults(ROBOT_TYPE_CONFIG_MAP[robot_type], requested_cfg) or requested_cfg


def expected_paths(
    data_root: Path,
    data_name: str,
    robot_type: str,
    position_limit: float,
    rotation_limit: float,
    filter_invalid_droid_task: bool,
    filter_delta_eef_steps: bool = False,
    valid_ratio: float = DEFAULT_VALID_RATIO,
) -> tuple[Path, Path]:
    dataset_meta_dir = data_root / data_name / "meta"
    data_cfg = build_effective_data_cfg(
        data_root=data_root,
        robot_type=robot_type,
        position_limit=position_limit,
        rotation_limit=rotation_limit,
        filter_invalid_droid_task=filter_invalid_droid_task,
        filter_delta_eef_steps=filter_delta_eef_steps,
        valid_ratio=valid_ratio,
    )
    cache_key = build_dataset_cache_key(
        dataset_name=data_name,
        filter_outlier_trajectory=bool(data_cfg.get("filter_outlier_trajectory", False)),
        outlier_abs_limit=float(data_cfg.get("outlier_abs_limit", np.pi)),
        filter_gripper_outlier_trajectory=bool(data_cfg.get("filter_gripper_outlier_trajectory", False)),
        gripper_outlier_abs_limit=float(
            data_cfg.get("gripper_outlier_abs_limit", data_cfg.get("outlier_abs_limit", np.pi))
        ),
        filter_delta_eef_trajectory=bool(data_cfg.get("filter_delta_eef_trajectory", False)),
        filter_delta_eef_steps=bool(data_cfg.get("filter_delta_eef_steps", False)),
        delta_eef_position_abs_limit=float(data_cfg.get("delta_eef_position_abs_limit", position_limit)),
        delta_eef_rotation_abs_limit=float(data_cfg.get("delta_eef_rotation_abs_limit", rotation_limit)),
        delta_eef_valid_ratio=float(data_cfg.get("delta_eef_valid_ratio", valid_ratio)),
        delete_pause_frame=bool(data_cfg.get("delete_pause_frame", DELETE_PAUSE_FRAME)),
        embodiment_tag=robot_type,
        data_cfg=data_cfg,
    )
    steps_path = dataset_meta_dir / f"steps_data_index_filtered_{cache_key}.pkl"
    stats_path = dataset_meta_dir / f"stats_gr00t_filtered_{cache_key}.json"
    return steps_path, stats_path


def has_complete_filtered_cache(steps_path: Path, stats_path: Path) -> bool:
    if not steps_path.exists() or not stats_path.exists():
        return False
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception:
        return False
    return all(mode in stats for mode in ("abs", "delta", "rel"))


def droid_action_delta_trajectory_is_large(
    trajectory_df: pd.DataFrame,
    position_limit: float,
    rotation_limit: float,
) -> bool:
    if ACTION_KEY not in trajectory_df.columns:
        raise ValueError(f"Missing required column `{ACTION_KEY}` for DROID delta-action filtering.")

    action = np.stack(trajectory_df[ACTION_KEY].to_numpy()).astype(np.float32)
    if action.ndim == 1:
        action = action[:, None]
    if action.shape[0] < 2:
        return False
    if action.shape[1] < ROT_DIMS.stop:
        raise ValueError(
            "DROID delta-action filtering expects at least 6 action dimensions, "
            f"got {action.shape[1]}."
        )

    delta = action[1:] - action[:-1]
    delta[:, ROT_DIMS] = _wrap_rotation_delta(delta[:, ROT_DIMS])

    position_bad = np.any(np.abs(delta[:, POS_DIMS]) > position_limit)
    rotation_bad = np.any(np.abs(delta[:, ROT_DIMS]) > rotation_limit)
    return bool(position_bad or rotation_bad)


def patched_trajectory_filter(self: LeRobotSingleDataset, trajectory_df: pd.DataFrame) -> bool:
    if not self._filter_delta_eef_trajectory_enabled():
        return False
    if str(self.data_cfg.get("action_target_mode", "legacy")).lower() == "delta_eef_velocity":
        return ORIGINAL_TRAJECTORY_FILTER(self, trajectory_df)
    if not str(self.tag).startswith("oxe_droid"):
        return ORIGINAL_TRAJECTORY_FILTER(self, trajectory_df)
    return droid_action_delta_trajectory_is_large(
        trajectory_df=trajectory_df,
        position_limit=self._get_delta_eef_position_abs_limit(),
        rotation_limit=self._get_delta_eef_rotation_abs_limit(),
    )


def load_cache_metadata(steps_path: Path, stats_path: Path) -> tuple[dict | None, dict | None]:
    steps_obj = None
    stats_obj = None
    try:
        with open(steps_path, "rb") as f:
            steps_obj = pickle.load(f)
    except Exception:
        steps_obj = None
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_obj = json.load(f)
    except Exception:
        stats_obj = None
    return steps_obj, stats_obj


def _noop_get_all_steps(self: LeRobotSingleDataset) -> list[tuple[int, int]]:
    return []


def _noop_get_metadata(self, embodiment_tag) -> None:
    return None


def _noop_set_transforms_metadata(self, metadata, original_metadata=None) -> None:
    return None


def make_lightweight_dataset(
    data_root: Path,
    data_name: str,
    robot_type: str,
    position_limit: float,
    rotation_limit: float,
    filter_invalid_droid_task: bool,
    filter_delta_eef_steps: bool,
    valid_ratio: float,
) -> LeRobotSingleDataset:
    LeRobotSingleDataset._get_all_steps = _noop_get_all_steps
    LeRobotSingleDataset._get_metadata = _noop_get_metadata
    LeRobotSingleDataset.set_transforms_metadata = _noop_set_transforms_metadata
    lightweight_data_cfg = build_data_cfg(
        data_root=data_root,
        position_limit=position_limit,
        rotation_limit=rotation_limit,
        filter_invalid_droid_task=filter_invalid_droid_task,
        filter_delta_eef_steps=filter_delta_eef_steps,
        valid_ratio=valid_ratio,
    )
    # The lightweight path skips metadata construction, so disable robot-default
    # manual stats overrides that expect metadata.statistics to exist.
    lightweight_data_cfg["manual_action_normalization_statistics"] = {}
    try:
        return make_LeRobotSingleDataset(
            data_root_dir=data_root,
            data_name=data_name,
            robot_type=robot_type,
            delete_pause_frame=DELETE_PAUSE_FRAME,
            data_cfg=lightweight_data_cfg,
        )
    finally:
        LeRobotSingleDataset._get_all_steps = ORIGINAL_GET_ALL_STEPS
        LeRobotSingleDataset._get_metadata = ORIGINAL_GET_METADATA
        LeRobotSingleDataset.set_transforms_metadata = ORIGINAL_SET_TRANSFORMS_METADATA


def _split_trajectory_work(
    trajectory_ids: np.ndarray,
    trajectory_lengths: np.ndarray,
    num_workers: int,
) -> list[list[tuple[int, int]]]:
    num_items = int(len(trajectory_ids))
    if num_items == 0:
        return []
    worker_count = max(1, min(int(num_workers), num_items))
    chunk_size = (num_items + worker_count - 1) // worker_count
    chunks: list[list[tuple[int, int]]] = []
    for start in range(0, num_items, chunk_size):
        end = min(start + chunk_size, num_items)
        chunks.append(
            [
                (int(trajectory_ids[idx]), int(trajectory_lengths[idx]))
                for idx in range(start, end)
            ]
        )
    return chunks


def _compute_steps_for_chunk(
    chunk: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int, int, int, int, int]:
    dataset = WORKER_DATASET
    if dataset is None:
        raise RuntimeError("Worker dataset is not initialized.")

    all_steps: list[tuple[int, int]] = []
    skipped_trajectories = 0
    filtered_steps = 0
    processed_trajectories = 0
    semantic_filtered_trajectories = 0
    semantic_filtered_steps = 0

    for trajectory_id, trajectory_length in chunk:
        try:
            data = dataset.get_trajectory_data(trajectory_id)
            trajectory_skipped = False
            valid_step_mask = None

            if dataset._filter_outlier_trajectory_enabled() and dataset._trajectory_has_outlier_action(data):
                skipped_trajectories += 1
                trajectory_skipped = True
                continue

            if dataset._filter_delta_eef_trajectory_enabled() and dataset._trajectory_has_large_delta_eef(data):
                skipped_trajectories += 1
                trajectory_skipped = True
                continue

            if dataset._trajectory_has_invalid_droid_task(data):
                skipped_trajectories += 1
                semantic_filtered_trajectories += 1
                semantic_filtered_steps += int(trajectory_length)
                trajectory_skipped = True
                continue

            if dataset._filter_delta_eef_steps_enabled():
                valid_step_mask = dataset._get_valid_delta_eef_step_mask(data, int(trajectory_length))
                filtered_steps += int(len(valid_step_mask) - np.count_nonzero(valid_step_mask))
                if not np.any(valid_step_mask):
                    skipped_trajectories += 1
                    trajectory_skipped = True
                    continue

            if WORKER_HAS_LANGUAGE_MODALITY and WORKER_LANGUAGE_KEY is not None:
                dataset.curr_traj_data = data
                language_instruction = dataset.get_language(trajectory_id, WORKER_LANGUAGE_KEY, 0)
                if not language_instruction or language_instruction[0] == "":
                    skipped_trajectories += 1
                    semantic_filtered_trajectories += 1
                    semantic_filtered_steps += int(trajectory_length)
                    trajectory_skipped = True
                    continue
        except Exception as e:
            print(f"Skipping trajectory {trajectory_id} due to read error: {e}")
            skipped_trajectories += 1
            trajectory_skipped = True
            continue

        if not trajectory_skipped:
            processed_trajectories += 1

        if valid_step_mask is None:
            for base_index in range(int(trajectory_length)):
                all_steps.append((int(trajectory_id), int(base_index)))
        else:
            for base_index in np.flatnonzero(valid_step_mask):
                all_steps.append((int(trajectory_id), int(base_index)))

    return (
        all_steps,
        processed_trajectories,
        skipped_trajectories,
        filtered_steps,
        semantic_filtered_trajectories,
        semantic_filtered_steps,
    )


def build_filtered_steps_cache(
    *,
    data_root: Path,
    data_name: str,
    robot_type: str,
    position_limit: float,
    rotation_limit: float,
    filter_invalid_droid_task: bool,
    filter_delta_eef_steps: bool,
    valid_ratio: float,
    num_workers: int,
) -> Path:
    global WORKER_DATASET, WORKER_HAS_LANGUAGE_MODALITY, WORKER_LANGUAGE_KEY

    dataset = make_lightweight_dataset(
        data_root=data_root,
        data_name=data_name,
        robot_type=robot_type,
        position_limit=position_limit,
        rotation_limit=rotation_limit,
        filter_invalid_droid_task=filter_invalid_droid_task,
        filter_delta_eef_steps=filter_delta_eef_steps,
        valid_ratio=valid_ratio,
    )
    WORKER_DATASET = dataset
    WORKER_HAS_LANGUAGE_MODALITY = (
        "language" in dataset.modality_keys and len(dataset.modality_keys["language"]) > 0
    )
    WORKER_LANGUAGE_KEY = dataset.modality_keys["language"][0] if WORKER_HAS_LANGUAGE_MODALITY else None

    steps_path = dataset._get_steps_cache_path()
    config_key = dataset._get_steps_config_key()
    if steps_path.exists():
        try:
            with open(steps_path, "rb") as f:
                existing_steps = pickle.load(f)
            if (
                isinstance(existing_steps, dict)
                and existing_steps.get("config_key") == config_key
                and "steps" in existing_steps
            ):
                print(f"[SKIP] Existing filtered steps cache found: {steps_path}")
                return steps_path
        except Exception as e:
            print(f"Failed to load existing steps cache ({e}); rebuilding {steps_path}")

    chunks = _split_trajectory_work(dataset.trajectory_ids, dataset.trajectory_lengths, num_workers)

    all_steps: list[tuple[int, int]] = []
    processed_trajectories = 0
    skipped_trajectories = 0
    filtered_steps = 0
    semantic_filtered_trajectories = 0
    semantic_filtered_steps = 0
    original_step_count = int(np.sum(dataset.trajectory_lengths))

    if not chunks:
        raise ValueError(f"No trajectories found in dataset {data_name}")

    if num_workers <= 1 or len(chunks) <= 1:
        results = [_compute_steps_for_chunk(chunk) for chunk in chunks]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(int(num_workers), len(chunks))) as pool:
            results = list(pool.imap(_compute_steps_for_chunk, chunks))

    for (
        steps_chunk,
        processed_chunk,
        skipped_chunk,
        filtered_chunk,
        semantic_traj_chunk,
        semantic_step_chunk,
    ) in results:
        all_steps.extend(steps_chunk)
        processed_trajectories += int(processed_chunk)
        skipped_trajectories += int(skipped_chunk)
        filtered_steps += int(filtered_chunk)
        semantic_filtered_trajectories += int(semantic_traj_chunk)
        semantic_filtered_steps += int(semantic_step_chunk)

    cache_data = {
        "config_key": config_key,
        "steps": all_steps,
        "num_trajectories": len(dataset.trajectory_ids),
        "filtered_trajectory_count": len({int(trajectory_id) for trajectory_id, _ in all_steps}),
        "total_steps": len(all_steps),
        "original_step_count": int(np.sum(dataset.trajectory_lengths)),
        "computed_timestamp": pd.Timestamp.now().isoformat(),
        "delete_pause_frame": dataset.delete_pause_frame,
        "semantic_filtered_trajectory_count": int(semantic_filtered_trajectories),
        "semantic_filtered_step_count": int(semantic_filtered_steps),
        "post_semantic_step_count": int(original_step_count - semantic_filtered_steps),
        "delta_eef_filtered_step_count": int(filtered_steps),
    }

    steps_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = steps_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(steps_path)

    print(
        f"Precomputed filtered steps with {num_workers} worker(s): "
        f"processed {processed_trajectories}, skipped {skipped_trajectories}"
    )
    print(
        "Semantic/language filtering removed "
        f"{semantic_filtered_trajectories} instruction trajectories "
        f"and {semantic_filtered_steps} candidate action chunks"
    )
    if filter_delta_eef_steps:
        print(f"Step-level delta-eef filtering removed {filtered_steps} steps")
    print(
        f"Action chunks: original={original_step_count}, "
        f"after_semantic={original_step_count - semantic_filtered_steps}, "
        f"remaining={len(all_steps)}"
    )
    print(f"Total steps: {len(all_steps)} from {len(dataset.trajectory_ids)} trajectories")

    WORKER_DATASET = None
    WORKER_HAS_LANGUAGE_MODALITY = False
    WORKER_LANGUAGE_KEY = None

    return steps_path


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    steps_path, stats_path = expected_paths(
        data_root=data_root,
        data_name=args.data_name,
        robot_type=args.robot_type,
        position_limit=args.delta_eef_position_abs_limit,
        rotation_limit=args.delta_eef_rotation_abs_limit,
        filter_invalid_droid_task=args.filter_invalid_droid_task,
        filter_delta_eef_steps=args.filter_delta_eef_steps,
        valid_ratio=args.delta_eef_valid_ratio,
    )

    print(f"Dataset: {args.data_name}")
    print(f"Robot type: {args.robot_type}")
    print(f"Position limit: {args.delta_eef_position_abs_limit}")
    print(f"Rotation limit: {args.delta_eef_rotation_abs_limit}")
    print(f"Valid ratio: {args.delta_eef_valid_ratio}")
    print(f"Filter invalid task: {args.filter_invalid_droid_task}")
    print(f"Filter delta-eef steps: {args.filter_delta_eef_steps}")
    print(f"Workers: {args.num_workers}")
    print(f"Expected steps cache: {steps_path}")
    print(f"Expected stats cache: {stats_path}")

    if args.skip_existing and has_complete_filtered_cache(steps_path, stats_path):
        print("[SKIP] Existing complete filtered caches found.")
        return

    LeRobotSingleDataset._trajectory_has_large_delta_eef = patched_trajectory_filter
    try:
        steps_path = build_filtered_steps_cache(
            data_root=data_root,
            data_name=args.data_name,
            robot_type=args.robot_type,
            position_limit=args.delta_eef_position_abs_limit,
            rotation_limit=args.delta_eef_rotation_abs_limit,
            filter_invalid_droid_task=args.filter_invalid_droid_task,
            filter_delta_eef_steps=args.filter_delta_eef_steps,
            valid_ratio=args.delta_eef_valid_ratio,
            num_workers=args.num_workers,
        )
        dataset = make_LeRobotSingleDataset(
            data_root_dir=data_root,
            data_name=args.data_name,
            robot_type=args.robot_type,
            delete_pause_frame=DELETE_PAUSE_FRAME,
            data_cfg=build_data_cfg(
                data_root=data_root,
                position_limit=args.delta_eef_position_abs_limit,
                rotation_limit=args.delta_eef_rotation_abs_limit,
                filter_invalid_droid_task=args.filter_invalid_droid_task,
                filter_delta_eef_steps=args.filter_delta_eef_steps,
                valid_ratio=args.delta_eef_valid_ratio,
            ),
        )
        steps_path = dataset._get_steps_cache_path()
        stats_path = dataset._get_stats_cache_path()
    finally:
        LeRobotSingleDataset._trajectory_has_large_delta_eef = ORIGINAL_TRAJECTORY_FILTER

    missing = []
    if not steps_path.exists():
        missing.append(steps_path.name)
    if not stats_path.exists():
        missing.append(stats_path.name)
    if missing:
        raise FileNotFoundError(f"Missing outputs after cache generation: {missing}")

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    if not all(mode in stats for mode in ("abs", "delta", "rel")):
        raise ValueError(f"Stats file missing required modes: {stats_path}")

    steps_obj, stats_obj = load_cache_metadata(steps_path, stats_path)
    kept_trajectories = (
        stats_obj.get("__filtered_trajectory_count", steps_obj.get("filtered_trajectory_count"))
        if stats_obj and steps_obj
        else None
    )
    original_trajectories = (
        stats_obj.get("__original_trajectory_count", steps_obj.get("num_trajectories"))
        if stats_obj and steps_obj
        else None
    )
    total_steps = steps_obj.get("total_steps") if steps_obj else None
    semantic_filtered_trajectories = (
        steps_obj.get("semantic_filtered_trajectory_count") if steps_obj else None
    )
    semantic_filtered_steps = steps_obj.get("semantic_filtered_step_count") if steps_obj else None
    post_semantic_steps = steps_obj.get("post_semantic_step_count") if steps_obj else None

    print(f"[DONE] Saved steps cache to {steps_path}")
    print(f"[DONE] Saved stats cache to {stats_path}")
    if kept_trajectories is not None and original_trajectories is not None:
        print(f"Trajectories kept: {kept_trajectories}/{original_trajectories}")
    if semantic_filtered_trajectories is not None and semantic_filtered_steps is not None:
        print(
            "Semantic/language filtering removed "
            f"{semantic_filtered_trajectories} instruction trajectories "
            f"and {semantic_filtered_steps} candidate action chunks"
        )
    if steps_obj is not None and total_steps is not None:
        print(
            f"Action chunks: original={steps_obj.get('original_step_count')}, "
            f"after_semantic={post_semantic_steps}, remaining={total_steps}"
        )
    if total_steps is not None:
        print(f"Remaining filtered steps: {total_steps}")


if __name__ == "__main__":
    main()
