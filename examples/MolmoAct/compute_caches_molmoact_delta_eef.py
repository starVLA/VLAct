#!/usr/bin/env python3
"""Batch-generate MolmoAct cache files (`pkl + json`) for training.

This script follows the same idea as the DROID / InternA1 cache builders, but
targets the converted MolmoAct v2.1 datasets such as:

  - molmoact_household_v21
  - molmoact_tabletop_v21

It supports two cache modes:

1. raw
   - meta/steps_data_index.pkl
   - meta/stats_gr00t.json

2. filtered
   - meta/steps_data_index_filtered_<hash>.pkl
   - meta/stats_gr00t_filtered_<hash>.json

The filtered mode uses the same delta-eef velocity thresholds used elsewhere in
this repo:

  - position abs limit: 0.5
  - rotation abs limit: 1.0
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.datasets import build_dataset_cache_key
from starVLA.dataloader.lerobot_datasets import (
    _merge_robot_type_data_cfg_defaults,
    make_LeRobotSingleDataset,
)


DATA_ROOT = Path("/project/vonneumann1/wcy/copy/starVLA-VLAct/playground/Datasets/MolmoAct-Dataset")
DEFAULT_DATASETS = ["molmoact_household_v21", "molmoact_tabletop_v21"]
DEFAULT_ROBOT_TYPES = [
    "molmoact_franka_exterior1_wrist_manualvel_50",
    "molmoact_franka_exterior2_wrist_manualvel_50",
]
DELETE_PAUSE_FRAME = False
DELTA_EEF_POSITION_ABS_LIMIT = 0.5
DELTA_EEF_ROTATION_ABS_LIMIT = 1.0
DELTA_EEF_VALID_RATIO = 0.5
NUM_WORKERS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--dataset-names",
        nargs="*",
        default=None,
        help="Dataset directories under data-root. Defaults to discovered molmoact_*_v21 directories.",
    )
    parser.add_argument(
        "--robot-types",
        nargs="*",
        default=None,
        help="Robot types to build caches for. Defaults to the two 2-view MolmoAct robot types.",
    )
    parser.add_argument(
        "--mode",
        choices=["raw", "filtered", "both"],
        default="both",
        help="Which cache variant to generate.",
    )
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--delta-eef-position-abs-limit",
        type=float,
        default=DELTA_EEF_POSITION_ABS_LIMIT,
    )
    parser.add_argument(
        "--delta-eef-rotation-abs-limit",
        type=float,
        default=DELTA_EEF_ROTATION_ABS_LIMIT,
    )
    parser.add_argument(
        "--filter-delta-eef-steps",
        action="store_true",
        help="Filter individual steps whose action chunk contains delta-eef outliers.",
    )
    parser.add_argument(
        "--delta-eef-valid-ratio",
        type=float,
        default=DELTA_EEF_VALID_RATIO,
        help="Keep action chunks whose valid-step ratio is at least this value.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cache variants whose target pkl/json already exist and stats contain abs/delta/rel.",
    )
    return parser.parse_args()


def discover_dataset_names(data_root: Path) -> list[str]:
    discovered = []
    if not data_root.exists():
        return discovered
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith("molmoact_") or not child.name.endswith("_v21"):
            continue
        if (child / "meta" / "info.json").exists():
            discovered.append(child.name)
    return discovered


def resolve_dataset_names(data_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    discovered = discover_dataset_names(data_root)
    if discovered:
        return discovered
    return list(DEFAULT_DATASETS)


def resolve_robot_types(requested: list[str] | None) -> list[str]:
    robot_types = requested or list(DEFAULT_ROBOT_TYPES)
    for robot_type in robot_types:
        if robot_type not in ROBOT_TYPE_CONFIG_MAP:
            raise KeyError(f"Unknown robot type: {robot_type}")
    return robot_types


def build_base_data_cfg(data_root: Path) -> dict[str, object]:
    return {
        "data_root_dir": str(data_root),
        "action_type": "delta_eef",
        "action_mode": "delta",
        "action_mode_reference": "action",
        "action_target_mode": "delta_eef_velocity",
        "prompt_prefix_fps": True,
        "video_backend": "pyav",
        "sequential_step_sampling": False,
    }


def build_raw_data_cfg(data_root: Path) -> dict[str, object]:
    return dict(build_base_data_cfg(data_root))


def build_filtered_data_cfg(
    data_root: Path,
    delta_eef_position_abs_limit: float,
    delta_eef_rotation_abs_limit: float,
    filter_delta_eef_steps: bool = False,
    delta_eef_valid_ratio: float = DELTA_EEF_VALID_RATIO,
) -> dict[str, object]:
    data_cfg = build_base_data_cfg(data_root)
    data_cfg.update(
        {
            "filter_delta_eef_steps": bool(filter_delta_eef_steps),
            "filter_delta_eef_trajectory": not bool(filter_delta_eef_steps),
            "delta_eef_position_abs_limit": float(delta_eef_position_abs_limit),
            "delta_eef_rotation_abs_limit": float(delta_eef_rotation_abs_limit),
            "delta_eef_valid_ratio": float(delta_eef_valid_ratio),
        }
    )
    return data_cfg


def build_effective_data_cfg(data_root: Path, robot_type: str, data_cfg: dict[str, object]) -> dict[str, object]:
    return _merge_robot_type_data_cfg_defaults(ROBOT_TYPE_CONFIG_MAP[robot_type], data_cfg) or data_cfg


def expected_paths(
    dataset_name: str,
    data_root: Path,
    robot_type: str,
    mode: str,
    delta_eef_position_abs_limit: float,
    delta_eef_rotation_abs_limit: float,
    filter_delta_eef_steps: bool = False,
    delta_eef_valid_ratio: float = DELTA_EEF_VALID_RATIO,
) -> tuple[Path, Path]:
    dataset_meta_dir = data_root / dataset_name / "meta"
    if mode == "raw":
        return dataset_meta_dir / "steps_data_index.pkl", dataset_meta_dir / "stats_gr00t.json"

    data_cfg = build_effective_data_cfg(
        data_root=data_root,
        robot_type=robot_type,
        data_cfg=build_filtered_data_cfg(
            data_root=data_root,
            delta_eef_position_abs_limit=delta_eef_position_abs_limit,
            delta_eef_rotation_abs_limit=delta_eef_rotation_abs_limit,
            filter_delta_eef_steps=filter_delta_eef_steps,
            delta_eef_valid_ratio=delta_eef_valid_ratio,
        ),
    )
    cache_key = build_dataset_cache_key(
        dataset_name=dataset_name,
        filter_outlier_trajectory=bool(data_cfg.get("filter_outlier_trajectory", False)),
        outlier_abs_limit=float(data_cfg.get("outlier_abs_limit", np.pi)),
        filter_gripper_outlier_trajectory=bool(data_cfg.get("filter_gripper_outlier_trajectory", False)),
        gripper_outlier_abs_limit=float(
            data_cfg.get("gripper_outlier_abs_limit", data_cfg.get("outlier_abs_limit", np.pi))
        ),
        filter_delta_eef_trajectory=bool(data_cfg.get("filter_delta_eef_trajectory", False)),
        filter_delta_eef_steps=bool(data_cfg.get("filter_delta_eef_steps", False)),
        delta_eef_position_abs_limit=float(
            data_cfg.get("delta_eef_position_abs_limit", delta_eef_position_abs_limit)
        ),
        delta_eef_rotation_abs_limit=float(
            data_cfg.get("delta_eef_rotation_abs_limit", delta_eef_rotation_abs_limit)
        ),
        delta_eef_valid_ratio=float(data_cfg.get("delta_eef_valid_ratio", delta_eef_valid_ratio)),
        delete_pause_frame=bool(data_cfg.get("delete_pause_frame", DELETE_PAUSE_FRAME)),
        embodiment_tag=robot_type,
        data_cfg=data_cfg,
    )
    return (
        dataset_meta_dir / f"steps_data_index_filtered_{cache_key}.pkl",
        dataset_meta_dir / f"stats_gr00t_filtered_{cache_key}.json",
    )


def has_complete_cache(steps_path: Path, stats_path: Path) -> bool:
    if not steps_path.exists() or not stats_path.exists():
        return False
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception:
        return False
    return all(mode in stats for mode in ("abs", "delta", "rel"))


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


def build_mode_data_cfg(
    data_root: Path,
    mode: str,
    delta_eef_position_abs_limit: float,
    delta_eef_rotation_abs_limit: float,
    filter_delta_eef_steps: bool = False,
    delta_eef_valid_ratio: float = DELTA_EEF_VALID_RATIO,
) -> dict[str, object]:
    if mode == "raw":
        return build_raw_data_cfg(data_root)
    if mode == "filtered":
        return build_filtered_data_cfg(
            data_root=data_root,
            delta_eef_position_abs_limit=delta_eef_position_abs_limit,
            delta_eef_rotation_abs_limit=delta_eef_rotation_abs_limit,
            filter_delta_eef_steps=filter_delta_eef_steps,
            delta_eef_valid_ratio=delta_eef_valid_ratio,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def compute_one_mode(
    dataset_name: str,
    data_root: Path,
    robot_type: str,
    mode: str,
    delta_eef_position_abs_limit: float,
    delta_eef_rotation_abs_limit: float,
    skip_existing: bool,
    filter_delta_eef_steps: bool,
    delta_eef_valid_ratio: float,
) -> str:
    dataset_dir = data_root / dataset_name
    if not dataset_dir.exists():
        return f"[FAIL] {dataset_name} [{robot_type}] ({mode}): dataset directory missing"

    steps_path, stats_path = expected_paths(
        dataset_name=dataset_name,
        data_root=data_root,
        robot_type=robot_type,
        mode=mode,
        delta_eef_position_abs_limit=delta_eef_position_abs_limit,
        delta_eef_rotation_abs_limit=delta_eef_rotation_abs_limit,
        filter_delta_eef_steps=filter_delta_eef_steps,
        delta_eef_valid_ratio=delta_eef_valid_ratio,
    )
    if skip_existing and has_complete_cache(steps_path, stats_path):
        return f"[SKIP] {dataset_name} [{robot_type}] ({mode})"

    data_cfg = build_mode_data_cfg(
        data_root=data_root,
        mode=mode,
        delta_eef_position_abs_limit=delta_eef_position_abs_limit,
        delta_eef_rotation_abs_limit=delta_eef_rotation_abs_limit,
        filter_delta_eef_steps=filter_delta_eef_steps,
        delta_eef_valid_ratio=delta_eef_valid_ratio,
    )
    try:
        dataset = make_LeRobotSingleDataset(
            data_root_dir=data_root,
            data_name=dataset_name,
            robot_type=robot_type,
            delete_pause_frame=DELETE_PAUSE_FRAME,
            data_cfg=data_cfg,
        )
        steps_path = dataset._get_steps_cache_path()
        stats_path = dataset._get_stats_cache_path()
    except Exception as e:
        return f"[FAIL] {dataset_name} [{robot_type}] ({mode}): {e}"

    missing = []
    if not steps_path.exists():
        missing.append(steps_path.name)
    if not stats_path.exists():
        missing.append(stats_path.name)
    if missing:
        return f"[FAIL] {dataset_name} [{robot_type}] ({mode}): missing outputs {missing}"

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception as e:
        return f"[FAIL] {dataset_name} [{robot_type}] ({mode}): failed to read stats json ({e})"
    if not all(key in stats for key in ("abs", "delta", "rel")):
        return f"[FAIL] {dataset_name} [{robot_type}] ({mode}): stats json missing abs/delta/rel"

    steps_obj, stats_obj = load_cache_metadata(steps_path, stats_path)
    if steps_obj is None:
        return f"[DONE] {dataset_name} [{robot_type}] ({mode})"

    kept_trajectories = (
        stats_obj.get("__filtered_trajectory_count", steps_obj.get("filtered_trajectory_count"))
        if stats_obj
        else steps_obj.get("filtered_trajectory_count")
    )
    original_trajectories = (
        stats_obj.get("__original_trajectory_count", steps_obj.get("num_trajectories"))
        if stats_obj
        else steps_obj.get("num_trajectories")
    )
    total_steps = steps_obj.get("total_steps")

    semantic_filtered_trajectories = steps_obj.get("semantic_filtered_trajectory_count") if steps_obj else None
    semantic_filtered_steps = steps_obj.get("semantic_filtered_step_count") if steps_obj else None
    post_semantic_steps = steps_obj.get("post_semantic_step_count") if steps_obj else None

    if mode == "filtered" and steps_obj is not None:
        parts = [f"[DONE] {dataset_name} [{robot_type}] ({mode})"]
        if kept_trajectories is not None and original_trajectories is not None and total_steps is not None:
            parts.append(f"kept {kept_trajectories}/{original_trajectories} trajectories, {total_steps} steps")
        if semantic_filtered_trajectories is not None and semantic_filtered_steps is not None:
            parts.append(
                "semantic_filtered="
                f"{semantic_filtered_trajectories} instructions/{semantic_filtered_steps} chunks"
            )
        if total_steps is not None:
            parts.append(
                "action_chunks="
                f"{steps_obj.get('original_step_count')}->{post_semantic_steps}->{total_steps}"
            )
        return ": ".join([parts[0], ", ".join(parts[1:])]) if len(parts) > 1 else parts[0]
    if kept_trajectories is not None and original_trajectories is not None and total_steps is not None:
        return (
            f"[DONE] {dataset_name} [{robot_type}] ({mode}): kept {kept_trajectories}/{original_trajectories} "
            f"trajectories, {total_steps} steps"
        )
    return f"[DONE] {dataset_name} [{robot_type}] ({mode})"


def compute_one(task: tuple[str, Path, str, str, float, float, bool, bool, float]) -> list[str]:
    (
        dataset_name,
        data_root,
        robot_type,
        mode,
        delta_eef_position_abs_limit,
        delta_eef_rotation_abs_limit,
        skip_existing,
        filter_delta_eef_steps,
        delta_eef_valid_ratio,
    ) = task

    modes = ["raw", "filtered"] if mode == "both" else [mode]
    return [
        compute_one_mode(
            dataset_name=dataset_name,
            data_root=data_root,
            robot_type=robot_type,
            mode=single_mode,
            delta_eef_position_abs_limit=delta_eef_position_abs_limit,
            delta_eef_rotation_abs_limit=delta_eef_rotation_abs_limit,
            skip_existing=skip_existing,
            filter_delta_eef_steps=filter_delta_eef_steps,
            delta_eef_valid_ratio=delta_eef_valid_ratio,
        )
        for single_mode in modes
    ]


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    dataset_names = resolve_dataset_names(data_root, args.dataset_names)
    robot_types = resolve_robot_types(args.robot_types)

    if not dataset_names:
        raise FileNotFoundError(f"No MolmoAct dataset directories found under {data_root}")

    print(f"Data root: {data_root}")
    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"Robot types: {', '.join(robot_types)}")
    print(f"Mode: {args.mode}")
    print(f"Workers: {args.num_workers}")
    print(f"Delta EEF position abs limit: {args.delta_eef_position_abs_limit}")
    print(f"Delta EEF rotation abs limit: {args.delta_eef_rotation_abs_limit}")
    print(f"Delta EEF valid ratio: {args.delta_eef_valid_ratio}")
    print(f"Filter delta-eef steps: {args.filter_delta_eef_steps}")

    tasks = [
        (
            dataset_name,
            data_root,
            robot_type,
            str(args.mode),
            float(args.delta_eef_position_abs_limit),
            float(args.delta_eef_rotation_abs_limit),
            bool(args.skip_existing),
            bool(args.filter_delta_eef_steps),
            float(args.delta_eef_valid_ratio),
        )
        for dataset_name in dataset_names
        for robot_type in robot_types
    ]

    if args.num_workers <= 1 or len(tasks) <= 1:
        for idx, task in enumerate(tasks, start=1):
            for message in compute_one(task):
                print(f"[{idx}/{len(tasks)}] {message}")
    else:
        with Pool(args.num_workers) as pool:
            for idx, messages in enumerate(pool.imap_unordered(compute_one, tasks), start=1):
                for message in messages:
                    print(f"[{idx}/{len(tasks)}] {message}")

    print("All done.")


if __name__ == "__main__":
    main()
