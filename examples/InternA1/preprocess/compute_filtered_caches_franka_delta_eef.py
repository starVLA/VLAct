#!/usr/bin/env python3
"""Batch-generate filtered training caches for InternA1 Franka datasets.

This script mirrors the training-time cache generation for the filtered Franka
setup used by `exps/0411_oft_interna1_franka_filter/train.sh`.

For each Franka task it generates:
- `meta/steps_data_index_filtered_<hash>.pkl`
- `meta/stats_gr00t_filtered_<hash>.json`

Correctness strategy:
- instantiate the exact same `LeRobotSingleDataset` that training uses
- rely on the dataset implementation to build cache file names and contents
- verify both output files exist and that the stats file contains abs/delta/rel
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.datasets import build_dataset_cache_key
from starVLA.dataloader.lerobot_datasets import (
    _merge_robot_type_data_cfg_defaults,
    make_LeRobotSingleDataset,
)

DATA_ROOT = Path("/project/vonneumann1/wcy/code/starVLA-dev/playground/Datasets/InternData-A1")
TASK_LIST = DATA_ROOT / "franka_tasks.txt"
ROBOT_TYPE = "interna1_franka_manualvel_50"
NUM_WORKERS = 64
DELETE_PAUSE_FRAME = False
SKIP_EXISTING = False
DELTA_EEF_POSITION_ABS_LIMIT = 1.5
DELTA_EEF_ROTATION_ABS_LIMIT = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--task-list", type=Path, default=TASK_LIST)
    parser.add_argument("--robot-type", type=str, default=ROBOT_TYPE)
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
        "--skip-existing",
        action="store_true",
        help="Skip tasks whose filtered stats file already contains abs/delta/rel.",
    )
    parser.add_argument(
        "--task-prefixes",
        nargs="*",
        default=None,
        help="Optional prefixes to limit processed task names.",
    )
    return parser.parse_args()


def build_data_cfg(
    data_root: Path,
    delta_eef_position_abs_limit: float,
    delta_eef_rotation_abs_limit: float,
) -> dict:
    return {
        "data_root_dir": str(data_root),
        "data_mix": "interna1_franka_manualvel_50",
        "action_mode": "delta",
        "action_type": "delta_eef",
        "action_mode_reference": "action",
        "action_target_mode": "delta_eef_velocity",
        "prompt_prefix_fps": True,
        "video_backend": "pyav",
        "sequential_step_sampling": False,
        "filter_delta_eef_trajectory": True,
        "delta_eef_position_abs_limit": delta_eef_position_abs_limit,
        "delta_eef_rotation_abs_limit": delta_eef_rotation_abs_limit,
        "delete_pause_frame": DELETE_PAUSE_FRAME,
    }


def build_effective_data_cfg(
    data_root: Path,
    robot_type: str,
    delta_eef_position_abs_limit: float,
    delta_eef_rotation_abs_limit: float,
) -> dict:
    requested_cfg = build_data_cfg(
        data_root=data_root,
        delta_eef_position_abs_limit=delta_eef_position_abs_limit,
        delta_eef_rotation_abs_limit=delta_eef_rotation_abs_limit,
    )
    return _merge_robot_type_data_cfg_defaults(ROBOT_TYPE_CONFIG_MAP[robot_type], requested_cfg) or requested_cfg


def build_filter_cache_key(task: str, data_cfg: dict, robot_type: str) -> str:
    return build_dataset_cache_key(
        dataset_name=task,
        filter_outlier_trajectory=bool(data_cfg.get("filter_outlier_trajectory", False)),
        outlier_abs_limit=float(data_cfg.get("outlier_abs_limit", 3.1415926)),
        filter_gripper_outlier_trajectory=bool(data_cfg.get("filter_gripper_outlier_trajectory", False)),
        gripper_outlier_abs_limit=(
            float(data_cfg["gripper_outlier_abs_limit"])
            if data_cfg.get("gripper_outlier_abs_limit", None) is not None
            else None
        ),
        filter_delta_eef_trajectory=bool(data_cfg.get("filter_delta_eef_trajectory", False)),
        delta_eef_position_abs_limit=float(
            data_cfg.get("delta_eef_position_abs_limit", DELTA_EEF_POSITION_ABS_LIMIT)
        ),
        delta_eef_rotation_abs_limit=float(
            data_cfg.get("delta_eef_rotation_abs_limit", DELTA_EEF_ROTATION_ABS_LIMIT)
        ),
        delete_pause_frame=bool(data_cfg.get("delete_pause_frame", DELETE_PAUSE_FRAME)),
        embodiment_tag=robot_type,
        data_cfg=data_cfg,
    )


def expected_paths(
    task: str,
    data_root: Path,
    robot_type: str,
    delta_eef_position_abs_limit: float,
    delta_eef_rotation_abs_limit: float,
) -> tuple[Path, Path]:
    dataset_meta_dir = data_root / task / "meta"
    data_cfg = build_effective_data_cfg(
        data_root=data_root,
        robot_type=robot_type,
        delta_eef_position_abs_limit=delta_eef_position_abs_limit,
        delta_eef_rotation_abs_limit=delta_eef_rotation_abs_limit,
    )
    cache_key = build_filter_cache_key(task, data_cfg, robot_type)
    steps_path = dataset_meta_dir / f"steps_data_index_filtered_{cache_key}.pkl"
    stats_path = dataset_meta_dir / f"stats_gr00t_filtered_{cache_key}.json"
    return steps_path, stats_path


def has_complete_filtered_cache(
    task: str,
    data_root: Path,
    robot_type: str,
    delta_eef_position_abs_limit: float,
    delta_eef_rotation_abs_limit: float,
) -> bool:
    steps_path, stats_path = expected_paths(
        task=task,
        data_root=data_root,
        robot_type=robot_type,
        delta_eef_position_abs_limit=delta_eef_position_abs_limit,
        delta_eef_rotation_abs_limit=delta_eef_rotation_abs_limit,
    )
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
        import pickle

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


def list_tasks(task_list: Path, task_prefixes: list[str] | None) -> list[str]:
    with open(task_list, "r", encoding="utf-8") as f:
        tasks = [line.strip() for line in f if line.strip()]
    if task_prefixes:
        tasks = [task for task in tasks if any(task.startswith(prefix) for prefix in task_prefixes)]
    return tasks


def compute_one(task: str) -> str:
    steps_path, stats_path = expected_paths(
        task=task,
        data_root=DATA_ROOT,
        robot_type=ROBOT_TYPE,
        delta_eef_position_abs_limit=DELTA_EEF_POSITION_ABS_LIMIT,
        delta_eef_rotation_abs_limit=DELTA_EEF_ROTATION_ABS_LIMIT,
    )
    if SKIP_EXISTING and has_complete_filtered_cache(
        task=task,
        data_root=DATA_ROOT,
        robot_type=ROBOT_TYPE,
        delta_eef_position_abs_limit=DELTA_EEF_POSITION_ABS_LIMIT,
        delta_eef_rotation_abs_limit=DELTA_EEF_ROTATION_ABS_LIMIT,
    ):
        return f"[SKIP] {task}"

    try:
        dataset = make_LeRobotSingleDataset(
            data_root_dir=DATA_ROOT,
            data_name=task,
            robot_type=ROBOT_TYPE,
            delete_pause_frame=DELETE_PAUSE_FRAME,
            data_cfg=build_data_cfg(
                data_root=DATA_ROOT,
                delta_eef_position_abs_limit=DELTA_EEF_POSITION_ABS_LIMIT,
                delta_eef_rotation_abs_limit=DELTA_EEF_ROTATION_ABS_LIMIT,
            ),
        )
        steps_path = dataset._get_steps_cache_path()
        stats_path = dataset._get_stats_cache_path()
    except Exception as e:
        return f"[FAIL] {task}: {e}"

    missing = []
    if not steps_path.exists():
        missing.append(steps_path.name)
    if not stats_path.exists():
        missing.append(stats_path.name)
    if missing:
        return f"[FAIL] {task}: missing outputs {missing}"

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception as e:
        return f"[FAIL] {task}: failed to read stats json ({e})"
    if not all(mode in stats for mode in ("abs", "delta", "rel")):
        return f"[FAIL] {task}: stats json missing abs/delta/rel"

    steps_obj, stats_obj = load_cache_metadata(steps_path, stats_path)
    if steps_obj is not None and stats_obj is not None:
        total_steps = int(steps_obj.get("total_steps", -1))
        filtered_trajectory_count = int(
            stats_obj.get("__filtered_trajectory_count", steps_obj.get("filtered_trajectory_count", -1))
        )
        if total_steps == 0 or filtered_trajectory_count == 0:
            return (
                f"[DONE-EMPTY] {task}: kept "
                f"{stats_obj.get('__filtered_trajectory_count', filtered_trajectory_count)}/"
                f"{stats_obj.get('__original_trajectory_count', steps_obj.get('num_trajectories', '?'))} trajectories"
            )

    return f"[DONE] {task}"


def main() -> None:
    global DATA_ROOT
    global ROBOT_TYPE
    global SKIP_EXISTING
    global DELTA_EEF_POSITION_ABS_LIMIT
    global DELTA_EEF_ROTATION_ABS_LIMIT

    args = parse_args()
    DATA_ROOT = args.data_root.resolve()
    ROBOT_TYPE = str(args.robot_type)
    task_list = args.task_list.resolve()
    if task_list == TASK_LIST.resolve() and not task_list.exists():
        task_list = DATA_ROOT / "franka_tasks.txt"
    SKIP_EXISTING = args.skip_existing
    DELTA_EEF_POSITION_ABS_LIMIT = args.delta_eef_position_abs_limit
    DELTA_EEF_ROTATION_ABS_LIMIT = args.delta_eef_rotation_abs_limit

    tasks = list_tasks(task_list, args.task_prefixes)
    print(f"Total tasks: {len(tasks)}")
    print(f"Using {args.num_workers} worker processes across tasks")
    print(f"Robot type: {ROBOT_TYPE}")
    print(f"Task list: {task_list}")
    print(f"Delta EEF position abs limit: {DELTA_EEF_POSITION_ABS_LIMIT}")
    print(f"Delta EEF rotation abs limit: {DELTA_EEF_ROTATION_ABS_LIMIT}")

    with Pool(args.num_workers) as pool:
        for idx, msg in enumerate(pool.imap_unordered(compute_one, tasks), start=1):
            print(f"[{idx}/{len(tasks)}] {msg}")
    print("All done.")


if __name__ == "__main__":
    main()
