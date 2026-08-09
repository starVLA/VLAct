#!/usr/bin/env python3
"""Batch-generate training caches for InternA1 split_aloha datasets.

This mirrors the training-time cache generation path and supports wrap/flip-wrap
robot configs, so variant-specific caches can be precomputed in parallel.
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

DATA_ROOT = Path("/project/vonneumann1/datasets/InternData-A1")
TASK_LIST = DATA_ROOT / "split_aloha_tasks.txt"
ROBOT_TYPE = "split_aloha_flip_wrap50"
NUM_WORKERS = 16
DELETE_PAUSE_FRAME = False
SKIP_EXISTING = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--task-list", type=Path, default=TASK_LIST)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--robot-type",
        type=str,
        default=ROBOT_TYPE,
        choices=[
            "split_aloha",
            "split_aloha32",
            "split_aloha50",
            "split_aloha_wrap",
            "split_aloha_wrap32",
            "split_aloha_wrap50",
            "split_aloha_flip_wrap",
            "split_aloha_flip_wrap32",
            "split_aloha_flip_wrap50",
        ],
        help="Robot config to use when generating caches.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks whose target cache already contains abs/delta/rel.",
    )
    parser.add_argument(
        "--task-prefixes",
        nargs="*",
        default=None,
        help="Optional prefixes to limit processed task names.",
    )
    return parser.parse_args()


def build_data_cfg(data_root: Path) -> dict:
    return {
        "data_root_dir": str(data_root),
        "data_mix": "interna1_split_aloha",
        "video_backend": "pyav",
        "sequential_step_sampling": False,
        "prompt_prefix_fps": True,
        "delete_pause_frame": DELETE_PAUSE_FRAME,
    }


def build_effective_data_cfg(data_root: Path, robot_type: str) -> dict:
    requested_cfg = build_data_cfg(data_root)
    return _merge_robot_type_data_cfg_defaults(ROBOT_TYPE_CONFIG_MAP[robot_type], requested_cfg) or requested_cfg


def expected_paths(task: str, data_root: Path, robot_type: str) -> tuple[Path, Path]:
    dataset_meta_dir = data_root / task / "meta"
    data_cfg = build_effective_data_cfg(data_root, robot_type)
    use_variant_cache = bool(data_cfg.get("cache_variant"))
    if use_variant_cache:
        cache_key = build_dataset_cache_key(
            dataset_name=task,
            filter_outlier_trajectory=bool(data_cfg.get("filter_outlier_trajectory", False)),
            outlier_abs_limit=float(data_cfg.get("outlier_abs_limit", 3.1415926)),
            filter_gripper_outlier_trajectory=bool(data_cfg.get("filter_gripper_outlier_trajectory", False)),
            gripper_outlier_abs_limit=(
                float(data_cfg["gripper_outlier_abs_limit"])
                if data_cfg.get("gripper_outlier_abs_limit", None) is not None
                else None
            ),
            delete_pause_frame=bool(data_cfg.get("delete_pause_frame", DELETE_PAUSE_FRAME)),
            data_cfg=data_cfg,
        )
        steps_path = dataset_meta_dir / f"steps_data_index_filtered_{cache_key}.pkl"
        stats_path = dataset_meta_dir / f"stats_gr00t_filtered_{cache_key}.json"
    else:
        steps_path = dataset_meta_dir / "steps_data_index.pkl"
        stats_path = dataset_meta_dir / "stats_gr00t.json"
    return steps_path, stats_path


def has_complete_cache(task: str, data_root: Path, robot_type: str) -> bool:
    steps_path, stats_path = expected_paths(task, data_root, robot_type)
    if not steps_path.exists() or not stats_path.exists():
        return False
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception:
        return False
    return all(mode in stats for mode in ("abs", "delta", "rel"))


def list_tasks(task_list: Path, task_prefixes: list[str] | None) -> list[str]:
    with open(task_list, "r", encoding="utf-8") as f:
        tasks = [line.strip() for line in f if line.strip()]
    if task_prefixes:
        tasks = [task for task in tasks if any(task.startswith(prefix) for prefix in task_prefixes)]
    return tasks


def compute_one(task: str) -> str:
    steps_path, stats_path = expected_paths(task, DATA_ROOT, ROBOT_TYPE)
    if SKIP_EXISTING and has_complete_cache(task, DATA_ROOT, ROBOT_TYPE):
        return f"[SKIP] {task}"

    try:
        dataset = make_LeRobotSingleDataset(
            data_root_dir=DATA_ROOT,
            data_name=task,
            robot_type=ROBOT_TYPE,
            delete_pause_frame=DELETE_PAUSE_FRAME,
            data_cfg=build_data_cfg(DATA_ROOT),
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
    return f"[DONE] {task}"


def main() -> None:
    global DATA_ROOT
    global ROBOT_TYPE
    global SKIP_EXISTING

    args = parse_args()
    DATA_ROOT = args.data_root.resolve()
    ROBOT_TYPE = args.robot_type
    SKIP_EXISTING = args.skip_existing
    task_list = args.task_list.resolve()
    if task_list == TASK_LIST.resolve() and not task_list.exists():
        task_list = DATA_ROOT / "split_aloha_tasks.txt"

    tasks = list_tasks(task_list, args.task_prefixes)
    print(f"Total tasks: {len(tasks)}")
    print(f"Using {args.num_workers} worker processes across tasks")
    print(f"Robot type: {ROBOT_TYPE}")
    print(f"Task list: {task_list}")

    with Pool(args.num_workers) as pool:
        for idx, msg in enumerate(pool.imap_unordered(compute_one, tasks), start=1):
            print(f"[{idx}/{len(tasks)}] {msg}")
    print("All done.")


if __name__ == "__main__":
    main()
