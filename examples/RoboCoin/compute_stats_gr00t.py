#!/usr/bin/env python3
"""Batch-generate training caches for RoboCOIN datasets.

This script matches the current RoboCOIN training-time cache path, including
robot-type defaults such as joint/gripper outlier filtering and cache variants
like `ROBOCOIN.AgileX_flip_wrap`.

For each RoboCOIN task it generates:
- `meta/steps_data_index_filtered_<hash>.pkl`
- `meta/stats_gr00t_filtered_<hash>.json`

Parallelism model:
- one worker process handles one dataset/task at a time
- multiple RoboCOIN tasks are processed in parallel
- we do NOT spawn multiple processes inside a single task
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.datasets import build_dataset_cache_key
from starVLA.dataloader.lerobot_datasets import (
    _merge_robot_type_data_cfg_defaults,
    make_LeRobotSingleDataset,
)

ROBOT_TYPE = "ROBOCOIN.AgileX"
DATA_ROOT = Path("/project/vonneumann1/wcy/copy/starVLA-VLAct/playground/Datasets/RoboCOIN")
NUM_WORKERS = 8
SKIP_EXISTING = False
OUTLIER_ABS_LIMIT = 6.2831852
DELETE_PAUSE_FRAME = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--robot-type",
        type=str,
        default=ROBOT_TYPE,
        choices=[
            "ROBOCOIN.AgileX",
            "ROBOCOIN.AgileX_wrap",
            "ROBOCOIN.AgileX_flip_wrap",
        ],
        help="Robot config to use when generating caches.",
    )
    parser.add_argument("--outlier-abs-limit", type=float, default=OUTLIER_ABS_LIMIT)
    parser.add_argument(
        "--prefixes",
        nargs="*",
        default=["Cobot_Magic_", "Split_aloha_"],
        help="Only process dataset dirs with these prefixes.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets whose stats_gr00t.json already contains abs/delta/rel.",
    )
    return parser.parse_args()


def build_data_cfg(data_root: Path, outlier_abs_limit: float) -> dict:
    return {
        "data_root_dir": str(data_root.parent),
        "data_mix": "robocoin",
        "action_type": "abs_qpos",
        "video_backend": "pyav",
        "outlier_abs_limit": outlier_abs_limit,
        "prompt_prefix_fps": True,
        "delete_pause_frame": DELETE_PAUSE_FRAME,
    }


def build_effective_data_cfg(data_root: Path, outlier_abs_limit: float, robot_type: str) -> dict:
    requested_cfg = build_data_cfg(data_root, outlier_abs_limit)
    return _merge_robot_type_data_cfg_defaults(ROBOT_TYPE_CONFIG_MAP[robot_type], requested_cfg) or requested_cfg


def build_trajectory_filter_cache_key(task: str, data_cfg: dict) -> str:
    return build_dataset_cache_key(
        dataset_name=task,
        filter_outlier_trajectory=bool(data_cfg.get("filter_outlier_trajectory", False)),
        outlier_abs_limit=float(data_cfg.get("outlier_abs_limit", OUTLIER_ABS_LIMIT)),
        filter_gripper_outlier_trajectory=bool(data_cfg.get("filter_gripper_outlier_trajectory", False)),
        gripper_outlier_abs_limit=float(
            data_cfg.get("gripper_outlier_abs_limit", data_cfg.get("outlier_abs_limit", OUTLIER_ABS_LIMIT))
        ),
        delete_pause_frame=bool(data_cfg.get("delete_pause_frame", DELETE_PAUSE_FRAME)),
        data_cfg=data_cfg,
    )


def expected_paths(task: str, outlier_abs_limit: float, robot_type: str) -> tuple[Path, Path]:
    dataset_dir = DATA_ROOT / task / "meta"
    data_cfg = build_effective_data_cfg(DATA_ROOT, outlier_abs_limit, robot_type)
    cache_key = build_trajectory_filter_cache_key(task, data_cfg)
    use_variant_cache = bool(data_cfg.get("cache_variant"))
    if bool(data_cfg.get("filter_outlier_trajectory", False)) or use_variant_cache:
        steps_path = dataset_dir / f"steps_data_index_filtered_{cache_key}.pkl"
        stats_path = dataset_dir / f"stats_gr00t_filtered_{cache_key}.json"
    else:
        steps_path = dataset_dir / "steps_data_index.pkl"
        stats_path = dataset_dir / "stats_gr00t.json"
    return steps_path, stats_path


def has_complete_filtered_cache(task: str, outlier_abs_limit: float, robot_type: str) -> bool:
    steps_path, stats_path = expected_paths(task, outlier_abs_limit, robot_type)
    if not steps_path.exists() or not stats_path.exists():
        return False
    try:
        with open(stats_path) as f:
            stats = json.load(f)
    except Exception:
        return False
    return all(mode in stats for mode in ("abs", "delta", "rel"))


def list_tasks(data_root: Path, prefixes: list[str]) -> list[str]:
    tasks = []
    for path in sorted(data_root.iterdir()):
        if not path.is_dir():
            continue
        if prefixes and not any(path.name.startswith(prefix) for prefix in prefixes):
            continue
        if (path / "meta" / "info.json").exists() and (path / "data").exists():
            tasks.append(path.name)
    return tasks


def compute_one(task: str) -> str:
    steps_path, stats_path = expected_paths(task, OUTLIER_ABS_LIMIT, ROBOT_TYPE)
    if SKIP_EXISTING and has_complete_filtered_cache(task, OUTLIER_ABS_LIMIT, ROBOT_TYPE):
        return f"[SKIP] {task}"
    try:
        data_cfg = build_data_cfg(DATA_ROOT, OUTLIER_ABS_LIMIT)
        dataset = make_LeRobotSingleDataset(
            data_root_dir=DATA_ROOT,
            data_name=task,
            robot_type=ROBOT_TYPE,
            delete_pause_frame=DELETE_PAUSE_FRAME,
            data_cfg=data_cfg,
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


if __name__ == "__main__":
    args = parse_args()
    DATA_ROOT = args.data_root
    SKIP_EXISTING = args.skip_existing
    OUTLIER_ABS_LIMIT = args.outlier_abs_limit
    ROBOT_TYPE = args.robot_type
    tasks = list_tasks(DATA_ROOT, args.prefixes)
    print(f"Total tasks: {len(tasks)}")
    print(f"Using {args.num_workers} worker processes across tasks")
    print(f"Robot type: {ROBOT_TYPE}")
    with Pool(args.num_workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(compute_one, tasks), start=1):
            print(f"[{i}/{len(tasks)}] {msg}")
    print("All done.")
