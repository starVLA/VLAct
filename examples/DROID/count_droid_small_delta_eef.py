#!/usr/bin/env python3
"""Count DROID trajectories whose action deltas stay small under simple rules.

This script measures smoothness of the raw DROID action stream. The layout used
in this repository is:

    action = [x, y, z, roll, pitch, yaw, gripper]

We define:

    delta_action[t] = action[t + 1] - action[t]

Rules:
1. rotation rule:
   roll/pitch/yaw action delta is wrapped into [-pi, pi] and checked against
   [-rotation_threshold, rotation_threshold]
2. position rule:
   x/y/z action delta is checked against [-position_threshold, position_threshold]
3. both rule:
   satisfy rotation rule and position rule at the same time
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

DEFAULT_ROOT = Path("/project/vonneumann1/wcy/copy/starVLA-VLAct/playground/Datasets/DROID")
DEFAULT_OUTPUT = Path(
    "/project/vonneumann1/wcy/copy/starVLA-VLAct/examples/DROID/out/droid_small_delta_eef_summary.json"
)
ACTION_KEY = "action"
POS_DIMS = (0, 1, 2)
ROT_DIMS = (3, 4, 5)
DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
DEFAULT_POSITION_THRESHOLD = 0.02
DEFAULT_ROTATION_THRESHOLD = 0.05
DEFAULT_NUM_WORKERS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--position-threshold", type=float, default=DEFAULT_POSITION_THRESHOLD)
    parser.add_argument("--rotation-threshold", type=float, default=DEFAULT_ROTATION_THRESHOLD)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + math.pi) % (2.0 * math.pi) - math.pi


def list_parquet_paths(root: Path) -> list[Path]:
    data_dir = root / "data"
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.rglob("*.parquet"))


def process_parquet(task: tuple[str, float, float]) -> dict:
    parquet_path_str, position_threshold, rotation_threshold = task
    parquet_path = Path(parquet_path_str)

    try:
        df = pd.read_parquet(parquet_path, columns=[ACTION_KEY])
    except Exception as exc:
        return {"parquet": str(parquet_path), "error": f"read_failed: {exc}"}

    if len(df) < 2:
        return {
            "parquet": str(parquet_path),
            "num_steps": 0,
            "delta_min": [None] * len(DIM_NAMES),
            "delta_max": [None] * len(DIM_NAMES),
            "rotation_ok_steps": 0,
            "position_ok_steps": 0,
            "both_ok_steps": 0,
            "rotation_ok_trajectory": False,
            "position_ok_trajectory": False,
            "both_ok_trajectory": False,
        }

    action = np.vstack(df[ACTION_KEY].to_numpy()).astype(np.float32)
    if action.shape[1] < 7:
        return {
            "parquet": str(parquet_path),
            "error": f"unexpected_dims: action={action.shape}",
        }

    delta = action[1:] - action[:-1]
    delta[:, ROT_DIMS] = wrap_to_pi(delta[:, ROT_DIMS])

    rot_ok = np.all(np.abs(delta[:, ROT_DIMS]) <= rotation_threshold, axis=1)
    pos_ok = np.all(np.abs(delta[:, POS_DIMS]) <= position_threshold, axis=1)
    both_ok = rot_ok & pos_ok

    return {
        "parquet": str(parquet_path),
        "num_steps": int(delta.shape[0]),
        "delta_min": delta.min(axis=0).tolist(),
        "delta_max": delta.max(axis=0).tolist(),
        "rotation_ok_steps": int(rot_ok.sum()),
        "position_ok_steps": int(pos_ok.sum()),
        "both_ok_steps": int(both_ok.sum()),
        "rotation_ok_trajectory": bool(np.all(rot_ok)),
        "position_ok_trajectory": bool(np.all(pos_ok)),
        "both_ok_trajectory": bool(np.all(both_ok)),
    }


def reduce_results(results: list[dict]) -> dict:
    ok_results = [item for item in results if "error" not in item]
    errors = [item for item in results if "error" in item]

    total_trajectories = len(ok_results)
    total_steps = sum(item["num_steps"] for item in ok_results)
    global_min = np.full(len(DIM_NAMES), np.inf, dtype=np.float64)
    global_max = np.full(len(DIM_NAMES), -np.inf, dtype=np.float64)

    for item in ok_results:
        if item["num_steps"] == 0:
            continue
        global_min = np.minimum(global_min, np.asarray(item["delta_min"], dtype=np.float64))
        global_max = np.maximum(global_max, np.asarray(item["delta_max"], dtype=np.float64))

    if total_steps == 0:
        min_list = [None] * len(DIM_NAMES)
        max_list = [None] * len(DIM_NAMES)
    else:
        min_list = global_min.tolist()
        max_list = global_max.tolist()

    summary = {
        "action_key": ACTION_KEY,
        "delta_formula": "action[t + 1] - action[t]",
        "total_trajectories": total_trajectories,
        "total_steps": total_steps,
        "delta_min": min_list,
        "delta_max": max_list,
        "errors": errors,
    }

    for rule_name in ("rotation", "position", "both"):
        matched_steps = sum(item[f"{rule_name}_ok_steps"] for item in ok_results)
        matched_trajectories = sum(int(item[f"{rule_name}_ok_trajectory"]) for item in ok_results)
        summary[f"{rule_name}_matched_steps"] = matched_steps
        summary[f"{rule_name}_step_ratio"] = matched_steps / total_steps if total_steps else 0.0
        summary[f"{rule_name}_matched_trajectories"] = matched_trajectories
        summary[f"{rule_name}_trajectory_ratio"] = (
            matched_trajectories / total_trajectories if total_trajectories else 0.0
        )

    return summary


def main() -> None:
    args = parse_args()
    parquet_paths = list_parquet_paths(args.root)
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {args.root / 'data'}")

    tasks = [
        (str(path), float(args.position_threshold), float(args.rotation_threshold))
        for path in parquet_paths
    ]

    if args.num_workers <= 1:
        iterator = map(process_parquet, tasks)
        results = list(tqdm(iterator, total=len(tasks), desc="Processing DROID") if tqdm else iterator)
    else:
        with Pool(args.num_workers) as pool:
            iterator = pool.imap_unordered(process_parquet, tasks)
            results = list(tqdm(iterator, total=len(tasks), desc="Processing DROID") if tqdm else iterator)

    summary = reduce_results(results)
    summary["root"] = str(args.root)
    summary["position_threshold"] = float(args.position_threshold)
    summary["rotation_threshold"] = float(args.rotation_threshold)
    summary["num_parquet"] = len(parquet_paths)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to: {args.output}")
    print(f"total_trajectories: {summary['total_trajectories']}")
    print(f"total_steps: {summary['total_steps']}")
    print(
        "position / rotation / both trajectory ratio:",
        f"{summary['position_trajectory_ratio']:.6f}",
        f"{summary['rotation_trajectory_ratio']:.6f}",
        f"{summary['both_trajectory_ratio']:.6f}",
    )
    print(
        "position / rotation / both step ratio:",
        f"{summary['position_step_ratio']:.6f}",
        f"{summary['rotation_step_ratio']:.6f}",
        f"{summary['both_step_ratio']:.6f}",
    )


if __name__ == "__main__":
    main()
