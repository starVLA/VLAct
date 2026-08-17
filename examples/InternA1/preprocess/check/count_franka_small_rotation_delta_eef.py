#!/usr/bin/env python3
"""Count Franka trajectories whose delta EEF stays small under simple rules.

Scan all Franka datasets under a sim_updated root, compute:

    delta_eef = actions.gripper.pose - states.gripper.pose

Rules:
1. rotation rule:
   roll/pitch/yaw delta is wrapped into [-pi, pi] and checked against [-0.05, 0.05]
2. position rule:
   x/y/z delta is checked against [-0.01, 0.01]
3. both rule:
   satisfy rotation rule and position rule at the same time

The script writes a compact JSON summary and also prints a short text summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

DEFAULT_ROOT = Path(
    "/project/vonneumann1/wcy/copy/starVLA-VLAct/playground/Datasets/InternData-A1/sim_updated"
)
DEFAULT_OUTPUT = Path(
    "/project/vonneumann1/wcy/copy/starVLA-VLAct/examples/InternA1/preprocess/check/out/"
    "franka_small_rotation_delta_eef_summary.json"
)
ACTION_KEY = "actions.gripper.pose"
STATE_KEY = "states.gripper.pose"
ACTION_GRIPPER_KEY = "actions.gripper.position"
STATE_GRIPPER_KEY = "states.gripper.position"
ROT_DIMS = (3, 4, 5)
POS_DIMS = (0, 1, 2)
DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
DEFAULT_ROTATION_THRESHOLD = 0.05
DEFAULT_POSITION_THRESHOLD = 0.01
DEFAULT_NUM_WORKERS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rotation-threshold", type=float, default=DEFAULT_ROTATION_THRESHOLD)
    parser.add_argument("--position-threshold", type=float, default=DEFAULT_POSITION_THRESHOLD)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + math.pi) % (2.0 * math.pi) - math.pi


def discover_franka_dataset_dirs(root: Path) -> list[Path]:
    dataset_dirs: list[Path] = []
    seen: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in {"data", "videos"}]
        if current.name != "meta" or "info.json" not in filenames:
            continue
        info_path = current / "info.json"
        if "/franka/" not in info_path.as_posix():
            continue
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            continue
        features = info.get("features", {})
        if (
            ACTION_KEY not in features
            or STATE_KEY not in features
            or ACTION_GRIPPER_KEY not in features
            or STATE_GRIPPER_KEY not in features
        ):
            continue
        dataset_dir = current.parent
        if dataset_dir not in seen:
            dataset_dirs.append(dataset_dir)
            seen.add(dataset_dir)
    return sorted(dataset_dirs)


def list_parquet_paths(dataset_dir: Path) -> list[Path]:
    data_dir = dataset_dir / "data"
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.rglob("*.parquet"))


def process_dataset(task: tuple[str, str, float, float]) -> dict:
    dataset_dir_str, root_str, rotation_threshold, position_threshold = task
    dataset_dir = Path(dataset_dir_str)
    root = Path(root_str)
    parquet_paths = list_parquet_paths(dataset_dir)

    total_steps = 0
    total_trajectories = 0
    delta_min = np.full(len(DIM_NAMES), np.inf, dtype=np.float64)
    delta_max = np.full(len(DIM_NAMES), -np.inf, dtype=np.float64)
    rule_stats = {
        "rotation": {
            "matched_steps": 0,
            "matched_trajectories": 0,
            "good_traj_steps": 0,
            "good_traj_delta_min": np.full(len(DIM_NAMES), np.inf, dtype=np.float64),
            "good_traj_delta_max": np.full(len(DIM_NAMES), -np.inf, dtype=np.float64),
        },
        "position": {
            "matched_steps": 0,
            "matched_trajectories": 0,
            "good_traj_steps": 0,
            "good_traj_delta_min": np.full(len(DIM_NAMES), np.inf, dtype=np.float64),
            "good_traj_delta_max": np.full(len(DIM_NAMES), -np.inf, dtype=np.float64),
        },
        "both": {
            "matched_steps": 0,
            "matched_trajectories": 0,
            "good_traj_steps": 0,
            "good_traj_delta_min": np.full(len(DIM_NAMES), np.inf, dtype=np.float64),
            "good_traj_delta_max": np.full(len(DIM_NAMES), -np.inf, dtype=np.float64),
        },
    }

    for parquet_path in parquet_paths:
        try:
            df = pd.read_parquet(
                parquet_path,
                columns=[ACTION_KEY, STATE_KEY, ACTION_GRIPPER_KEY, STATE_GRIPPER_KEY],
            )
        except Exception as exc:
            return {
                "dataset_dir": str(dataset_dir.relative_to(root)),
                "error": f"read_failed: {exc}",
            }

        if df.empty:
            continue

        action_pose = np.vstack(df[ACTION_KEY].to_numpy()).astype(np.float32)
        state_pose = np.vstack(df[STATE_KEY].to_numpy()).astype(np.float32)
        delta_pose = action_pose - state_pose
        delta_pose[:, ROT_DIMS] = wrap_to_pi(delta_pose[:, ROT_DIMS])
        action_gripper = df[ACTION_GRIPPER_KEY].to_numpy(dtype=np.float32).reshape(-1, 1)
        state_gripper = df[STATE_GRIPPER_KEY].to_numpy(dtype=np.float32).reshape(-1, 1)
        delta_gripper = action_gripper - state_gripper
        delta = np.concatenate([delta_pose, delta_gripper], axis=1)

        rot_ok = np.all(np.abs(delta_pose[:, ROT_DIMS]) <= rotation_threshold, axis=1)
        pos_ok = np.all(np.abs(delta_pose[:, POS_DIMS]) <= position_threshold, axis=1)
        both_ok = rot_ok & pos_ok

        total_trajectories += 1
        total_steps += int(delta.shape[0])
        delta_min = np.minimum(delta_min, delta.min(axis=0))
        delta_max = np.maximum(delta_max, delta.max(axis=0))

        for rule_name, step_ok in (
            ("rotation", rot_ok),
            ("position", pos_ok),
            ("both", both_ok),
        ):
            traj_ok = bool(np.all(step_ok))
            rule_stats[rule_name]["matched_steps"] += int(step_ok.sum())
            rule_stats[rule_name]["matched_trajectories"] += int(traj_ok)
            if traj_ok:
                rule_stats[rule_name]["good_traj_steps"] += int(delta.shape[0])
                rule_stats[rule_name]["good_traj_delta_min"] = np.minimum(
                    rule_stats[rule_name]["good_traj_delta_min"], delta.min(axis=0)
                )
                rule_stats[rule_name]["good_traj_delta_max"] = np.maximum(
                    rule_stats[rule_name]["good_traj_delta_max"], delta.max(axis=0)
                )

    output = {
        "dataset_dir": str(dataset_dir.relative_to(root)),
        "num_parquet": len(parquet_paths),
        "total_trajectories": total_trajectories,
        "total_steps": total_steps,
        "delta_min": delta_min.tolist() if total_trajectories else [None] * len(DIM_NAMES),
        "delta_max": delta_max.tolist() if total_trajectories else [None] * len(DIM_NAMES),
    }
    for rule_name, stats in rule_stats.items():
        matched_trajectories = stats["matched_trajectories"]
        output[f"{rule_name}_matched_trajectories"] = matched_trajectories
        output[f"{rule_name}_trajectory_ratio"] = (
            matched_trajectories / total_trajectories if total_trajectories else 0.0
        )
        output[f"{rule_name}_matched_steps"] = stats["matched_steps"]
        output[f"{rule_name}_step_ratio"] = (
            stats["matched_steps"] / total_steps if total_steps else 0.0
        )
        output[f"{rule_name}_good_traj_steps"] = stats["good_traj_steps"]
        output[f"{rule_name}_good_traj_delta_min"] = (
            stats["good_traj_delta_min"].tolist()
            if matched_trajectories
            else [None] * len(DIM_NAMES)
        )
        output[f"{rule_name}_good_traj_delta_max"] = (
            stats["good_traj_delta_max"].tolist()
            if matched_trajectories
            else [None] * len(DIM_NAMES)
        )
    return output


def reduce_results(results: list[dict]) -> dict:
    ok_results = [item for item in results if "error" not in item]
    errors = [item for item in results if "error" in item]

    total_trajectories = sum(item["total_trajectories"] for item in ok_results)
    total_steps = sum(item["total_steps"] for item in ok_results)

    global_min = np.full(len(DIM_NAMES), np.inf, dtype=np.float64)
    global_max = np.full(len(DIM_NAMES), -np.inf, dtype=np.float64)
    rules = {}
    for item in ok_results:
        if item["total_trajectories"] == 0:
            continue
        global_min = np.minimum(global_min, np.asarray(item["delta_min"], dtype=np.float64))
        global_max = np.maximum(global_max, np.asarray(item["delta_max"], dtype=np.float64))

    if total_trajectories == 0:
        min_list = [None] * len(DIM_NAMES)
        max_list = [None] * len(DIM_NAMES)
    else:
        min_list = global_min.tolist()
        max_list = global_max.tolist()

    for rule_name in ("rotation", "position", "both"):
        matched_trajectories = sum(
            item[f"{rule_name}_matched_trajectories"] for item in ok_results
        )
        matched_steps = sum(item[f"{rule_name}_matched_steps"] for item in ok_results)
        good_traj_steps = sum(item[f"{rule_name}_good_traj_steps"] for item in ok_results)
        good_traj_global_min = np.full(len(DIM_NAMES), np.inf, dtype=np.float64)
        good_traj_global_max = np.full(len(DIM_NAMES), -np.inf, dtype=np.float64)
        for item in ok_results:
            if item[f"{rule_name}_matched_trajectories"] == 0:
                continue
            good_traj_global_min = np.minimum(
                good_traj_global_min,
                np.asarray(item[f"{rule_name}_good_traj_delta_min"], dtype=np.float64),
            )
            good_traj_global_max = np.maximum(
                good_traj_global_max,
                np.asarray(item[f"{rule_name}_good_traj_delta_max"], dtype=np.float64),
            )
        rules[rule_name] = {
            "matched_trajectories": matched_trajectories,
            "trajectory_ratio": (
                matched_trajectories / total_trajectories if total_trajectories else 0.0
            ),
            "matched_steps": matched_steps,
            "step_ratio": matched_steps / total_steps if total_steps else 0.0,
            "good_traj_steps": good_traj_steps,
            "good_traj_delta_min": (
                good_traj_global_min.tolist()
                if matched_trajectories
                else [None] * len(DIM_NAMES)
            ),
            "good_traj_delta_max": (
                good_traj_global_max.tolist()
                if matched_trajectories
                else [None] * len(DIM_NAMES)
            ),
        }

    return {
        "num_dataset_dirs": len(results),
        "num_ok_dataset_dirs": len(ok_results),
        "num_error_dataset_dirs": len(errors),
        "total_trajectories": total_trajectories,
        "total_steps": total_steps,
        "delta_min": min_list,
        "delta_max": max_list,
        "rules": rules,
        "position_dims": {
            "indices": list(POS_DIMS),
            "names": [DIM_NAMES[i] for i in POS_DIMS],
        },
        "rotation_dims": {
            "indices": list(ROT_DIMS),
            "names": [DIM_NAMES[i] for i in ROT_DIMS],
        },
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    if args.rotation_threshold <= 0:
        raise ValueError("--rotation-threshold must be positive")
    if args.position_threshold <= 0:
        raise ValueError("--position-threshold must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")

    dataset_dirs = discover_franka_dataset_dirs(args.root)
    print(f"found franka dataset dirs: {len(dataset_dirs)}", flush=True)
    if not dataset_dirs:
        raise RuntimeError("No Franka dataset directories found.")

    tasks = [
        (
            str(path),
            str(args.root),
            float(args.rotation_threshold),
            float(args.position_threshold),
        )
        for path in dataset_dirs
    ]
    results: list[dict] = []
    with Pool(args.num_workers) as pool:
        iterator = pool.imap_unordered(process_dataset, tasks)
        progress = (
            tqdm(iterator, total=len(tasks), desc="datasets", dynamic_ncols=True)
            if tqdm is not None
            else iterator
        )
        for idx, item in enumerate(progress, start=1):
            results.append(item)
            label = item["dataset_dir"]
            if tqdm is not None:
                if "error" in item:
                    tqdm.write(f"ERROR {label}: {item['error']}")
                else:
                    progress.set_postfix_str(
                        f"{label} both traj {item['both_matched_trajectories']}/"
                        f"{item['total_trajectories']}"
                    )
            else:
                if "error" in item:
                    print(f"[{idx}/{len(tasks)}] ERROR {label}: {item['error']}", flush=True)
                else:
                    print(
                        f"[{idx}/{len(tasks)}] {label} | "
                        f"rot traj {item['rotation_matched_trajectories']}/{item['total_trajectories']} | "
                        f"pos traj {item['position_matched_trajectories']}/{item['total_trajectories']} | "
                        f"both traj {item['both_matched_trajectories']}/{item['total_trajectories']}",
                        flush=True,
                    )

    results.sort(key=lambda x: x["dataset_dir"])
    summary = reduce_results(results)
    payload = {
        "root": str(args.root),
        "action_key": ACTION_KEY,
        "state_key": STATE_KEY,
        "action_gripper_key": ACTION_GRIPPER_KEY,
        "state_gripper_key": STATE_GRIPPER_KEY,
        "delta_formula": f"{ACTION_KEY} - {STATE_KEY}",
        "gripper_delta_formula": f"{ACTION_GRIPPER_KEY} - {STATE_GRIPPER_KEY}",
        "rotation_wrap": "[-pi, pi]",
        "rotation_threshold": args.rotation_threshold,
        "position_threshold": args.position_threshold,
        "dimensions": DIM_NAMES,
        "summary": summary,
        "datasets": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("", flush=True)
    print(f"saved: {args.output}", flush=True)
    for rule_name in ("rotation", "position", "both"):
        rule = summary["rules"][rule_name]
        print(
            f"{rule_name} trajectory match: "
            f"{rule['matched_trajectories']}/{summary['total_trajectories']} "
            f"({rule['trajectory_ratio']:.4f})",
            flush=True,
        )
        print(
            f"{rule_name} step match: "
            f"{rule['matched_steps']}/{summary['total_steps']} "
            f"({rule['step_ratio']:.4f})",
            flush=True,
        )
        print(f"{rule_name} good trajectory delta min: {rule['good_traj_delta_min']}", flush=True)
        print(f"{rule_name} good trajectory delta max: {rule['good_traj_delta_max']}", flush=True)


if __name__ == "__main__":
    main()
