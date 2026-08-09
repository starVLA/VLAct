#!/usr/bin/env python3
"""Precompute unified_norm statistics for InternData-A1 split_aloha mixtures.

This script mirrors the runtime unified_norm logic in
`starVLA/dataloader/gr00t_lerobot/datasets.py`:
- supports both `joint` and `eef` coordinate systems
- supports `abs`, `delta`, and `rel` action modes
- applies FPS scaling only for delta actions
- writes cache files compatible with unified_norm cache lookup
"""

import argparse
import glob
import hashlib
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = "/project/vonneumann1/datasets/InternData-A1"
DEFAULT_TASK_LIST = (
    "/project/vonneumann1/wcy/code/starVLA-dev/playground/Datasets/InternData-A1/"
    "split_aloha_tasks.txt"
)
DEFAULT_NUM_WORKERS = 16
STAT_NAMES = ["mean", "std", "min", "max", "q01", "q99"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--task-list", default=DEFAULT_TASK_LIST)
    parser.add_argument(
        "--coordinate-system",
        choices=["joint", "eef"],
        default="joint",
        help="Which state/action representation to pool.",
    )
    parser.add_argument(
        "--eef-frame",
        choices=["armbase", "robot"],
        default="armbase",
        help="EEF reference frame when --coordinate-system=eef.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["abs", "delta", "rel"],
        default="abs",
        help="Action representation after conversion, matching unified_norm.",
    )
    parser.add_argument(
        "--action-type",
        default=None,
        help="Optional cache-key hint. Defaults to a value inferred from coordinate-system and action-mode.",
    )
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def build_modality(first_dir: Path, coordinate_system: str, eef_frame: str) -> dict:
    modality = load_json(first_dir / "meta" / "modality.json")
    if coordinate_system == "joint":
        return modality

    if coordinate_system != "eef":
        raise ValueError(f"Unsupported coordinate system: {coordinate_system}")

    if eef_frame == "armbase":
        left_state_key = "states.left_ee_to_left_armbase_pose"
        right_state_key = "states.right_ee_to_right_armbase_pose"
        left_action_key = "actions.left_ee_to_left_armbase_pose"
        right_action_key = "actions.right_ee_to_right_armbase_pose"
        left_logical_key = "left_armbase_eef_pose"
        right_logical_key = "right_armbase_eef_pose"
    elif eef_frame == "robot":
        left_state_key = "states.left_ee_to_robot_pose"
        right_state_key = "states.right_ee_to_robot_pose"
        left_action_key = "actions.left_ee_to_robot_pose"
        right_action_key = "actions.right_ee_to_robot_pose"
        left_logical_key = "left_robot_eef_pose"
        right_logical_key = "right_robot_eef_pose"
    else:
        raise ValueError(f"Unsupported eef frame: {eef_frame}")

    eef_modality = {
        "state": {
            left_logical_key: {
                "start": 0,
                "end": 7,
                "original_key": left_state_key,
            },
            right_logical_key: {
                "start": 0,
                "end": 7,
                "original_key": right_state_key,
            },
            "left_gripper": modality["state"]["left_gripper"],
            "right_gripper": modality["state"]["right_gripper"],
        },
        "action": {
            left_logical_key: {
                "start": 0,
                "end": 7,
                "original_key": left_action_key,
            },
            right_logical_key: {
                "start": 0,
                "end": 7,
                "original_key": right_action_key,
            },
            "left_gripper": modality["action"]["left_gripper"],
            "right_gripper": modality["action"]["right_gripper"],
        },
        "video": modality["video"],
        "annotation": modality.get("annotation", {}),
    }
    return eef_modality


def infer_state_subkey(action_subkey: str) -> str:
    return action_subkey


def infer_action_type(coordinate_system: str, action_mode: str, eef_frame: str) -> str:
    if coordinate_system == "joint":
        mapping = {
            "abs": "abs_qpos",
            "delta": "delta_qpos",
            "rel": "rel_qpos",
        }
    else:
        if eef_frame == "armbase":
            mapping = {
                "abs": "abs_eef_arm",
                "delta": "delta_eef_arm",
                "rel": "rel_eef_arm",
            }
        elif eef_frame == "robot":
            mapping = {
                "abs": "abs_eef_robot",
                "delta": "delta_eef_robot",
                "rel": "rel_eef_robot",
            }
        else:
            raise ValueError(f"Unsupported eef frame: {eef_frame}")
    return mapping[action_mode]


def apply_action_mode(
    action_values: np.ndarray, state_values: np.ndarray, action_mode: str, is_rotation: bool = False,
) -> np.ndarray:
    if action_mode == "abs":
        return action_values

    if action_values.ndim != 2 or state_values.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays for action/state, got {action_values.shape} and {state_values.shape}"
        )
    if action_values.shape[1] != state_values.shape[1]:
        raise ValueError(
            f"Action/state dim mismatch: {action_values.shape} vs {state_values.shape}"
        )

    state0 = state_values[0]
    if action_mode == "delta":
        output = action_values.copy()
        if len(output) > 1:
            output[1:] = action_values[1:] - action_values[:-1]
        output[0] = action_values[0] - state0
        if is_rotation:
            output = (output + np.pi) % (2 * np.pi) - np.pi
        return output
    if action_mode == "rel":
        return action_values - state0
    raise ValueError(f"Unsupported action mode: {action_mode}")


def load_task(args):
    """Read parquet files for one task and extract per-key arrays."""
    task, data_root, modality, action_mode = args
    dataset_dir = Path(data_root) / task
    pqs = sorted(glob.glob(str(dataset_dir / "data" / "*" / "*.parquet")))
    if not pqs:
        return task, {}

    info = load_json(dataset_dir / "meta" / "info.json")
    fps = info.get("fps", 1)
    dfs = [pd.read_parquet(p, use_threads=False) for p in pqs]
    df = pd.concat(dfs)
    del dfs

    state_arrays = {}
    for sk, meta in modality["state"].items():
        col = meta["original_key"]
        if col not in df.columns:
            continue
        arr = np.vstack([np.asarray(x, dtype=np.float32) for x in df[col]])
        state_arrays[sk] = arr[:, meta["start"] : meta["end"]]

    result = {}
    for sk, arr in state_arrays.items():
        result[("state", sk)] = arr

    for sk, meta in modality["action"].items():
        col = meta["original_key"]
        if col not in df.columns:
            continue
        action_arr = np.vstack([np.asarray(x, dtype=np.float32) for x in df[col]])
        action_arr = action_arr[:, meta["start"] : meta["end"]]
        if action_mode != "abs":
            state_sk = infer_state_subkey(sk)
            if state_sk not in state_arrays:
                raise ValueError(f"Missing state key for action.{sk}")
            action_arr = apply_action_mode(
                action_arr, state_arrays[state_sk], action_mode, is_rotation="rotation" in sk,
            )
        if action_mode == "delta" and "gripper" not in sk:
            action_arr = action_arr * fps
        result[("action", sk)] = action_arr
    return task, result


def compute_stats(array: np.ndarray) -> dict:
    return {
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def main():
    args = parse_args()
    action_type = args.action_type or infer_action_type(
        args.coordinate_system, args.action_mode, args.eef_frame
    )
    with open(args.task_list, "r") as f:
        tasks = [line.strip() for line in f if line.strip()]
    print(f"Total tasks: {len(tasks)}")

    first_dir = Path(args.data_root) / tasks[0]
    modality = build_modality(first_dir, args.coordinate_system, args.eef_frame)
    first_info = load_json(first_dir / "meta" / "info.json")
    print(
        f"coordinate_system={args.coordinate_system}, eef_frame={args.eef_frame}, "
        f"action_mode={args.action_mode}, "
        f"action_type={action_type}, "
        f"fps={first_info.get('fps', 1)}"
    )
    print(
        f"action keys: {list(modality['action'].keys())}, "
        f"state keys: {list(modality['state'].keys())}"
    )

    args_list = [
        (task, args.data_root, modality, args.action_mode)
        for task in tasks
    ]
    pooled = defaultdict(list)

    with Pool(args.num_workers) as pool:
        for i, (task, arrays) in enumerate(pool.imap_unordered(load_task, args_list)):
            status = "OK" if arrays else "NO DATA"
            print(f"[{i + 1}/{len(tasks)}] [{status}] {task}")
            for key, arr in arrays.items():
                pooled[key].append(arr)

    result = {}
    for mod, flat_key in [("state", "observation.state"), ("action", "action")]:
        subkeys = [sk for sk in modality[mod].keys() if (mod, sk) in pooled]
        flat = {name: [] for name in STAT_NAMES}
        for sk in subkeys:
            concatenated = np.concatenate(pooled[(mod, sk)])
            per_key = compute_stats(concatenated)
            for name in STAT_NAMES:
                values = per_key[name]
                flat[name].extend(values if isinstance(values, list) else [values])
        flat["_key_order"] = subkeys
        result[flat_key] = flat

    # Match the runtime unified_norm cache key exactly:
    # - datasets are keyed by dataset_path.name, not the full relative task path
    # - eef variants are already disambiguated by action_type
    cache_payload = {
        "datasets": sorted(Path(task).name for task in tasks),
        "coordinate_system": args.coordinate_system,
        "action_mode": args.action_mode,
        "action_type": action_type,
    }
    cache_key = hashlib.md5(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()[:12]
    cache_dir = Path(args.data_root) / ".unified_norm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"unified_stats_{cache_key}.json"
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {cache_path}")


if __name__ == "__main__":
    main()
