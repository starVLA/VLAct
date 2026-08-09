#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = "/project/vonneumann1/wcy/code/starVLA-dev/playground/Datasets/InternData-A1/sim_updated"
DEFAULT_STATS = "/project/vonneumann1/wcy/code/starVLA-dev/results/Checkpoints/0315_interna1_split_aloha_qwen3OFT/dataset_statistics.json"
DEFAULT_OUTPUT = "/project/vonneumann1/wcy/code/starVLA-dev/examples/InternA1/preprocess/check/out_of_range_videos.txt"
DEBUG_LIMIT = 10000

ACTION_COLUMNS = [
    "actions.left_joint.position",
    "actions.left_gripper.position",
    "actions.right_joint.position",
    "actions.right_gripper.position",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check how many actions in InternData-A1 sim_updated are outside q01/q99."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--stats", default=DEFAULT_STATS)
    parser.add_argument("--robot-filter", default="split_aloha")
    parser.add_argument("--video-key", default="images.rgb.head")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--debug", action="store_true", help="Only process first 10000 videos.")
    return parser.parse_args()


def load_bounds(stats_path: str):
    with open(stats_path, "r") as f:
        stats = json.load(f)

    action_stats = stats["new_embodiment"]["action"]
    q01 = np.asarray(action_stats["q01"], dtype=np.float32)
    q99 = np.asarray(action_stats["q99"], dtype=np.float32)
    min_vals = np.asarray(action_stats["min"], dtype=np.float32)
    max_vals = np.asarray(action_stats["max"], dtype=np.float32)
    return q01, q99, min_vals, max_vals


def discover_dataset_dirs(root: str, robot_filter: str):
    dataset_dirs = []
    for category in sorted(os.listdir(root)):
        cat_path = os.path.join(root, category)
        if not os.path.isdir(cat_path):
            continue
        robot_path = os.path.join(cat_path, robot_filter)
        if not os.path.isdir(robot_path):
            continue
        for task in sorted(os.listdir(robot_path)):
            task_path = os.path.join(robot_path, task)
            if not os.path.isdir(task_path):
                continue
            for sub in sorted(os.listdir(task_path)):
                sub_path = os.path.join(task_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                info_path = os.path.join(sub_path, "meta", "info.json")
                if os.path.isfile(info_path):
                    dataset_dirs.append(sub_path)
                else:
                    for obj in sorted(os.listdir(sub_path)):
                        obj_path = os.path.join(sub_path, obj)
                        obj_info = os.path.join(obj_path, "meta", "info.json")
                        if os.path.isdir(obj_path) and os.path.isfile(obj_info):
                            dataset_dirs.append(obj_path)
    return sorted(dataset_dirs)


def is_compatible_dataset(dataset_dir: str):
    info_path = Path(dataset_dir) / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    features = info.get("features", {})
    return all(col in features for col in ACTION_COLUMNS)


def collect_parquet_paths(dataset_dirs):
    parquet_paths = []
    skipped_videos = 0
    for dataset_dir in dataset_dirs:
        if not is_compatible_dataset(dataset_dir):
            info_path = Path(dataset_dir) / "meta" / "info.json"
            with open(info_path, "r") as f:
                info = json.load(f)
            skipped_videos += int(info.get("total_episodes", 0))
            continue

        data_dir = Path(dataset_dir) / "data"
        if not data_dir.is_dir():
            continue
        for chunk_dir in sorted(data_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            for parquet_path in sorted(chunk_dir.glob("*.parquet")):
                parquet_paths.append(parquet_path)
    return parquet_paths, skipped_videos


def parquet_to_video_path(parquet_path: Path, video_key: str):
    return parquet_path.parent.parent.parent / "videos" / parquet_path.parent.name / video_key / parquet_path.with_suffix(".mp4").name


def load_action_matrix(parquet_path: Path):
    df = pd.read_parquet(parquet_path, columns=ACTION_COLUMNS)
    left_joint = np.vstack(df["actions.left_joint.position"].to_numpy()).astype(np.float32)
    left_gripper = df["actions.left_gripper.position"].to_numpy(dtype=np.float32).reshape(-1, 1)
    right_joint = np.vstack(df["actions.right_joint.position"].to_numpy()).astype(np.float32)
    right_gripper = df["actions.right_gripper.position"].to_numpy(dtype=np.float32).reshape(-1, 1)
    return np.concatenate([left_joint, left_gripper, right_joint, right_gripper], axis=1)


def main():
    args = parse_args()
    q01, q99, min_vals, max_vals = load_bounds(args.stats)

    dataset_dirs = discover_dataset_dirs(args.root, args.robot_filter)
    parquet_paths, skipped_videos = collect_parquet_paths(dataset_dirs)

    if args.debug:
        parquet_paths = parquet_paths[:DEBUG_LIMIT]

    total_videos = len(parquet_paths)
    total_action_rows = 0
    out_of_range_action_rows = 0
    out_of_range_videos = []

    print(f"Found {len(dataset_dirs)} dataset dirs")
    print(f"Checking {total_videos} videos")
    print(f"robot_filter={args.robot_filter}")
    print(f"Skipped incompatible videos={skipped_videos}")
    print(f"video_key={args.video_key}")
    print(f"min={min_vals.tolist()}")
    print(f"max={max_vals.tolist()}")
    print(f"q01={q01.tolist()}")
    print(f"q99={q99.tolist()}")

    for idx, parquet_path in enumerate(parquet_paths, start=1):
        rel_path = os.path.relpath(parquet_path, args.root)
        if idx % 200 == 0 or idx == total_videos:
            print(f"[{idx}/{total_videos}] {rel_path}", flush=True)

        action_matrix = load_action_matrix(parquet_path)
        total_action_rows += len(action_matrix)

        import pdb; pdb.set_trace()

        row_out_of_range = np.any((action_matrix < q01) | (action_matrix > q99), axis=1)
        num_bad_rows = int(row_out_of_range.sum())
        out_of_range_action_rows += num_bad_rows

        if num_bad_rows > 0:
            video_path = parquet_to_video_path(parquet_path, args.video_key)
            out_of_range_videos.append(str(video_path))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_of_range_videos) + ("\n" if out_of_range_videos else ""))

    print()
    print(f"total_videos={total_videos}")
    print(f"total_action_rows={total_action_rows}")
    print(f"out_of_range_action_rows={out_of_range_action_rows}")
    print(f"out_of_range_videos={len(out_of_range_videos)}")
    print(f"skipped_videos={skipped_videos}")
    print(f"saved_video_list={output_path}")


if __name__ == "__main__":
    main()
