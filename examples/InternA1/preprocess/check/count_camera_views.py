#!/usr/bin/env python3
"""
Count camera views used by all split_aloha datasets.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = Path("/project/vonneumann1/wcy/dataset/VLM-VLA/InternData-A1/sim_updated")
DATASET_TAG = "split_aloha"


def discover_datasets(root: Path):
    datasets = []

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue

        split_root = category_dir / DATASET_TAG
        if not split_root.is_dir():
            continue

        for task_dir in sorted(split_root.iterdir()):
            if not task_dir.is_dir():
                continue

            for sub_dir in sorted(task_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue

                if (sub_dir / "meta" / "info.json").is_file():
                    datasets.append(sub_dir)
                    continue

                for obj_dir in sorted(sub_dir.iterdir()):
                    if (obj_dir / "meta" / "info.json").is_file():
                        datasets.append(obj_dir)

    return sorted(set(datasets))


def collect_views(dataset_dir: Path):
    view_to_videos = Counter()
    for video_path in dataset_dir.glob("videos/chunk-*/*/episode_*.mp4"):
        view_to_videos[video_path.parent.name] += 1
    return view_to_videos


def build_summary(root: Path):
    datasets = discover_datasets(root)
    dataset_count_by_view = Counter()
    video_count_by_view = Counter()
    rows = []

    for dataset_dir in datasets:
        view_to_videos = collect_views(dataset_dir)
        views = sorted(view_to_videos)
        rel_path = dataset_dir.relative_to(root)

        for view_name, num_videos in view_to_videos.items():
            dataset_count_by_view[view_name] += 1
            video_count_by_view[view_name] += num_videos

        rows.append(
            {
                "dataset": str(rel_path),
                "num_views": len(views),
                "views": views,
                "videos_per_view": dict(sorted(view_to_videos.items())),
            }
        )

    return rows, dataset_count_by_view, video_count_by_view


def print_summary(rows, dataset_count_by_view, video_count_by_view):
    print(f"Found {len(rows)} split_aloha datasets\n")

    print("=== View Summary ===")
    print(f"{'View':<24s} {'Datasets':>10s} {'Videos':>12s}")
    print("-" * 48)
    for view_name in sorted(dataset_count_by_view):
        print(
            f"{view_name:<24s} "
            f"{dataset_count_by_view[view_name]:>10d} "
            f"{video_count_by_view[view_name]:>12d}"
        )

    print("\n=== Dataset Details ===")
    for row in rows:
        view_text = ", ".join(row["views"]) if row["views"] else "(no videos found)"
        print(f"{row['dataset']}: {view_text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    rows, dataset_count_by_view, video_count_by_view = build_summary(args.root)
    print_summary(rows, dataset_count_by_view, video_count_by_view)

    if args.output_json:
        payload = {
            "root": str(args.root),
            "num_datasets": len(rows),
            "view_summary": [
                {
                    "view": view_name,
                    "num_datasets": dataset_count_by_view[view_name],
                    "num_videos": video_count_by_view[view_name],
                }
                for view_name in sorted(dataset_count_by_view)
            ],
            "datasets": rows,
        }
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
