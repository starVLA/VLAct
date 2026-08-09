"""
Count total_episodes and total_frames per folder for split_aloha datasets.
"""

import json
import os

ROOT = "/project/vonneumann1/datasets/InternData-A1/sim_updated"
ROBOT_FILTER = "split_aloha"


def discover_datasets():
    datasets = []
    for category in sorted(os.listdir(ROOT)):
        cat_path = os.path.join(ROOT, category)
        if not os.path.isdir(cat_path):
            continue
        robot_path = os.path.join(cat_path, ROBOT_FILTER)
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
                info = os.path.join(sub_path, "meta", "info.json")
                if os.path.isfile(info):
                    datasets.append(sub_path)
                else:
                    for obj in sorted(os.listdir(sub_path)):
                        obj_path = os.path.join(sub_path, obj)
                        if os.path.isdir(obj_path) and os.path.isfile(
                            os.path.join(obj_path, "meta", "info.json")
                        ):
                            datasets.append(obj_path)
    return datasets


def main():
    datasets = discover_datasets()
    print(f"Found {len(datasets)} {ROBOT_FILTER} datasets\n")

    total_ep = 0
    total_fr = 0
    total_vid = 0

    print(f"{'Folder':<90s} {'Episodes':>10s} {'Frames':>12s} {'Videos':>10s}")
    print("-" * 125)

    for ds in datasets:
        rel = os.path.relpath(ds, ROOT)
        info_path = os.path.join(ds, "meta", "info.json")
        try:
            with open(info_path) as f:
                info = json.load(f)
        except Exception as e:
            print(f"{rel:<90s} ERROR: {e}")
            continue

        ep = info.get("total_episodes", 0)
        fr = info.get("total_frames", 0)
        vid = info.get("total_videos", 0)
        total_ep += ep
        total_fr += fr
        total_vid += vid

        print(f"{rel:<90s} {ep:>10d} {fr:>12d} {vid:>10d}")

    print("-" * 125)
    print(f"{'TOTAL':<90s} {total_ep:>10d} {total_fr:>12d} {total_vid:>10d}")


if __name__ == "__main__":
    main()
