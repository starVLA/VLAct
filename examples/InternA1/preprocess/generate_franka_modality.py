#!/usr/bin/env python3
"""Generate modality.json for InternA1 Franka tasks."""

import argparse
import json
from pathlib import Path

REQUIRED_FEATURES = {
    "images.rgb.head",
    "images.rgb.hand",
    "states.gripper.pose",
    "states.gripper.position",
    "actions.gripper.pose",
    "actions.gripper.position",
}
TASK_LIST = Path(__file__).resolve().parents[3] / "playground" / "Datasets" / "InternData-A1" / "franka_tasks.txt"


def build_modality_payload() -> dict:
    return {
        "state": {
            "eef_position": {
                "start": 0,
                "end": 3,
                "dtype": "float32",
                "original_key": "states.gripper.pose",
            },
            "eef_rotation": {
                "start": 3,
                "end": 6,
                "dtype": "float32",
                "rotation_type": "euler_angles_rpy",
                "original_key": "states.gripper.pose",
            },
            "gripper_position": {
                "start": 0,
                "end": 1,
                "dtype": "float32",
                "original_key": "states.gripper.position",
            },
        },
        "action": {
            "eef_position": {
                "start": 0,
                "end": 3,
                "dtype": "float32",
                "original_key": "actions.gripper.pose",
            },
            "eef_rotation": {
                "start": 3,
                "end": 6,
                "dtype": "float32",
                "rotation_type": "euler_angles_rpy",
                "original_key": "actions.gripper.pose",
            },
            "gripper_position": {
                "start": 0,
                "end": 1,
                "dtype": "float32",
                "original_key": "actions.gripper.position",
            },
        },
        "video": {
            "primary_image": {
                "original_key": "images.rgb.head",
            },
            "wrist_image": {
                "original_key": "images.rgb.hand",
            },
        },
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            },
        },
    }


def generate_one(dataset_dir: Path, data_root: Path, overwrite: bool) -> str:
    task = str(dataset_dir.relative_to(data_root))
    info_path = dataset_dir / "meta" / "info.json"
    modality_path = dataset_dir / "meta" / "modality.json"

    if not info_path.exists():
        return f"[NO INFO] {task}"
    if modality_path.exists() and not overwrite:
        return f"[SKIP] {task}"

    with open(info_path, "r", encoding="utf-8") as f:
        info_meta = json.load(f)
    features = info_meta.get("features", {})
    if not REQUIRED_FEATURES.issubset(features):
        return f"[UNSUPPORTED] {task}"

    modality_path.parent.mkdir(parents=True, exist_ok=True)
    with open(modality_path, "w", encoding="utf-8") as f:
        json.dump(build_modality_payload(), f, indent=2)
    return f"[DONE] {task}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/project/vonneumann1/datasets/InternData-A1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(TASK_LIST, "r", encoding="utf-8") as f:
        task_paths = [line.strip() for line in f if line.strip()]
    print(f"Total franka tasks: {len(task_paths)}")
    for idx, task_path in enumerate(task_paths, start=1):
        print(
            f"[{idx}/{len(task_paths)}] "
            f"{generate_one(args.data_root / task_path, args.data_root, args.overwrite)}"
        )


if __name__ == "__main__":
    main()
