#!/usr/bin/env python3
"""Generate minimal modality.json files for RoboCOIN datasets."""

import argparse
import json
from pathlib import Path

DATA_ROOT = Path("/project/vonneumann1/datasets/RoboCOIN-cy")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--prefixes",
        nargs="*",
        default=["Cobot_Magic_", "Split_aloha_"],
        help="Only process dataset dirs with these prefixes.",
    )
    return parser.parse_args()


def list_tasks(data_root: Path, prefixes: list[str]) -> list[Path]:
    tasks = []
    for path in sorted(data_root.iterdir()):
        if not path.is_dir():
            continue
        if prefixes and not any(path.name.startswith(prefix) for prefix in prefixes):
            continue
        if (path / "meta" / "info.json").exists():
            tasks.append(path)
    return tasks


def build_modality(info: dict) -> dict:
    features = info["features"]
    state_dim = features["observation.state"]["shape"][0]
    action_dim = features["action"]["shape"][0]
    if state_dim != action_dim:
        raise ValueError(f"state/action dim mismatch: {state_dim} vs {action_dim}")

    if state_dim == 14:
        state_slices = {
            "left_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
            "right_joints": {"start": 7, "end": 13, "original_key": "observation.state"},
            "right_gripper": {"start": 13, "end": 14, "original_key": "observation.state"},
        }
        action_slices = {
            "left_joints": {"start": 0, "end": 6, "original_key": "action"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "action"},
            "right_joints": {"start": 7, "end": 13, "original_key": "action"},
            "right_gripper": {"start": 13, "end": 14, "original_key": "action"},
        }
    elif state_dim == 26:
        state_slices = {
            "left_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
            "right_joints": {"start": 13, "end": 19, "original_key": "observation.state"},
            "right_gripper": {"start": 19, "end": 20, "original_key": "observation.state"},
        }
        action_slices = {
            "left_joints": {"start": 0, "end": 6, "original_key": "action"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "action"},
            "right_joints": {"start": 13, "end": 19, "original_key": "action"},
            "right_gripper": {"start": 19, "end": 20, "original_key": "action"},
        }
    else:
        raise ValueError(f"Unsupported RoboCOIN state/action dim: {state_dim}")

    modality = {
        "state": state_slices,
        "action": action_slices,
        "video": {},
        "annotation": {},
    }

    for key in sorted(features):
        if key.startswith("observation.images."):
            name = key.removeprefix("observation.images.")
            modality["video"][name] = {"original_key": key}

    for key in ("task_index", "scene_annotation", "subtask_annotation"):
        if key in features:
            modality["annotation"][key] = {"original_key": key}

    return modality


def main():
    args = parse_args()
    tasks = list_tasks(args.data_root, args.prefixes)
    print(f"Total tasks: {len(tasks)}")

    count = 0
    for dataset_dir in tasks:
        info_path = dataset_dir / "meta" / "info.json"
        out_path = dataset_dir / "meta" / "modality.json"
        with open(info_path) as f:
            info = json.load(f)
        with open(out_path, "w") as f:
            json.dump(build_modality(info), f, indent=2)
        count += 1
        print(f"[{count}/{len(tasks)}] {dataset_dir.name}")

    print("All done.")


if __name__ == "__main__":
    main()
