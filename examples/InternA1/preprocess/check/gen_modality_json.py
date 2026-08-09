"""Generate modality.json for all InternData-A1 split_aloha datasets."""
import json
from pathlib import Path

DATA_ROOT = Path("/project/vonneumann1/datasets/InternData-A1")
TASK_LIST = Path(__file__).resolve().parent / "../../playground/Datasets/InternData-A1/split_aloha_tasks.txt"

MODALITY = {
    "state": {
        "left_joints":  {"start": 0, "end": 6, "original_key": "states.left_joint.position"},
        "left_gripper": {"start": 0, "end": 1, "original_key": "states.left_gripper.position"},
        "right_joints": {"start": 0, "end": 6, "original_key": "states.right_joint.position"},
        "right_gripper":{"start": 0, "end": 1, "original_key": "states.right_gripper.position"},
    },
    "action": {
        "left_joints":  {"start": 0, "end": 6, "original_key": "actions.left_joint.position"},
        "left_gripper": {"start": 0, "end": 1, "original_key": "actions.left_gripper.position"},
        "right_joints": {"start": 0, "end": 6, "original_key": "actions.right_joint.position"},
        "right_gripper":{"start": 0, "end": 1, "original_key": "actions.right_gripper.position"},
    },
    "video": {
        "rgb_head":       {"original_key": "images.rgb.head"},
        "rgb_hand_left":  {"original_key": "images.rgb.hand_left"},
        "rgb_hand_right": {"original_key": "images.rgb.hand_right"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}

tasks = [l.strip() for l in TASK_LIST.read_text().splitlines() if l.strip()]
count = 0
for t in tasks:
    meta_dir = DATA_ROOT / t / "meta"
    if not meta_dir.exists():
        print(f"[SKIP] {meta_dir}")
        continue
    out = meta_dir / "modality.json"
    with open(out, "w") as f:
        json.dump(MODALITY, f, indent=4)
    count += 1

print(f"Done. Generated {count} modality.json files.")
