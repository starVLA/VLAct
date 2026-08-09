#!/usr/bin/env python3
"""Rearrange converted DOMINO LeRobot datasets into split/task folders."""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path


DOMINO_35_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_can_basket",
    "place_container_plate",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stamp_seal",
]

SPLIT_MARKERS = {
    "_clean_": ("Clean_Dynamic", "Clean"),
    "_randomized_": ("Random_Dynamic", "Randomized"),
}

REPO_ROOT = Path(__file__).resolve().parents[3]


def is_lerobot_dataset(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "meta" / "info.json").exists()
        and (path / "meta" / "modality.json").exists()
        and (path / "data").is_dir()
    )


def parse_domino_dataset_name(name: str) -> tuple[tuple[str, ...], str] | None:
    for task in sorted(DOMINO_35_TASKS, key=len, reverse=True):
        if not name.startswith(f"{task}_"):
            continue

        suffix = name[len(task) :]
        for marker, split_names in SPLIT_MARKERS.items():
            if marker in suffix:
                return split_names, task

    return None


def remove_destination(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def place_dataset(source: Path, destination: Path, mode: str, overwrite: bool, dry_run: bool) -> str:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return "exists"
        if not overwrite:
            return "conflict"
        if not dry_run:
            remove_destination(destination)

    if dry_run:
        return "planned"

    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(source.resolve(), destination)
    elif mode == "move":
        shutil.move(str(source), str(destination))
    elif mode == "copy":
        shutil.copytree(source, destination, symlinks=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return "created"


def rearrange(args: argparse.Namespace) -> Counter:
    source_root = args.source.resolve()
    target_root = args.target.resolve()
    counts: Counter = Counter()

    for source in sorted(source_root.iterdir()):
        if not is_lerobot_dataset(source):
            counts["skipped_non_lerobot"] += 1
            continue

        parsed = parse_domino_dataset_name(source.name)
        if parsed is None:
            counts["skipped_unmatched"] += 1
            print(f"[WARN] cannot parse DOMINO split/task from: {source.name}")
            continue

        split_names, task = parsed
        for split_name in split_names:
            destination = target_root / split_name / task / source.name
            status = place_dataset(source, destination, args.mode, args.overwrite, args.dry_run)
            counts[status] += 1
            if args.verbose or status in {"conflict", "planned"}:
                print(f"[{status}] {source} -> {destination}")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "playground/Datasets/DOMINO/converted_all_tasks/lerobot/local",
        help="Directory containing flat converted LeRobot datasets.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=REPO_ROOT / "playground/Datasets/DOMINO",
        help="DOMINO dataset root to receive Clean_Dynamic/Random_Dynamic and Clean/Randomized folders.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "move", "copy"),
        default="symlink",
        help="How to place datasets under the new layout. Default keeps the original local data untouched.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destinations.")
    parser.add_argument("--dry-run", action="store_true", help="Print intended operations without changing files.")
    parser.add_argument("--verbose", action="store_true", help="Print every created or existing mapping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = rearrange(args)
    print(
        "Summary: "
        + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    )


if __name__ == "__main__":
    main()
