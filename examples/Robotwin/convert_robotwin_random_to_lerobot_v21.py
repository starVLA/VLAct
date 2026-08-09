"""
Convert RoboTwin-Random raw task folders into per-task LeRobot v2.1 datasets.

The generated directory for each task matches the on-disk layout used by the
existing `RoboTwin-Clean/<task>` datasets in this repository:

    <dst_root>/<task>/
      data/
      videos/
      meta/

Example:
    conda activate starVLA-dev
    python examples/Robotwin/convert_robotwin_random_to_lerobot_v21.py \
        --src-root /project/vonneumann1/wcy/code/starVLA-dev/playground/Datasets/RoboTwin-Random \
        --dst-root /project/vonneumann1/datasets/RoboTwin-Random-v21

Notes:
  - Raw inputs are expected to look like:
      <src_root>/<task>/aloha-agilex_randomized_500/
        data/episode*.hdf5
        instructions/episode*.json
        scene_info.json
  - All cameras found under `/observation/*/rgb` are preserved.
  - Video resolution is kept exactly as stored in the raw files.
  - Raw `joint_action/vector` is converted with the same temporal alignment as
    the existing RoboTwin-Clean parquet files in this workspace:
      observation.state[t] = joint_vector[t]
      action[t]            = joint_vector[min(t + 1, last)]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
import pandas as pd
import tqdm


MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]

DEFAULT_CAMERA_NAME_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
    "front_camera": "cam_front",
}

EPISODE_RE = re.compile(r"episode(\d+)\.hdf5$")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ConvertConfig:
    src_root: Path
    dst_root: Path
    tasks: list[str] | None
    episodes_per_task: int | None
    instruction_split: str
    instruction_policy: str
    seed: int
    fps: int
    overwrite: bool
    video_codec: str
    num_processes: int
    copy_scene_info: bool


def parse_args() -> ConvertConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/project/vonneumann1/wcy/code/starVLA-dev/playground/Datasets/RoboTwin-Random"),
        help="Root of RoboTwin-Random raw task folders.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        required=True,
        help="Destination root that will contain one LeRobot dataset directory per task.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of task names to convert. Default: convert every task under src-root.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=None,
        help="Optional cap on episodes converted per task.",
    )
    parser.add_argument(
        "--instruction-split",
        choices=["seen", "unseen", "prefer_seen", "prefer_unseen", "all"],
        default="seen",
        help="Which prompt bucket to sample task text from.",
    )
    parser.add_argument(
        "--instruction-policy",
        choices=["random", "first", "cycle"],
        default="random",
        help="How to pick one instruction for each episode from the chosen split.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic prompt selection.")
    parser.add_argument("--fps", type=int, default=15, help="Output LeRobot dataset fps.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output task directory if present.",
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default="h264",
        help="Output video codec label. Recommended: h264 or av1.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=64,
        help="Worker process count for episode-level parallel conversion.",
    )
    parser.add_argument(
        "--copy-scene-info",
        action="store_true",
        help="Copy the raw task-level scene_info.json into meta/raw_scene_info.json.",
    )
    args = parser.parse_args()
    return ConvertConfig(
        src_root=args.src_root,
        dst_root=args.dst_root,
        tasks=args.tasks,
        episodes_per_task=args.episodes_per_task,
        instruction_split=args.instruction_split,
        instruction_policy=args.instruction_policy,
        seed=args.seed,
        fps=args.fps,
        overwrite=args.overwrite,
        video_codec=args.video_codec,
        num_processes=max(1, args.num_processes),
        copy_scene_info=args.copy_scene_info,
    )


def discover_tasks(src_root: Path, requested_tasks: list[str] | None) -> list[str]:
    if requested_tasks:
        return requested_tasks

    tasks = []
    for child in sorted(src_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if find_raw_task_dir(child) is not None:
            tasks.append(child.name)
    return tasks


def find_raw_task_dir(task_root: Path) -> Path | None:
    candidates = []
    for child in sorted(task_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "data").is_dir() and (child / "instructions").is_dir():
            candidates.append(child)
    if not candidates:
        return None
    if len(candidates) > 1:
        raise RuntimeError(f"Expected one raw dataset dir under {task_root}, found {len(candidates)}: {candidates}")
    return candidates[0]


def list_episode_ids(raw_task_dir: Path) -> list[int]:
    episode_ids = []
    for child in sorted((raw_task_dir / "data").iterdir()):
        if not child.is_file():
            continue
        match = EPISODE_RE.match(child.name)
        if match:
            episode_ids.append(int(match.group(1)))
    return episode_ids


def decode_image_buffer(buf: np.ndarray) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return img


def normalize_camera_name(raw_camera_name: str) -> str:
    if raw_camera_name in DEFAULT_CAMERA_NAME_MAP:
        return DEFAULT_CAMERA_NAME_MAP[raw_camera_name]

    name = raw_camera_name.lower().strip()
    name = name.removesuffix("_camera")
    name = NON_ALNUM_RE.sub("_", name).strip("_")
    return name or raw_camera_name


def discover_camera_mapping(ep_path: Path) -> OrderedDict[str, str]:
    with h5py.File(ep_path, "r") as f:
        raw_camera_names = sorted(f["observation"].keys())

    mapping: OrderedDict[str, str] = OrderedDict()
    used_names: set[str] = set()
    for raw_name in raw_camera_names:
        output_name = normalize_camera_name(raw_name)
        base_name = output_name
        suffix = 2
        while output_name in used_names:
            output_name = f"{base_name}_{suffix}"
            suffix += 1
        mapping[raw_name] = output_name
        used_names.add(output_name)
    return mapping


def load_instruction_candidates(instruction_path: Path, split: str) -> list[str]:
    with instruction_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    seen = [str(x) for x in data.get("seen", []) if str(x).strip()]
    unseen = [str(x) for x in data.get("unseen", []) if str(x).strip()]

    if split == "seen":
        candidates = seen
    elif split == "unseen":
        candidates = unseen
    elif split == "prefer_seen":
        candidates = seen or unseen
    elif split == "prefer_unseen":
        candidates = unseen or seen
    elif split == "all":
        candidates = seen + unseen
    else:
        raise ValueError(f"Unsupported instruction split: {split}")

    if not candidates:
        raise ValueError(f"No instruction candidates found in {instruction_path} for split={split}")
    return candidates


def choose_instruction(candidates: list[str], episode_id: int, cfg: ConvertConfig) -> str:
    if cfg.instruction_policy == "first":
        return candidates[0]
    if cfg.instruction_policy == "cycle":
        return candidates[episode_id % len(candidates)]
    if cfg.instruction_policy == "random":
        rng = random.Random(cfg.seed + episode_id)
        return rng.choice(candidates)
    raise ValueError(f"Unsupported instruction policy: {cfg.instruction_policy}")


def load_episode_arrays(
    ep_path: Path,
    camera_mapping: dict[str, str],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with h5py.File(ep_path, "r") as f:
        joint_vector = np.asarray(f["joint_action/vector"][:], dtype=np.float32)
        if joint_vector.ndim != 2 or joint_vector.shape[1] != len(MOTORS):
            raise ValueError(f"Unexpected joint_action/vector shape in {ep_path}: {joint_vector.shape}")

        states = joint_vector.copy()
        actions = np.concatenate([joint_vector[1:], joint_vector[-1:]], axis=0)

        images_per_camera: dict[str, np.ndarray] = {}
        for raw_camera, output_camera in camera_mapping.items():
            dataset = f[f"observation/{raw_camera}/rgb"]
            if dataset.ndim == 4:
                images = np.asarray(dataset[:])
            else:
                images = np.stack([decode_image_buffer(buf) for buf in dataset], axis=0)
            images_per_camera[output_camera] = images

    num_frames = states.shape[0]
    for output_camera, images in images_per_camera.items():
        if len(images) != num_frames:
            raise ValueError(
                f"Frame count mismatch for {ep_path} camera {output_camera}: "
                f"{len(images)} images vs {num_frames} state/action frames"
            )
    return images_per_camera, states, actions


def build_modality_template(camera_names: Iterable[str]) -> dict:
    modality = {
        "action": {
            "left_joints": {"start": 0, "end": 6, "original_key": "action"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "action"},
            "right_joints": {"start": 7, "end": 13, "original_key": "action"},
            "right_gripper": {"start": 13, "end": 14, "original_key": "action"},
        },
        "state": {
            "left_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
            "left_gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
            "right_joints": {"start": 7, "end": 13, "original_key": "observation.state"},
            "right_gripper": {"start": 13, "end": 14, "original_key": "observation.state"},
        },
        "video": {},
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            }
        },
    }
    for camera_name in camera_names:
        modality["video"][camera_name] = {"original_key": f"observation.images.{camera_name}"}
    return modality


def write_modality_json(meta_dir: Path, camera_names: Iterable[str]) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / "modality.json").open("w", encoding="utf-8") as f:
        json.dump(build_modality_template(camera_names), f, indent=4)
        f.write("\n")


def maybe_copy_scene_info(raw_task_dir: Path, meta_dir: Path, enabled: bool) -> None:
    if not enabled:
        return
    scene_info_path = raw_task_dir / "scene_info.json"
    if scene_info_path.exists():
        shutil.copy2(scene_info_path, meta_dir / "raw_scene_info.json")


def prepare_task_output_dir(dst_task_dir: Path, overwrite: bool) -> None:
    if dst_task_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_task_dir}. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(dst_task_dir)

    (dst_task_dir / "data").mkdir(parents=True, exist_ok=True)
    (dst_task_dir / "videos").mkdir(parents=True, exist_ok=True)
    (dst_task_dir / "meta").mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def write_episode_video(video_path: Path, frames: np.ndarray, fps: int, codec: str) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]

    codec_name = codec.lower()
    if codec_name in {"h264", "avc", "avc1", "libx264"}:
        ffmpeg_codec = "libx264"
        info_codec = "h264"
        extra_args = ["-preset", "veryfast", "-crf", "18"]
    elif codec_name in {"av1", "libaom-av1"}:
        ffmpeg_codec = "libaom-av1"
        info_codec = "av1"
        extra_args = ["-cpu-used", "8", "-crf", "30", "-row-mt", "1"]
    else:
        ffmpeg_codec = codec
        info_codec = codec
        extra_args = []

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        ffmpeg_codec,
        *extra_args,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        return_code = proc.wait()
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()

    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path} with codec={codec} (resolved {info_codec}): {stderr.strip()}"
        )


def build_info_json(
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    camera_shapes: dict[str, tuple[int, int]],
    fps: int,
    codec: str,
) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(MOTORS)],
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": [len(MOTORS)],
            "names": [MOTORS],
        },
    }

    for camera_name, (image_height, image_width) in camera_shapes.items():
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": [3, image_height, image_width],
            "names": ["channels", "height", "width"],
            "info": {
                "video.height": image_height,
                "video.width": image_width,
                "video.codec": codec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    for scalar_name, dtype in [
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ]:
        features[scalar_name] = {
            "dtype": dtype,
            "shape": [1],
            "names": None,
        }

    return {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(camera_shapes),
        "total_chunks": ceil(total_episodes / 1000),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {
            "train": f"0:{total_episodes}",
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def normalize_info_codec(codec: str) -> str:
    codec_name = codec.lower()
    if codec_name in {"h264", "avc", "avc1", "libx264"}:
        return "h264"
    if codec_name in {"av1", "libaom-av1"}:
        return "av1"
    return codec


def write_info_json(
    meta_dir: Path,
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    camera_shapes: dict[str, tuple[int, int]],
    cfg: ConvertConfig,
) -> None:
    info = build_info_json(
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=total_tasks,
        camera_shapes=camera_shapes,
        fps=cfg.fps,
        codec=normalize_info_codec(cfg.video_codec),
    )
    with (meta_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
        f.write("\n")


def iter_episode_ids(all_episode_ids: list[int], limit: int | None) -> Iterable[int]:
    if limit is None:
        return all_episode_ids
    return all_episode_ids[:limit]


def get_episode_num_frames(ep_path: Path) -> int:
    with h5py.File(ep_path, "r") as f:
        return int(f["joint_action/vector"].shape[0])


def convert_single_episode(job: dict) -> dict:
    ep_path = Path(job["ep_path"])
    instruction_path = Path(job["instruction_path"])
    dst_task_dir = Path(job["dst_task_dir"])
    camera_mapping = OrderedDict(job["camera_mapping"])
    output_episode_index = int(job["output_episode_index"])
    global_frame_start = int(job["global_frame_start"])
    fps = int(job["fps"])
    video_codec = str(job["video_codec"])
    instruction_split = str(job["instruction_split"])
    instruction_policy = str(job["instruction_policy"])
    seed = int(job["seed"])
    source_episode_id = int(job["source_episode_id"])

    local_cfg = ConvertConfig(
        src_root=Path("."),
        dst_root=Path("."),
        tasks=None,
        episodes_per_task=None,
        instruction_split=instruction_split,
        instruction_policy=instruction_policy,
        seed=seed,
        fps=fps,
        overwrite=False,
        video_codec=video_codec,
        num_processes=1,
        copy_scene_info=False,
    )

    images_per_camera, states, actions = load_episode_arrays(ep_path, camera_mapping)
    instruction = choose_instruction(
        load_instruction_candidates(instruction_path, instruction_split),
        source_episode_id,
        local_cfg,
    )

    rows = []
    for frame_idx in range(states.shape[0]):
        rows.append(
            {
                "observation.state": states[frame_idx].astype(np.float32),
                "action": actions[frame_idx].astype(np.float32),
                "timestamp": np.float32(frame_idx / fps),
                "frame_index": np.int64(frame_idx),
                "episode_index": np.int64(output_episode_index),
                "index": np.int64(global_frame_start + frame_idx),
                "task_index": instruction,
            }
        )

    chunk_index = output_episode_index // 1000
    parquet_path = (
        dst_task_dir
        / "data"
        / f"chunk-{chunk_index:03d}"
        / f"episode_{output_episode_index:06d}.parquet"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    for camera_name, frames in images_per_camera.items():
        video_path = (
            dst_task_dir
            / "videos"
            / f"chunk-{chunk_index:03d}"
            / f"observation.images.{camera_name}"
            / f"episode_{output_episode_index:06d}.mp4"
        )
        write_episode_video(video_path, frames, fps, video_codec)

    camera_shapes = {
        camera_name: tuple(int(x) for x in frames[0].shape[:2])
        for camera_name, frames in images_per_camera.items()
    }
    return {
        "episode_index": output_episode_index,
        "instruction": instruction,
        "length": len(rows),
        "camera_shapes": camera_shapes,
    }


def convert_task(task_name: str, cfg: ConvertConfig) -> None:
    task_root = cfg.src_root / task_name
    raw_task_dir = find_raw_task_dir(task_root)
    if raw_task_dir is None:
        raise FileNotFoundError(f"Could not find raw task directory under {task_root}")

    all_episode_ids = list_episode_ids(raw_task_dir)
    if not all_episode_ids:
        raise RuntimeError(f"No HDF5 episodes found under {raw_task_dir / 'data'}")

    selected_episode_ids = list(iter_episode_ids(all_episode_ids, cfg.episodes_per_task))
    first_episode_path = raw_task_dir / "data" / f"episode{selected_episode_ids[0]}.hdf5"
    camera_mapping = discover_camera_mapping(first_episode_path)
    dst_task_dir = cfg.dst_root / task_name
    prepare_task_output_dir(dst_task_dir, cfg.overwrite)

    frame_lengths = [
        get_episode_num_frames(raw_task_dir / "data" / f"episode{episode_id}.hdf5")
        for episode_id in selected_episode_ids
    ]

    global_frame_starts = []
    total_frames = 0
    for length in frame_lengths:
        global_frame_starts.append(total_frames)
        total_frames += length

    jobs = []
    for output_episode_index, episode_id in enumerate(selected_episode_ids):
        jobs.append(
            {
                "ep_path": str(raw_task_dir / "data" / f"episode{episode_id}.hdf5"),
                "instruction_path": str(raw_task_dir / "instructions" / f"episode{episode_id}.json"),
                "dst_task_dir": str(dst_task_dir),
                "camera_mapping": list(camera_mapping.items()),
                "output_episode_index": output_episode_index,
                "global_frame_start": global_frame_starts[output_episode_index],
                "fps": cfg.fps,
                "video_codec": cfg.video_codec,
                "instruction_split": cfg.instruction_split,
                "instruction_policy": cfg.instruction_policy,
                "seed": cfg.seed,
                "source_episode_id": episode_id,
            }
        )

    results: list[dict] = []
    progress = tqdm.tqdm(total=len(jobs), desc=task_name, leave=True)
    with ProcessPoolExecutor(max_workers=cfg.num_processes) as executor:
        futures = [executor.submit(convert_single_episode, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            progress.update(1)
    progress.close()

    results.sort(key=lambda x: x["episode_index"])

    task_to_index: OrderedDict[str, int] = OrderedDict()
    episodes_meta: list[dict] = []
    camera_shapes: dict[str, tuple[int, int]] = {}

    for result in results:
        instruction = result["instruction"]
        if instruction not in task_to_index:
            task_to_index[instruction] = len(task_to_index)
        task_index = task_to_index[instruction]

        parquet_path = (
            dst_task_dir
            / "data"
            / f"chunk-{result['episode_index'] // 1000:03d}"
            / f"episode_{result['episode_index']:06d}.parquet"
        )
        df = pd.read_parquet(parquet_path)
        df["task_index"] = np.int64(task_index)
        df.to_parquet(parquet_path, index=False)

        episodes_meta.append(
            {
                "episode_index": result["episode_index"],
                "tasks": [instruction],
                "length": result["length"],
            }
        )
        if not camera_shapes:
            camera_shapes = result["camera_shapes"]

    tasks_meta = [
        {
            "task_index": task_index,
            "task": task_text,
        }
        for task_text, task_index in task_to_index.items()
    ]

    write_jsonl(dst_task_dir / "meta" / "tasks.jsonl", tasks_meta)
    write_jsonl(dst_task_dir / "meta" / "episodes.jsonl", episodes_meta)
    write_info_json(
        dst_task_dir / "meta",
        total_episodes=len(selected_episode_ids),
        total_frames=total_frames,
        total_tasks=len(tasks_meta),
        camera_shapes=camera_shapes,
        cfg=cfg,
    )
    write_modality_json(dst_task_dir / "meta", camera_mapping.values())
    maybe_copy_scene_info(raw_task_dir, dst_task_dir / "meta", cfg.copy_scene_info)


def main() -> None:
    cfg = parse_args()

    if not cfg.src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {cfg.src_root}")

    tasks = discover_tasks(cfg.src_root, cfg.tasks)
    if not tasks:
        raise RuntimeError(f"No convertible tasks found under {cfg.src_root}")

    print(f"Found {len(tasks)} task(s) to convert.")
    print(f"Source root:      {cfg.src_root}")
    print(f"Destination root: {cfg.dst_root}")
    print(f"Instruction mode: {cfg.instruction_split}/{cfg.instruction_policy}")
    print(f"Num processes:    {cfg.num_processes}")

    for task_name in tasks:
        print(f"\n=== Converting task: {task_name} ===")
        convert_task(task_name, cfg)

    print("\nAll requested tasks have been converted.")


if __name__ == "__main__":
    main()
