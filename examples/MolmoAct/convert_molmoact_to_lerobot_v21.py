#!/usr/bin/env python3
"""
Convert MolmoAct subsets into LeRobot v2.1 video-style datasets.

The source MolmoAct subsets already look like LeRobot v2.1 datasets, but the
camera frames are embedded inside parquet files as HuggingFace `image` values.
This script normalizes them into the layout used by the datasets in this repo:

  <dst_root>/molmoact_<subset>_v21/
    data/
    videos/
    meta/

Key behaviors:
  - preserve episode/frame/task indices from the source subset
  - convert `first_view/second_view/wrist_image` into mp4 videos
  - rename numeric features to `observation.state` and `action`
  - generate `modality.json`
  - keep `episodes_stats.jsonl` with renamed feature keys
  - intentionally do NOT generate `stats.json`

Example:
  python examples/MolmoAct/convert_molmoact_to_lerobot_v21.py \
      --dst-root /project/vonneumann1/datasets/MolmoAct-v21 \
      --subsets household tabletop \
      --num-processes 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import tqdm


DEFAULT_SRC_ROOT = Path(
    "/project/vonneumann1/wcy/copy/starVLA-VLAct/playground/Datasets/MolmoAct-Dataset"
)
DEFAULT_DST_ROOT = Path(
    "/project/vonneumann1/wcy/copy/starVLA-VLAct/playground/Datasets/MolmoAct-Dataset"
)
SUBSET_TO_SOURCE_DIR = {
    "household": "molmoact_dataset_household/train",
    "tabletop": "molmoact_dataset_tabletop/train",
}
IMAGE_KEY_MAP = {
    "first_view": "observation.images.exterior_1",
    "second_view": "observation.images.exterior_2",
    "wrist_image": "observation.images.wrist",
}
TABULAR_KEY_MAP = {
    "state": "observation.state",
    "actions": "action",
    "timestamp": "timestamp",
    "frame_index": "frame_index",
    "episode_index": "episode_index",
    "index": "index",
    "task_index": "task_index",
}
STATE_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_roll",
    "eef_pitch",
    "eef_yaw",
    "gripper",
]
ACTION_NAMES = [
    "delta_eef_x",
    "delta_eef_y",
    "delta_eef_z",
    "delta_eef_roll",
    "delta_eef_pitch",
    "delta_eef_yaw",
    "gripper_command",
]


@dataclass(frozen=True)
class ConvertConfig:
    src_root: Path
    dst_root: Path
    subsets: list[str]
    max_episodes: int | None
    num_processes: int
    overwrite: bool
    ffmpeg_threads: int
    video_crf: int
    video_codec: str


@dataclass(frozen=True)
class EpisodeJob:
    src_parquet: Path
    dst_dataset_dir: Path
    episode_index: int
    episode_chunk: int
    fps: int
    overwrite: bool
    ffmpeg_threads: int
    video_crf: int
    video_codec: str


def parse_args() -> ConvertConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--dst-root", type=Path, default=DEFAULT_DST_ROOT)
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=sorted(SUBSET_TO_SOURCE_DIR),
        default=["household", "tabletop"],
        help="MolmoAct subsets to convert.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on episodes converted from each subset.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=min(os.cpu_count() or 1, 16),
        help="Episode-level worker process count.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing episode outputs under dst-root.",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=1,
        help="Threads used by each ffmpeg encoder process.",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=18,
        help="libx264 CRF for output videos.",
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default="libx264",
        help="ffmpeg video codec used for mp4 generation.",
    )
    args = parser.parse_args()

    return ConvertConfig(
        src_root=args.src_root,
        dst_root=args.dst_root,
        subsets=args.subsets,
        max_episodes=args.max_episodes,
        num_processes=max(1, args.num_processes),
        overwrite=args.overwrite,
        ffmpeg_threads=max(1, args.ffmpeg_threads),
        video_crf=args.video_crf,
        video_codec=args.video_codec,
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def dataset_dir_for_subset(dst_root: Path, subset: str) -> Path:
    return dst_root / f"molmoact_{subset}_v21"


def encode_video_ffmpeg(
    encoded_frames: list[bytes],
    output_path: Path,
    fps: int,
    codec: str,
    crf: int,
    threads: int,
) -> None:
    if not encoded_frames:
        raise ValueError(f"No frames provided for {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-threads",
        str(threads),
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for frame_bytes in encoded_frames:
        proc.stdin.write(frame_bytes)
    proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {output_path}: {stderr.strip()}")


def build_target_dataframe(source_df: pd.DataFrame) -> pd.DataFrame:
    target: dict[str, Any] = {}
    for src_key, dst_key in TABULAR_KEY_MAP.items():
        if src_key not in source_df.columns:
            raise KeyError(f"Missing required column: {src_key}")
        target[dst_key] = source_df[src_key].tolist()
    return pd.DataFrame(target)


def convert_one_episode(job: EpisodeJob) -> dict[str, Any]:
    dst_parquet = (
        job.dst_dataset_dir
        / "data"
        / f"chunk-{job.episode_chunk:03d}"
        / f"episode_{job.episode_index:06d}.parquet"
    )
    dst_videos = {
        video_key: (
            job.dst_dataset_dir
            / "videos"
            / f"chunk-{job.episode_chunk:03d}"
            / video_key
            / f"episode_{job.episode_index:06d}.mp4"
        )
        for video_key in IMAGE_KEY_MAP.values()
    }

    if (
        not job.overwrite
        and dst_parquet.exists()
        and all(path.exists() for path in dst_videos.values())
    ):
        return {
            "episode_index": job.episode_index,
            "length": None,
            "video_shapes": None,
            "skipped": True,
        }

    df = pd.read_parquet(job.src_parquet)
    for src_image_key, video_key in IMAGE_KEY_MAP.items():
        encoded_frames = [item["bytes"] for item in df[src_image_key].tolist()]
        encode_video_ffmpeg(
            encoded_frames=encoded_frames,
            output_path=dst_videos[video_key],
            fps=job.fps,
            codec=job.video_codec,
            crf=job.video_crf,
            threads=job.ffmpeg_threads,
        )

    target_df = build_target_dataframe(df)
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    target_df.to_parquet(dst_parquet, index=False)

    return {
        "episode_index": job.episode_index,
        "length": int(len(df)),
        "video_shapes": None,
        "skipped": False,
    }


def rename_stats_payload(stats: dict[str, Any]) -> dict[str, Any]:
    renamed: dict[str, Any] = {}
    for key, value in stats.items():
        if key in IMAGE_KEY_MAP:
            renamed[IMAGE_KEY_MAP[key]] = value
        elif key in TABULAR_KEY_MAP:
            renamed[TABULAR_KEY_MAP[key]] = value
        else:
            renamed[key] = value
    return renamed


def build_modality_payload() -> dict[str, Any]:
    return {
        "state": {
            "eef_position": {
                "start": 0,
                "end": 3,
                "dtype": "float32",
                "original_key": "observation.state",
            },
            "eef_rotation": {
                "start": 3,
                "end": 6,
                "dtype": "float32",
                "rotation_type": "euler_angles_rpy",
                "original_key": "observation.state",
            },
            "gripper_position": {
                "start": 6,
                "end": 7,
                "dtype": "float32",
                "original_key": "observation.state",
            },
        },
        "action": {
            "delta_eef_position": {
                "start": 0,
                "end": 3,
                "dtype": "float32",
                "original_key": "action",
            },
            "delta_eef_rotation": {
                "start": 3,
                "end": 6,
                "dtype": "float32",
                "rotation_type": "euler_angles_rpy",
                "original_key": "action",
            },
            "gripper_command": {
                "start": 6,
                "end": 7,
                "dtype": "float32",
                "original_key": "action",
            },
        },
        "video": {
            "exterior_image_1": {
                "original_key": "observation.images.exterior_1",
            },
            "exterior_image_2": {
                "original_key": "observation.images.exterior_2",
            },
            "wrist_image": {
                "original_key": "observation.images.wrist",
            },
        },
        "annotation": {
            "language.language_instruction": {
                "original_key": "task_index",
            },
        },
    }


def build_target_info(
    src_info: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    video_shapes: dict[str, tuple[int, int, int]],
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for src_image_key, video_key in IMAGE_KEY_MAP.items():
        height, width, channels = video_shapes[video_key]
        features[video_key] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": src_info["fps"],
                "video.channels": channels,
                "has_audio": False,
            },
        }

    features["observation.state"] = {
        "dtype": "float32",
        "shape": [7],
        "names": STATE_NAMES,
    }
    features["action"] = {
        "dtype": "float32",
        "shape": [7],
        "names": ACTION_NAMES,
    }
    for key, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[key] = {
            "dtype": dtype,
            "shape": [1],
            "names": None,
        }

    return {
        "codebase_version": "v2.1",
        "robot_type": src_info["robot_type"],
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(IMAGE_KEY_MAP),
        "total_chunks": math.ceil(total_episodes / src_info["chunks_size"]),
        "chunks_size": src_info["chunks_size"],
        "fps": src_info["fps"],
        "splits": {
            "train": f"0:{total_episodes}",
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def prepare_meta_records(
    meta_dir: Path,
    max_episodes: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[int]]:
    src_info = read_json(meta_dir / "info.json")
    episodes = read_jsonl(meta_dir / "episodes.jsonl")
    tasks = read_jsonl(meta_dir / "tasks.jsonl")
    episodes_stats = read_jsonl(meta_dir / "episodes_stats.jsonl")

    if max_episodes is not None:
        episodes = episodes[:max_episodes]
        episodes_stats = episodes_stats[:max_episodes]

    selected_episode_indices = {int(item["episode_index"]) for item in episodes}
    episodes_stats = [item for item in episodes_stats if int(item["episode_index"]) in selected_episode_indices]

    used_task_names = {
        task_name
        for item in episodes
        for task_name in item.get("tasks", [])
    }
    tasks = [item for item in tasks if item["task"] in used_task_names]
    used_task_indices = {int(item["task_index"]) for item in tasks}

    renamed_episode_stats = []
    for item in episodes_stats:
        renamed_episode_stats.append(
            {
                "episode_index": item["episode_index"],
                "stats": rename_stats_payload(item["stats"]),
            }
        )

    return src_info, tasks, episodes, renamed_episode_stats, used_task_indices


def convert_subset(config: ConvertConfig, subset: str) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found in PATH")

    src_dataset_dir = config.src_root / SUBSET_TO_SOURCE_DIR[subset]
    if not src_dataset_dir.exists():
        raise FileNotFoundError(f"Missing source dataset dir: {src_dataset_dir}")

    dst_dataset_dir = dataset_dir_for_subset(config.dst_root, subset)
    dst_dataset_dir.mkdir(parents=True, exist_ok=True)

    src_info, tasks, episodes, episode_stats, used_task_indices = prepare_meta_records(
        src_dataset_dir / "meta",
        config.max_episodes,
    )
    episode_count = len(episodes)
    if episode_count == 0:
        raise RuntimeError(f"No episodes selected for subset {subset}")

    jobs = []
    for item in episodes:
        episode_index = int(item["episode_index"])
        jobs.append(
            EpisodeJob(
                src_parquet=src_dataset_dir
                / "data"
                / f"chunk-{episode_index // src_info['chunks_size']:03d}"
                / f"episode_{episode_index:06d}.parquet",
                dst_dataset_dir=dst_dataset_dir,
                episode_index=episode_index,
                episode_chunk=episode_index // src_info["chunks_size"],
                fps=int(src_info["fps"]),
                overwrite=config.overwrite,
                ffmpeg_threads=config.ffmpeg_threads,
                video_crf=config.video_crf,
                video_codec=config.video_codec,
            )
        )

    video_shapes: dict[str, tuple[int, int, int]] | None = None
    lengths_by_episode: dict[int, int] = {}

    with ProcessPoolExecutor(max_workers=config.num_processes) as executor:
        futures = {executor.submit(convert_one_episode, job): job for job in jobs}
        progress = tqdm.tqdm(total=len(futures), desc=f"convert-{subset}", unit="episode")
        for future in as_completed(futures):
            result = future.result()
            progress.update(1)
            if result["length"] is not None:
                lengths_by_episode[result["episode_index"]] = result["length"]
            if result["video_shapes"] is not None and video_shapes is None:
                video_shapes = result["video_shapes"]
        progress.close()

    if video_shapes is None:
        # All episodes were skipped; infer shapes from source metadata.
        video_shapes = {
            video_key: (
                int(src_info["features"][src_image_key]["shape"][0]),
                int(src_info["features"][src_image_key]["shape"][1]),
                int(src_info["features"][src_image_key]["shape"][2]),
            )
            for src_image_key, video_key in IMAGE_KEY_MAP.items()
        }

    total_frames = 0
    normalized_episodes = []
    for item in episodes:
        episode_index = int(item["episode_index"])
        length = lengths_by_episode.get(episode_index, int(item["length"]))
        normalized_episodes.append(
            {
                "episode_index": episode_index,
                "tasks": item["tasks"],
                "length": length,
            }
        )
        total_frames += length

    target_info = build_target_info(
        src_info=src_info,
        total_episodes=len(normalized_episodes),
        total_frames=total_frames,
        total_tasks=len(used_task_indices),
        video_shapes=video_shapes,
    )

    meta_dir = dst_dataset_dir / "meta"
    write_json(meta_dir / "info.json", target_info)
    write_json(meta_dir / "modality.json", build_modality_payload())
    write_jsonl(meta_dir / "tasks.jsonl", tasks)
    write_jsonl(meta_dir / "episodes.jsonl", normalized_episodes)
    write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats)

    print(
        f"[DONE] subset={subset} episodes={len(normalized_episodes)} "
        f"frames={total_frames} out={dst_dataset_dir}"
    )


def main() -> None:
    config = parse_args()
    for subset in config.subsets:
        convert_subset(config, subset)


if __name__ == "__main__":
    main()
