#!/usr/bin/env python3
"""Render episode gripper action values into an SVG line plot."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import pandas as pd


MAX_POINTS = 320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", type=Path, required=True, help="Path to episode video.")
    parser.add_argument("--output-path", type=Path, required=True, help="Where to write the SVG.")
    return parser.parse_args()


def build_points(values: list[float], sx, sy) -> str:
    n = len(values)
    if n <= MAX_POINTS:
        indices = range(n)
    else:
        step = (n - 1) / (MAX_POINTS - 1)
        indices = [round(i * step) for i in range(MAX_POINTS)]
    return " ".join(f"{sx(i):.2f},{sy(values[i]):.2f}" for i in indices)


def resolve_dataset_paths(video_path: Path) -> tuple[Path, Path]:
    parts = video_path.parts
    try:
        videos_idx = parts.index("videos")
    except ValueError as exc:
        raise ValueError(f"Video path does not follow expected dataset layout: {video_path}") from exc

    dataset_dir = Path(*parts[:videos_idx])
    chunk_name = video_path.parents[1].name
    parquet_path = dataset_dir / "data" / chunk_name / f"{video_path.stem}.parquet"
    return dataset_dir, parquet_path


def load_task(dataset_dir: Path, episode_index: int) -> str:
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            episode = json.loads(line)
            if int(episode["episode_index"]) == episode_index:
                tasks = episode.get("tasks") or []
                return tasks[0] if tasks else "unknown"
    return "unknown"


def to_float(value) -> float:
    if isinstance(value, (list, tuple)):
        return float(value[0])
    try:
        return float(value[0])
    except (TypeError, IndexError, KeyError):
        return float(value)


def normalize_action_names(names) -> list[str]:
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        return [str(x) for x in names[0]]
    return [str(x) for x in names]


def load_gripper_values(dataset_dir: Path, parquet_path: Path) -> tuple[list[float], list[float], int]:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(parquet_path)

    if "action" in df.columns:
        action_names = normalize_action_names(info["features"]["action"]["names"])
        if "left_gripper_open" in action_names and "right_gripper_open" in action_names:
            left_name = "left_gripper_open"
            right_name = "right_gripper_open"
        elif "left_gripper" in action_names and "right_gripper" in action_names:
            left_name = "left_gripper"
            right_name = "right_gripper"
        else:
            raise ValueError(
                f"Could not find left/right gripper names in action metadata: {action_names}"
            )
        left_idx = action_names.index(left_name)
        right_idx = action_names.index(right_name)
        left_vals = [float(step[left_idx]) for step in df["action"]]
        right_vals = [float(step[right_idx]) for step in df["action"]]
    else:
        left_col = "actions.left_gripper.position"
        right_col = "actions.right_gripper.position"
        if left_col not in df.columns or right_col not in df.columns:
            raise ValueError(
                "Could not find gripper action columns. Expected either 'action' or "
                "'actions.left_gripper.position'/'actions.right_gripper.position'."
            )
        left_vals = [to_float(step) for step in df[left_col]]
        right_vals = [to_float(step) for step in df[right_col]]

    episode_index = int(df["episode_index"].iloc[0])
    return left_vals, right_vals, episode_index


def render_svg(
    *,
    output_path: Path,
    title: str,
    dataset_dir: Path,
    video_path: Path,
    parquet_path: Path,
    task: str,
    left_vals: list[float],
    right_vals: list[float],
) -> None:
    traj_len = len(left_vals)
    ymin = min(min(left_vals), min(right_vals))
    ymax = max(max(left_vals), max(right_vals))
    if ymin == ymax:
        ymin -= 1e-6
        ymax += 1e-6

    width, height = 1400, 820
    margin = 80
    plot_w = width - 2 * margin
    plot_h = 420
    plot_top = 120
    plot_bottom = plot_top + plot_h

    def sx(i: int) -> float:
        return margin + (i / max(traj_len - 1, 1)) * plot_w

    def sy(v: float) -> float:
        return plot_bottom - ((v - ymin) / (ymax - ymin)) * plot_h

    left_points = build_points(left_vals, sx, sy)
    right_points = build_points(right_vals, sx, sy)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:monospace}.title{font-size:24px;font-weight:bold}.label{font-size:16px}.tick{font-size:13px;fill:#555}.axis{stroke:#333;stroke-width:1.5}</style>',
        f'<text x="{margin}" y="40" class="title">{escape(title)}</text>',
        f'<text x="{margin}" y="68" class="tick">dataset={escape(dataset_dir.name)} task={escape(task)}</text>',
        f'<text x="{margin}" y="88" class="tick">traj_len={traj_len} parquet={escape(parquet_path.name)} video_key={escape(video_path.parent.name)}</text>',
        f'<line x1="{margin}" y1="{plot_bottom}" x2="{margin + plot_w}" y2="{plot_bottom}" class="axis"/>',
        f'<line x1="{margin}" y1="{plot_top}" x2="{margin}" y2="{plot_bottom}" class="axis"/>',
        f'<text x="{margin - 10}" y="{plot_top + 5}" text-anchor="end" class="tick">{ymax:.3f}</text>',
        f'<text x="{margin - 10}" y="{plot_bottom}" text-anchor="end" class="tick">{ymin:.3f}</text>',
        f'<text x="{margin}" y="{plot_bottom + 24}" class="tick">0</text>',
        f'<text x="{margin + plot_w - 40}" y="{plot_bottom + 24}" class="tick">{traj_len - 1}</text>',
        f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{left_points}"/>',
        f'<polyline fill="none" stroke="#d62728" stroke-width="2" points="{right_points}"/>',
        f'<text x="{margin}" y="590" class="label" fill="#1f77b4">left_gripper_open</text>',
        f'<text x="{margin}" y="612" class="tick" fill="#1f77b4">min={min(left_vals):.6f} max={max(left_vals):.6f}</text>',
        f'<text x="{margin + 300}" y="590" class="label" fill="#d62728">right_gripper_open</text>',
        f'<text x="{margin + 300}" y="612" class="tick" fill="#d62728">min={min(right_vals):.6f} max={max(right_vals):.6f}</text>',
        f'<text x="{margin}" y="650" class="tick">video_name={escape(video_path.name)}</text>',
        f'<text x="{margin}" y="672" class="tick">video_dir={escape(str(video_path.parent))}</text>',
        f'<text x="{margin}" y="694" class="tick">source_parquet={escape(str(parquet_path))}</text>',
        "</svg>",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    video_path = args.video_path.resolve()
    output_path = args.output_path.resolve()

    dataset_dir, parquet_path = resolve_dataset_paths(video_path)
    left_vals, right_vals, episode_index = load_gripper_values(dataset_dir, parquet_path)
    task = load_task(dataset_dir, episode_index)

    title = f"Episode gripper action plot: {dataset_dir.name}"
    render_svg(
        output_path=output_path,
        title=title,
        dataset_dir=dataset_dir,
        video_path=video_path,
        parquet_path=parquet_path,
        task=task,
        left_vals=left_vals,
        right_vals=right_vals,
    )
    print(output_path)


if __name__ == "__main__":
    main()
