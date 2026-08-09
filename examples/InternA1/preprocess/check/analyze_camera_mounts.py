#!/usr/bin/env python3
"""
Sample split_aloha camera parameters and save simple SVG plots.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/project/vonneumann1/wcy/dataset/VLM-VLA/InternData-A1/sim_updated")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "camera_mount_plots"
DEFAULT_DATASET_TAG = "split_aloha"

CAMERA_SPECS = {
    "head": {
        "intrinsics": "head_camera_intrinsics",
        "extrinsics": "head_camera_to_robot_extrinsics",
        "color": "#2563eb",
    },
    "hand_left": {
        "intrinsics": "hand_left_camera_intrinsics",
        "extrinsics": "hand_left_camera_to_robot_extrinsics",
        "color": "#16a34a",
    },
    "hand_right": {
        "intrinsics": "hand_right_camera_intrinsics",
        "extrinsics": "hand_right_camera_to_robot_extrinsics",
        "color": "#dc2626",
    },
}

METRICS = ["fx", "fy", "cx", "cy", "tx", "ty", "tz", "roll_deg", "pitch_deg", "yaw_deg"]


def discover_datasets(root: Path, dataset_tag: str):
    datasets = []
    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        dataset_root = category_dir / dataset_tag
        if not dataset_root.is_dir():
            continue
        for task_dir in sorted(dataset_root.iterdir()):
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


def normalize_quaternion(qw: float, qx: float, qy: float, qz: float):
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0:
        return 1.0, 0.0, 0.0, 0.0
    return qw / norm, qx / norm, qy / norm, qz / norm


def quaternion_to_euler_deg(qw: float, qx: float, qy: float, qz: float):
    qw, qx, qy, qz = normalize_quaternion(qw, qx, qy, qz)

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


def choose_frame_indices(num_frames: int, frames_per_episode: int):
    if num_frames <= frames_per_episode:
        return list(range(num_frames))
    return sorted(set(np.linspace(0, num_frames - 1, frames_per_episode, dtype=int).tolist()))


def collect_samples(root: Path, dataset_tag: str, episode_stride: int, frames_per_episode: int, limit_datasets: int | None):
    datasets = discover_datasets(root, dataset_tag)
    if limit_datasets:
        datasets = datasets[:limit_datasets]

    rows = []
    intrinsics_counter = {camera_name: Counter() for camera_name in CAMERA_SPECS}
    sampled_episodes = 0

    required_columns = ["frame_index"]
    for spec in CAMERA_SPECS.values():
        required_columns.append(spec["intrinsics"])
        required_columns.append(spec["extrinsics"])

    print(f"Found {len(datasets)} datasets under tag={dataset_tag}")

    for idx, dataset_dir in enumerate(datasets, start=1):
        print(f"[{idx}/{len(datasets)}] {dataset_dir.relative_to(root)}")
        episode_files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))[::max(1, episode_stride)]

        for episode_path in episode_files:
            try:
                df = pd.read_parquet(episode_path, columns=required_columns)
            except Exception as exc:
                print(f"[warning] skip {episode_path}: {exc}")
                continue

            frame_indices = choose_frame_indices(len(df), frames_per_episode)
            if not frame_indices:
                continue

            sampled_episodes += 1
            df = df.iloc[frame_indices].reset_index(drop=True)

            for camera_name, spec in CAMERA_SPECS.items():
                intr_values = df[spec["intrinsics"]].tolist()
                extr_values = df[spec["extrinsics"]].tolist()
                frame_ids = df["frame_index"].tolist()

                for frame_id, intr, extr in zip(frame_ids, intr_values, extr_values):
                    if len(intr) != 4 or len(extr) != 7:
                        continue

                    fx, fy, cx, cy = [float(v) for v in intr]
                    tx, ty, tz, qw, qx, qy, qz = [float(v) for v in extr]
                    roll_deg, pitch_deg, yaw_deg = quaternion_to_euler_deg(qw, qx, qy, qz)

                    rows.append(
                        {
                            "dataset": str(dataset_dir.relative_to(root)),
                            "episode_file": episode_path.name,
                            "frame_index": int(frame_id),
                            "camera": camera_name,
                            "fx": fx,
                            "fy": fy,
                            "cx": cx,
                            "cy": cy,
                            "tx": tx,
                            "ty": ty,
                            "tz": tz,
                            "roll_deg": roll_deg,
                            "pitch_deg": pitch_deg,
                            "yaw_deg": yaw_deg,
                        }
                    )
                    intrinsics_counter[camera_name][tuple(round(float(v), 6) for v in intr)] += 1

    return pd.DataFrame(rows), intrinsics_counter, len(datasets), sampled_episodes


def stat_dict(series: pd.Series):
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def build_summary(df: pd.DataFrame, intrinsics_counter, dataset_count: int, sampled_episodes: int, episode_stride: int, frames_per_episode: int):
    summary = {
        "datasets_processed": int(dataset_count),
        "episodes_sampled": int(sampled_episodes),
        "records": int(len(df)),
        "sample_policy": {
            "episode_stride": int(episode_stride),
            "frames_per_episode": int(frames_per_episode),
            "meaning": "Use every Nth episode and uniformly sample a few frames from each selected episode.",
        },
        "cameras": {},
    }

    for camera_name in CAMERA_SPECS:
        camera_df = df[df["camera"] == camera_name]
        summary["cameras"][camera_name] = {
            "records": int(len(camera_df)),
            "stats": {metric: stat_dict(camera_df[metric]) for metric in METRICS},
            "top_intrinsics": [
                {"values": list(values), "count": int(count)}
                for values, count in intrinsics_counter[camera_name].most_common(5)
            ],
        }

    return summary


def svg_header(width: int, height: int):
    return f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}'>"


def svg_footer():
    return "</svg>"


def draw_hist_panel(values, title: str, color: str, x0: int, y0: int, width: int, height: int, bins: int = 24):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [f"<text x='{x0 + 12}' y='{y0 + 24}' font-size='14' fill='#334155'>{title}: no data</text>"]

    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5

    counts, _ = np.histogram(values, bins=bins, range=(vmin, vmax))
    max_count = max(int(counts.max()), 1)
    left = x0 + 42
    right = x0 + width - 16
    top = y0 + 24
    bottom = y0 + height - 26
    plot_width = right - left
    plot_height = bottom - top
    bar_width = plot_width / len(counts)

    parts = [
        f"<rect x='{x0}' y='{y0}' width='{width}' height='{height}' fill='white' stroke='#cbd5e1'/>",
        f"<text x='{left}' y='{y0 + 16}' font-size='13' fill='#0f172a'>{title}</text>",
        f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}' stroke='#94a3b8'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{bottom}' stroke='#94a3b8'/>",
        f"<text x='{left}' y='{bottom + 16}' font-size='10' fill='#475569'>{vmin:.3f}</text>",
        f"<text x='{right}' y='{bottom + 16}' text-anchor='end' font-size='10' fill='#475569'>{vmax:.3f}</text>",
        f"<text x='{left - 4}' y='{top + 8}' text-anchor='end' font-size='10' fill='#475569'>{max_count}</text>",
    ]

    for idx, count in enumerate(counts):
        bar_h = plot_height * count / max_count
        x = left + idx * bar_width + 1
        y = bottom - bar_h
        parts.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{max(bar_width - 2, 1):.2f}' height='{bar_h:.2f}' fill='{color}' fill-opacity='0.85'/>"
        )

    return parts


def draw_scatter_panel(xs, ys, title: str, color: str, x_label: str, y_label: str, x0: int, y0: int, width: int, height: int):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) == 0:
        return [f"<text x='{x0 + 12}' y='{y0 + 24}' font-size='14' fill='#334155'>{title}: no data</text>"]

    xmin = float(xs.min())
    xmax = float(xs.max())
    ymin = float(ys.min())
    ymax = float(ys.max())
    if math.isclose(xmin, xmax):
        xmin -= 0.5
        xmax += 0.5
    if math.isclose(ymin, ymax):
        ymin -= 0.5
        ymax += 0.5

    left = x0 + 46
    right = x0 + width - 16
    top = y0 + 24
    bottom = y0 + height - 28
    plot_width = right - left
    plot_height = bottom - top

    parts = [
        f"<rect x='{x0}' y='{y0}' width='{width}' height='{height}' fill='white' stroke='#cbd5e1'/>",
        f"<text x='{left}' y='{y0 + 16}' font-size='13' fill='#0f172a'>{title}</text>",
        f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}' stroke='#94a3b8'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{bottom}' stroke='#94a3b8'/>",
        f"<text x='{left}' y='{bottom + 16}' font-size='10' fill='#475569'>{xmin:.3f}</text>",
        f"<text x='{right}' y='{bottom + 16}' text-anchor='end' font-size='10' fill='#475569'>{xmax:.3f}</text>",
        f"<text x='{left - 4}' y='{top + 8}' text-anchor='end' font-size='10' fill='#475569'>{ymax:.3f}</text>",
        f"<text x='{left - 4}' y='{bottom}' text-anchor='end' font-size='10' fill='#475569'>{ymin:.3f}</text>",
        f"<text x='{(left + right) / 2:.1f}' y='{y0 + height - 6}' text-anchor='middle' font-size='11' fill='#475569'>{x_label}</text>",
        f"<text x='{x0 + 12}' y='{(top + bottom) / 2:.1f}' transform='rotate(-90 {x0 + 12} {(top + bottom) / 2:.1f})' text-anchor='middle' font-size='11' fill='#475569'>{y_label}</text>",
    ]

    for x_value, y_value in zip(xs, ys):
        px = left + (x_value - xmin) / (xmax - xmin) * plot_width
        py = bottom - (y_value - ymin) / (ymax - ymin) * plot_height
        parts.append(f"<circle cx='{px:.2f}' cy='{py:.2f}' r='1.7' fill='{color}' fill-opacity='0.28'/>")

    return parts


def save_hist_svg(values_map, output_path: Path, title_prefix: str, color: str):
    width = 1080
    height = 280
    panel_width = 340
    panel_height = 240
    parts = [svg_header(width, height), f"<rect width='{width}' height='{height}' fill='#f8fafc'/>"]

    titles = list(values_map.keys())
    for idx, title in enumerate(titles):
        x0 = 12 + idx * 352
        y0 = 20
        parts.extend(draw_hist_panel(values_map[title], f"{title_prefix} {title}", color, x0, y0, panel_width, panel_height))

    parts.append(svg_footer())
    output_path.write_text("".join(parts))


def save_scatter_svg(df: pd.DataFrame, output_path: Path, title_prefix: str, color: str):
    width = 1080
    height = 320
    panel_width = 340
    panel_height = 270
    parts = [svg_header(width, height), f"<rect width='{width}' height='{height}' fill='#f8fafc'/>"]

    pairs = [("tx", "ty"), ("tx", "tz"), ("ty", "tz")]
    for idx, (x_key, y_key) in enumerate(pairs):
        x0 = 12 + idx * 352
        y0 = 20
        parts.extend(
            draw_scatter_panel(
                xs=df[x_key].to_numpy(),
                ys=df[y_key].to_numpy(),
                title=f"{title_prefix} {x_key} vs {y_key}",
                color=color,
                x_label=x_key,
                y_label=y_key,
                x0=x0,
                y0=y0,
                width=panel_width,
                height=panel_height,
            )
        )

    parts.append(svg_footer())
    output_path.write_text("".join(parts))


def save_plots(df: pd.DataFrame, output_dir: Path):
    for camera_name, spec in CAMERA_SPECS.items():
        camera_df = df[df["camera"] == camera_name].reset_index(drop=True)
        if camera_df.empty:
            continue

        save_scatter_svg(
            camera_df,
            output_dir / f"{camera_name}_translation_scatter.svg",
            title_prefix=camera_name,
            color=spec["color"],
        )
        save_hist_svg(
            {
                "tx": camera_df["tx"].to_numpy(),
                "ty": camera_df["ty"].to_numpy(),
                "tz": camera_df["tz"].to_numpy(),
            },
            output_dir / f"{camera_name}_translation_hist.svg",
            title_prefix=camera_name,
            color=spec["color"],
        )
        save_hist_svg(
            {
                "roll": camera_df["roll_deg"].to_numpy(),
                "pitch": camera_df["pitch_deg"].to_numpy(),
                "yaw": camera_df["yaw_deg"].to_numpy(),
            },
            output_dir / f"{camera_name}_rotation_hist.svg",
            title_prefix=camera_name,
            color=spec["color"],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset-tag", default=DEFAULT_DATASET_TAG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-stride", type=int, default=10)
    parser.add_argument("--frames-per-episode", type=int, default=10)
    parser.add_argument("--limit-datasets", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, intrinsics_counter, dataset_count, sampled_episodes = collect_samples(
        root=args.root,
        dataset_tag=args.dataset_tag,
        episode_stride=args.episode_stride,
        frames_per_episode=args.frames_per_episode,
        limit_datasets=args.limit_datasets,
    )
    if df.empty:
        raise RuntimeError("No camera samples collected.")

    summary = build_summary(
        df=df,
        intrinsics_counter=intrinsics_counter,
        dataset_count=dataset_count,
        sampled_episodes=sampled_episodes,
        episode_stride=args.episode_stride,
        frames_per_episode=args.frames_per_episode,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    save_plots(df, args.output_dir)

    print(f"Saved summary to {args.output_dir / 'summary.json'}")
    for camera_name in CAMERA_SPECS:
        print(args.output_dir / f"{camera_name}_translation_scatter.svg")
        print(args.output_dir / f"{camera_name}_translation_hist.svg")
        print(args.output_dir / f"{camera_name}_rotation_hist.svg")


if __name__ == "__main__":
    main()
