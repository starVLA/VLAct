#!/usr/bin/env python3
"""Plot InternData-A1 split_aloha gripper action distributions."""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/project/vonneumann1/datasets/InternData-A1/sim_updated")
DEFAULT_OUT_DIR = Path("/project/vonneumann1/wcy/code/starVLA-dev/examples/InternA1/preprocess/check/out")
DEFAULT_SAMPLE_RATIO = 0.1
DEFAULT_SEED = 42
ACTION_COLUMNS = [
    "actions.left_gripper.position",
    "actions.right_gripper.position",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--robot-filter", default="split_aloha")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-parquets", type=int, default=0, help="0 means all parquet files")
    parser.add_argument("--sample-ratio", type=float, default=DEFAULT_SAMPLE_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bins", type=int, default=100)
    return parser.parse_args()


def discover_dataset_dirs(root: Path, robot_filter: str):
    dataset_dirs = []
    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        split_root = category_dir / robot_filter
        if not split_root.is_dir():
            continue
        for task_dir in sorted(split_root.iterdir()):
            if not task_dir.is_dir():
                continue
            for sub_dir in sorted(task_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue
                if (sub_dir / "meta" / "info.json").is_file():
                    dataset_dirs.append(sub_dir)
                    continue
                for obj_dir in sorted(sub_dir.iterdir()):
                    if (obj_dir / "meta" / "info.json").is_file():
                        dataset_dirs.append(obj_dir)
    return sorted(set(dataset_dirs))


def is_compatible_dataset(dataset_dir: Path):
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    features = info.get("features", {})
    return all(col in features for col in ACTION_COLUMNS)


def collect_parquet_paths(dataset_dirs, max_parquets: int):
    parquet_paths = []
    skipped_datasets = []
    for dataset_dir in dataset_dirs:
        if not is_compatible_dataset(dataset_dir):
            skipped_datasets.append(str(dataset_dir))
            continue
        data_dir = dataset_dir / "data"
        if not data_dir.is_dir():
            skipped_datasets.append(str(dataset_dir))
            continue
        for chunk_dir in sorted(data_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            for parquet_path in sorted(chunk_dir.glob("*.parquet")):
                parquet_paths.append(parquet_path)
                if max_parquets > 0 and len(parquet_paths) >= max_parquets:
                    return parquet_paths, skipped_datasets
    return parquet_paths, skipped_datasets


def sample_parquet_paths(parquet_paths, sample_ratio: float, seed: int):
    if not parquet_paths:
        return parquet_paths
    if sample_ratio <= 0 or sample_ratio >= 1:
        return parquet_paths
    sample_count = max(1, int(round(len(parquet_paths) * sample_ratio)))
    rng = random.Random(seed)
    selected = rng.sample(parquet_paths, sample_count)
    return sorted(selected)


def iter_gripper_arrays(parquet_paths):
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path, columns=ACTION_COLUMNS)
        left = df[ACTION_COLUMNS[0]].to_numpy(dtype=np.float32)
        right = df[ACTION_COLUMNS[1]].to_numpy(dtype=np.float32)
        yield parquet_path, left, right


def print_progress(prefix: str, idx: int, total: int):
    width = 28
    done = int(width * idx / max(total, 1))
    bar = "#" * done + "-" * (width - done)
    sys.stdout.write(f"\r{prefix} [{bar}] {idx}/{total}")
    sys.stdout.flush()
    if idx >= total:
        sys.stdout.write("\n")


def compute_range(parquet_paths):
    global_min = math.inf
    global_max = -math.inf
    total_rows = 0
    total = len(parquet_paths)
    for idx, (_, left, right) in enumerate(iter_gripper_arrays(parquet_paths), start=1):
        global_min = min(global_min, float(left.min()), float(right.min()))
        global_max = max(global_max, float(left.max()), float(right.max()))
        total_rows += len(left)
        print_progress("range pass", idx, total)
    if not math.isfinite(global_min) or not math.isfinite(global_max):
        raise ValueError("No gripper values found")
    if global_min == global_max:
        global_min -= 1e-3
        global_max += 1e-3
    return global_min, global_max, total_rows


def build_summary(parquet_paths, bins: int):
    value_min, value_max, total_rows = compute_range(parquet_paths)
    bin_edges = np.linspace(value_min, value_max, bins + 1, dtype=np.float64)
    hist_left = np.zeros(bins, dtype=np.int64)
    hist_right = np.zeros(bins, dtype=np.int64)
    left_sum = right_sum = 0.0
    left_sq_sum = right_sq_sum = 0.0
    left_min = right_min = math.inf
    left_max = right_max = -math.inf
    left_gt_005 = left_gt_049 = 0
    right_gt_005 = right_gt_049 = 0

    total = len(parquet_paths)
    for idx, (_, left, right) in enumerate(iter_gripper_arrays(parquet_paths), start=1):
        hist_left += np.histogram(left, bins=bin_edges)[0]
        hist_right += np.histogram(right, bins=bin_edges)[0]
        left_sum += float(left.sum())
        right_sum += float(right.sum())
        left_sq_sum += float(np.square(left).sum())
        right_sq_sum += float(np.square(right).sum())
        left_min = min(left_min, float(left.min()))
        right_min = min(right_min, float(right.min()))
        left_max = max(left_max, float(left.max()))
        right_max = max(right_max, float(right.max()))
        left_gt_005 += int((left > 0.05).sum())
        left_gt_049 += int((left > 0.49).sum())
        right_gt_005 += int((right > 0.05).sum())
        right_gt_049 += int((right > 0.49).sum())
        print_progress("hist  pass", idx, total)

    def stats_dict(name, hist, val_min, val_max, val_sum, val_sq_sum, gt_005, gt_049):
        mean = val_sum / total_rows
        var = max(val_sq_sum / total_rows - mean * mean, 0.0)
        return {
            "name": name,
            "count": total_rows,
            "min": val_min,
            "max": val_max,
            "mean": mean,
            "std": math.sqrt(var),
            "gt_0_05_ratio": gt_005 / total_rows,
            "gt_0_49_ratio": gt_049 / total_rows,
            "hist_counts": hist.tolist(),
        }

    return {
        "value_range": [value_min, value_max],
        "bin_edges": bin_edges.tolist(),
        "left_gripper": stats_dict(
            "actions.left_gripper.position",
            hist_left,
            left_min,
            left_max,
            left_sum,
            left_sq_sum,
            left_gt_005,
            left_gt_049,
        ),
        "right_gripper": stats_dict(
            "actions.right_gripper.position",
            hist_right,
            right_min,
            right_max,
            right_sum,
            right_sq_sum,
            right_gt_005,
            right_gt_049,
        ),
    }


def sx(value, vmin, vmax, left, width):
    return left + (value - vmin) / max(vmax - vmin, 1e-9) * width


def sy(value, vmax, top, height):
    return top + height - value / max(vmax, 1e-9) * height


def append_hist_panel(parts, left, top, width, height, title, counts, bin_edges, color):
    vmax = max(max(counts), 1)
    parts.append(f'<text x="{left}" y="{top - 18}" class="label">{title}</text>')
    parts.append(f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" class="axis"/>')
    for idx, count in enumerate(counts):
        x1 = sx(bin_edges[idx], bin_edges[0], bin_edges[-1], left, width)
        x2 = sx(bin_edges[idx + 1], bin_edges[0], bin_edges[-1], left, width)
        y = sy(count, vmax, top, height)
        parts.append(
            f'<rect x="{x1:.2f}" y="{y:.2f}" width="{max(x2 - x1 - 1, 0.5):.2f}" '
            f'height="{top + height - y:.2f}" fill="{color}" fill-opacity="0.75"/>'
        )
    parts.append(f'<text x="{left}" y="{top + height + 24}" class="tick">{bin_edges[0]:.3f}</text>')
    parts.append(f'<text x="{left + width - 60}" y="{top + height + 24}" class="tick">{bin_edges[-1]:.3f}</text>')
    parts.append(f'<text x="{left - 8}" y="{top + 10}" class="tick" text-anchor="end">{vmax}</text>')


def append_stats_text(parts, x, y, stats, color):
    rows = [
        stats["name"],
        f"count={stats['count']}",
        f"min={stats['min']:.6f}",
        f"max={stats['max']:.6f}",
        f"mean={stats['mean']:.6f}",
        f"std={stats['std']:.6f}",
        f">0.05={stats['gt_0_05_ratio']:.4f}",
        f">0.49={stats['gt_0_49_ratio']:.4f}",
    ]
    for idx, row in enumerate(rows):
        klass = "label" if idx == 0 else "tick"
        parts.append(f'<text x="{x}" y="{y + idx * 18}" class="{klass}" fill="{color}">{row}</text>')


def render_svg(summary, out_path: Path):
    width = 1400
    height = 860
    margin = 70
    panel_gap = 60
    panel_w = (width - margin * 2 - panel_gap) / 2
    hist_h = 320
    text_y = margin + hist_h + 60
    colors = {"left": "#1f77b4", "right": "#d62728"}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:monospace} .title{font-size:24px;font-weight:bold} .label{font-size:16px} .tick{font-size:13px;fill:#555} .axis{stroke:#333;stroke-width:1.5}</style>',
        f'<text x="{margin}" y="40" class="title">InternA1 split_aloha gripper action distribution</text>',
    ]
    append_hist_panel(
        parts,
        margin,
        margin,
        panel_w,
        hist_h,
        "Left gripper histogram",
        summary["left_gripper"]["hist_counts"],
        summary["bin_edges"],
        colors["left"],
    )
    append_hist_panel(
        parts,
        margin + panel_w + panel_gap,
        margin,
        panel_w,
        hist_h,
        "Right gripper histogram",
        summary["right_gripper"]["hist_counts"],
        summary["bin_edges"],
        colors["right"],
    )
    append_stats_text(parts, margin, text_y, summary["left_gripper"], colors["left"])
    append_stats_text(parts, margin + panel_w + panel_gap, text_y, summary["right_gripper"], colors["right"])
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = discover_dataset_dirs(args.root, args.robot_filter)
    parquet_paths, skipped_datasets = collect_parquet_paths(dataset_dirs, args.max_parquets)
    parquet_paths = sample_parquet_paths(parquet_paths, args.sample_ratio, args.seed)
    if not parquet_paths:
        raise ValueError("No compatible parquet files found")

    summary = build_summary(parquet_paths, args.bins)
    summary.update(
        {
            "root": str(args.root),
            "robot_filter": args.robot_filter,
            "num_datasets": len(dataset_dirs),
            "num_skipped_datasets": len(skipped_datasets),
            "num_parquets": len(parquet_paths),
            "sample_ratio": args.sample_ratio,
            "seed": args.seed,
        }
    )

    json_path = args.out_dir / "gripper_action_distribution.json"
    svg_path = args.out_dir / "gripper_action_distribution.svg"
    json_path.write_text(json.dumps(summary, indent=2))
    render_svg(summary, svg_path)

    print(f"datasets={len(dataset_dirs)}")
    print(f"skipped_datasets={len(skipped_datasets)}")
    print(f"parquets={len(parquet_paths)}")
    print(f"saved_json={json_path}")
    print(f"saved_svg={svg_path}")


if __name__ == "__main__":
    main()
