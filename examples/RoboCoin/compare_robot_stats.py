#!/usr/bin/env python3
"""Compare 14D joint+gripper ranges across datasets."""

import argparse
import json
from pathlib import Path

import numpy as np


DATA_ROOT = Path("/project/vonneumann1/datasets/RoboCOIN-cy")
DEFAULT_OUT_DIR = Path("/project/vonneumann1/wcy/code/starVLA-dev/examples/RoboCoin/out")
ROBOCOIN_PREFIXES = ("Cobot_Magic_", "Split_aloha_")
PI_LIMIT = 2 * np.pi
EXTERNAL_STATS = {
    "interna1": Path(
        "/project/vonneumann1/wcy/code/starVLA-dev/results/Checkpoints/"
        "0315_interna1_split_aloha_qwen3OFT/dataset_statistics.json"
    ),
    "robotwin": Path(
        "/project/vonneumann1/wcy/code/starVLA-dev/results/Checkpoints/"
        "0315_robotwin_qwen3OFT_50k/dataset_statistics.json"
    ),
}
ACTION_KEYS = ["left_joints", "right_joints", "left_gripper", "right_gripper"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode", choices=["abs", "delta", "rel"], default="abs")
    return parser.parse_args()


def find_task_dirs(data_root: Path, prefix: str):
    return sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith(prefix))


def load_joint_gripper_stats_from_robocoin(dataset_dir: Path, mode: str):
    with open(dataset_dir / "meta" / "modality.json") as f:
        modality = json.load(f)
    with open(dataset_dir / "meta" / "stats_gr00t.json") as f:
        stats = json.load(f)[mode]

    action_stats = stats["action"]
    mins = []
    maxs = []
    for key in ACTION_KEYS:
        meta = modality["action"][key]
        start, end = meta["start"], meta["end"]
        mins.extend(action_stats["min"][start:end])
        maxs.extend(action_stats["max"][start:end])
    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def summarize_robocoin_group(task_dirs, mode: str, abs_limit=None):
    agg_min = None
    agg_max = None
    valid_counts = None
    for dataset_dir in task_dirs:
        mins, maxs = load_joint_gripper_stats_from_robocoin(dataset_dir, mode)
        valid_mask = np.ones_like(mins, dtype=bool)
        if abs_limit is not None:
            valid_mask = (mins >= -abs_limit) & (maxs <= abs_limit)

        if agg_min is None:
            agg_min = np.full_like(mins, np.inf)
            agg_max = np.full_like(maxs, -np.inf)
            valid_counts = np.zeros_like(mins, dtype=np.int32)

        valid_counts += valid_mask.astype(np.int32)
        agg_min = np.minimum(agg_min, np.where(valid_mask, mins, np.inf))
        agg_max = np.maximum(agg_max, np.where(valid_mask, maxs, -np.inf))

    if agg_min is None:
        raise ValueError("No RoboCOIN tasks found")

    missing_dims = np.flatnonzero(valid_counts == 0)
    if len(missing_dims) > 0:
        dims_str = ", ".join(str(int(i)) for i in missing_dims)
        raise ValueError(f"No valid RoboCOIN stats left after filtering for dims: {dims_str}")

    return {
        "task_count": len(task_dirs),
        "dim": int(len(agg_min)),
        "min": agg_min,
        "max": agg_max,
    }


def load_external_stats(path: Path):
    with open(path) as f:
        payload = json.load(f)["new_embodiment"]["action"]
    mins = np.asarray(payload["min"], dtype=np.float32)
    maxs = np.asarray(payload["max"], dtype=np.float32)
    return {
        "task_count": 0,
        "dim": len(mins),
        "min": mins,
        "max": maxs,
    }


def jsonable(summary):
    return {
        "task_count": summary["task_count"],
        "dim": summary["dim"],
        "min": summary["min"].tolist(),
        "max": summary["max"].tolist(),
    }


def _sx(value, vmin, vmax, left, width):
    span = max(vmax - vmin, 1e-6)
    return left + (value - vmin) / span * width


def _sy(value, vmin, vmax, top, height):
    span = max(vmax - vmin, 1e-6)
    return top + height - (value - vmin) / span * height


def append_pi_guides_for_x(parts, left, top, panel_w, panel_h, xmin, xmax):
    for value, label in (
        (-2 * np.pi, "-2pi"),
        (-np.pi, "-pi"),
        (0.0, "0"),
        (np.pi, "+pi"),
        (2 * np.pi, "+2pi"),
    ):
        if xmin <= value <= xmax:
            x = _sx(value, xmin, xmax, left, panel_w)
            parts.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + panel_h}" '
                'stroke="#888" stroke-width="1.5" stroke-dasharray="6 6" stroke-opacity="0.9"/>'
            )
            parts.append(
                f'<text x="{x + 4:.1f}" y="{top + 18}" class="tick" fill="#666">{label}</text>'
            )


def append_pi_guides_for_y(parts, left, top, panel_w, panel_h, ymin, ymax):
    for value, label in (
        (-2 * np.pi, "-2pi"),
        (-np.pi, "-pi"),
        (0.0, "0"),
        (np.pi, "+pi"),
        (2 * np.pi, "+2pi"),
    ):
        if ymin <= value <= ymax:
            y = _sy(value, ymin, ymax, top, panel_h)
            parts.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_w}" y2="{y:.1f}" '
                'stroke="#888" stroke-width="1.5" stroke-dasharray="6 6" stroke-opacity="0.9"/>'
            )
            parts.append(
                f'<text x="{left + panel_w - 44}" y="{y - 6:.1f}" class="tick" fill="#666">{label}</text>'
            )


def append_flat_range_panel(parts, left, top, panel_w, panel_h, entries, colors):
    xmin = min(float(np.min(stats["min"])) for _, stats in entries)
    xmax = max(float(np.max(stats["max"])) for _, stats in entries)
    row_gap = panel_h / max(len(entries), 1)
    parts.extend(
        [
            f'<text x="{left}" y="{top - 18}" class="label">Flattened 14D action min/max range</text>',
            f'<line x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" y2="{top + panel_h}" class="axis"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_h}" class="axis"/>',
            f'<text x="{left}" y="{top + panel_h + 28}" class="tick">{xmin:.3f}</text>',
            f'<text x="{left + panel_w - 70}" y="{top + panel_h + 28}" class="tick">{xmax:.3f}</text>',
        ]
    )
    append_pi_guides_for_x(parts, left, top, panel_w, panel_h, xmin, xmax)
    for i, (label, stats) in enumerate(entries):
        color = colors[label]
        y = top + row_gap * (i + 0.5)
        lo = float(np.min(stats["min"]))
        hi = float(np.max(stats["max"]))
        x1 = _sx(lo, xmin, xmax, left, panel_w)
        x2 = _sx(hi, xmin, xmax, left, panel_w)
        parts.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="10" stroke-opacity="0.85"/>')
        parts.append(f'<circle cx="{x1:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        parts.append(f'<circle cx="{x2:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{left + 12}" y="{y - 12:.1f}" class="label" fill="{color}">{label}</text>')
        parts.append(f'<text x="{x1:.1f}" y="{y + 22:.1f}" class="tick" fill="{color}">{lo:.2f}</text>')
        parts.append(f'<text x="{x2 - 38:.1f}" y="{y + 22:.1f}" class="tick" fill="{color}">{hi:.2f}</text>')


def append_per_dim_panel(parts, left, top, panel_w, panel_h, entries, colors, dims=None, label_text=None):
    dims = np.arange(14) if dims is None else np.asarray(dims)
    ymax = max(float(np.max(stats["max"][dims])) for _, stats in entries)
    ymin = min(float(np.min(stats["min"][dims])) for _, stats in entries)
    offsets = np.linspace(-24, 24, len(entries))
    label_text = label_text or f"Per-dimension {len(dims)}D action min/max"
    parts.extend(
        [
            f'<text x="{left}" y="{top - 18}" class="label">{label_text}</text>',
            f'<line x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" y2="{top + panel_h}" class="axis"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_h}" class="axis"/>',
            f'<text x="{left - 55}" y="{top + 5}" class="tick">{ymax:.3f}</text>',
            f'<text x="{left - 55}" y="{top + panel_h}" class="tick">{ymin:.3f}</text>',
        ]
    )
    append_pi_guides_for_y(parts, left, top, panel_w, panel_h, ymin, ymax)
    for j, (label, stats) in enumerate(entries):
        color = colors[label]
        parts.append(f'<text x="{left + 12 + 180 * j}" y="{top + 20}" class="label" fill="{color}">{label}</text>')
        for axis_idx, dim_idx in enumerate(dims):
            base_x = left + (axis_idx + 0.5) * panel_w / len(dims)
            x = base_x + offsets[j]
            y1 = _sy(float(stats["min"][dim_idx]), ymin, ymax, top, panel_h)
            y2 = _sy(float(stats["max"][dim_idx]), ymin, ymax, top, panel_h)
            parts.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="5" stroke-opacity="0.85"/>')
            if j == 0:
                parts.append(f'<text x="{base_x - 8:.1f}" y="{top + panel_h + 24}" class="tick">{dim_idx}</text>')


def plot_compare(summaries, mode: str, out_path: Path):
    entries = [
        ("robocoin", summaries["robocoin"]),
        ("interna1", summaries["interna1"]),
        ("robotwin", summaries["robotwin"]),
    ]
    width, height = 1400, 960
    margin = 80
    panel_gap = 70
    panel_w = width - margin * 2
    panel_h = (height - margin * 2 - panel_gap) / 2
    colors = {
        "robocoin": "#1f77b4",
        "interna1": "#2ca02c",
        "robotwin": "#d62728",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:monospace} .title{font-size:24px;font-weight:bold} .label{font-size:16px} .tick{font-size:13px;fill:#555} .axis{stroke:#333;stroke-width:1.5}</style>',
        f'<text x="{margin}" y="40" class="title">14D joint+gripper min/max comparison ({mode})</text>',
    ]
    append_flat_range_panel(parts, margin, margin, panel_w, panel_h, entries, colors)
    append_per_dim_panel(parts, margin, margin + panel_h + panel_gap, panel_w, panel_h, entries, colors)
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def plot_joint_only_compare(summaries, mode: str, out_path: Path):
    entries = [
        ("robocoin", summaries["robocoin"]),
        ("interna1", summaries["interna1"]),
        ("robotwin", summaries["robotwin"]),
    ]
    width, height = 1400, 560
    margin = 80
    panel_w = width - margin * 2
    panel_h = height - margin * 2
    colors = {
        "robocoin": "#1f77b4",
        "interna1": "#2ca02c",
        "robotwin": "#d62728",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:monospace} .title{font-size:24px;font-weight:bold} .label{font-size:16px} .tick{font-size:13px;fill:#555} .axis{stroke:#333;stroke-width:1.5}</style>',
        f'<text x="{margin}" y="40" class="title">12D joint min/max comparison ({mode})</text>',
    ]
    append_per_dim_panel(
        parts,
        margin,
        margin,
        panel_w,
        panel_h,
        entries,
        colors,
        dims=np.arange(12),
        label_text="Per-dimension 12D joint min/max",
    )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    robocoin_task_dirs = []
    for prefix in ROBOCOIN_PREFIXES:
        robocoin_task_dirs.extend(find_task_dirs(args.data_root, prefix))
    summaries["robocoin"] = summarize_robocoin_group(
        sorted(robocoin_task_dirs),
        args.mode,
        abs_limit=PI_LIMIT,
    )
    for label, path in EXTERNAL_STATS.items():
        summaries[label] = load_external_stats(path)

    stats_path = args.out_dir / f"joint14_minmax_{args.mode}.json"
    figure_path = args.out_dir / f"joint14_minmax_{args.mode}.svg"
    joint_only_figure_path = args.out_dir / f"joint12_minmax_{args.mode}.svg"
    with open(stats_path, "w") as f:
        json.dump({k: jsonable(v) for k, v in summaries.items()}, f, indent=2)
    plot_compare(summaries, args.mode, figure_path)
    plot_joint_only_compare(summaries, args.mode, joint_only_figure_path)
    print(f"Saved stats to: {stats_path}")
    print(f"Saved figure to: {figure_path}")
    print(f"Saved joint-only figure to: {joint_only_figure_path}")


if __name__ == "__main__":
    main()
