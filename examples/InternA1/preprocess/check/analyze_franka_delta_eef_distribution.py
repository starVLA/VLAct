#!/usr/bin/env python3
"""Plot delta EEF action distribution for all Franka datasets in sim_updated.

The script assumes Franka datasets store absolute EEF targets in
`actions.gripper.pose` and current EEF state in `states.gripper.pose`.
Delta EEF action is computed as:

    delta = actions.gripper.pose - states.gripper.pose

For speed, the script:
1. discovers all dataset folders containing `/franka/`
2. collects all parquet files
3. keeps every N-th parquet as a deterministic sample
4. runs two multiprocessing passes:
   - pass 1: aggregate count/sum/sumsq/min/max
   - pass 2: aggregate histograms with global min/max bin edges
5. renders a single SVG figure with 6 histograms
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(
    "/project/vonneumann1/wcy/copy/starVLA-VLAct/playground/Datasets/InternData-A1/sim_updated"
)
DEFAULT_OUTPUT = Path(
    "/project/vonneumann1/wcy/copy/starVLA-VLAct/examples/InternA1/preprocess/check/out/franka_delta_eef_action_distribution.svg"
)
ACTION_KEY = "actions.gripper.pose"
STATE_KEY = "states.gripper.pose"
DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]
DEFAULT_NUM_WORKERS = 32
DEFAULT_SAMPLE_EVERY = 10
DEFAULT_NUM_BINS = 201


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    return parser.parse_args()


def discover_franka_dataset_dirs(root: Path) -> list[Path]:
    dataset_dirs: list[Path] = []
    rg_bin = shutil.which("rg")
    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        franka_root = category_dir / "franka"
        if not franka_root.is_dir():
            continue
        added_before = len(dataset_dirs)

        info_paths: list[Path] = []
        if rg_bin is not None:
            result = subprocess.run(
                [rg_bin, "--files", str(franka_root), "-g", "**/meta/info.json"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode in (0, 1):
                info_paths = [Path(line) for line in result.stdout.splitlines() if line]

        if not info_paths:
            for dirpath, _, filenames in os.walk(franka_root):
                if "info.json" not in filenames or not dirpath.endswith("/meta"):
                    continue
                info_paths.append(Path(dirpath) / "info.json")

        for info_path in info_paths:
            dataset_dir = info_path.parent.parent
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
            except Exception:
                continue
            features = info.get("features", {})
            if ACTION_KEY not in features or STATE_KEY not in features:
                continue
            dataset_dirs.append(dataset_dir)
        print(
            f"[discover] {category_dir.name}: +{len(dataset_dirs) - added_before} datasets",
            flush=True,
        )
    return dataset_dirs


def collect_parquet_paths(dataset_dirs: list[Path]) -> list[Path]:
    parquet_paths: list[Path] = []
    for dataset_dir in dataset_dirs:
        data_dir = dataset_dir / "data"
        if not data_dir.is_dir():
            continue
        parquet_paths.extend(sorted(data_dir.rglob("*.parquet")))
    return parquet_paths


def sample_parquet_paths(
    parquet_paths: list[Path], sample_every: int, sample_offset: int
) -> list[Path]:
    if sample_every <= 1:
        return parquet_paths
    offset = sample_offset % sample_every
    return [path for idx, path in enumerate(parquet_paths) if idx % sample_every == offset]


def load_delta_matrix(parquet_path: Path) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=[ACTION_KEY, STATE_KEY])
    action_arr = np.vstack(df[ACTION_KEY].to_numpy()).astype(np.float32)
    state_arr = np.vstack(df[STATE_KEY].to_numpy()).astype(np.float32)
    return action_arr - state_arr


def summarize_one(parquet_path: Path) -> dict:
    delta = load_delta_matrix(parquet_path)
    return {
        "rows": int(delta.shape[0]),
        "sum": delta.sum(axis=0, dtype=np.float64).tolist(),
        "sumsq": np.square(delta, dtype=np.float64).sum(axis=0, dtype=np.float64).tolist(),
        "min": delta.min(axis=0).astype(np.float64).tolist(),
        "max": delta.max(axis=0).astype(np.float64).tolist(),
    }


def build_bin_edges(global_min: np.ndarray, global_max: np.ndarray, num_bins: int) -> list[np.ndarray]:
    edges_list: list[np.ndarray] = []
    for lo, hi in zip(global_min, global_max):
        if not math.isfinite(float(lo)) or not math.isfinite(float(hi)):
            lo, hi = -1.0, 1.0
        if hi <= lo:
            center = float(lo)
            lo = center - 1e-6
            hi = center + 1e-6
        edges_list.append(np.linspace(lo, hi, num_bins + 1, dtype=np.float64))
    return edges_list


def histogram_one(parquet_path: Path, edges_list: list[np.ndarray]) -> list[list[int]]:
    delta = load_delta_matrix(parquet_path)
    histograms: list[list[int]] = []
    for dim_idx, edges in enumerate(edges_list):
        counts, _ = np.histogram(delta[:, dim_idx], bins=edges)
        histograms.append(counts.astype(np.int64).tolist())
    return histograms


def reduce_summaries(results: list[dict]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_rows = 0
    total_sum = np.zeros(len(DIM_NAMES), dtype=np.float64)
    total_sumsq = np.zeros(len(DIM_NAMES), dtype=np.float64)
    global_min = np.full(len(DIM_NAMES), np.inf, dtype=np.float64)
    global_max = np.full(len(DIM_NAMES), -np.inf, dtype=np.float64)

    for item in results:
        total_rows += int(item["rows"])
        total_sum += np.asarray(item["sum"], dtype=np.float64)
        total_sumsq += np.asarray(item["sumsq"], dtype=np.float64)
        global_min = np.minimum(global_min, np.asarray(item["min"], dtype=np.float64))
        global_max = np.maximum(global_max, np.asarray(item["max"], dtype=np.float64))

    return total_rows, total_sum, total_sumsq, global_min, global_max


def run_summary_pass(parquet_paths: list[Path], num_workers: int) -> list[dict]:
    results: list[dict] = []
    with Pool(num_workers) as pool:
        for idx, item in enumerate(pool.imap_unordered(summarize_one, parquet_paths), start=1):
            results.append(item)
            if idx % 200 == 0 or idx == len(parquet_paths):
                print(f"[summary {idx}/{len(parquet_paths)}]", flush=True)
    return results


def run_histogram_pass(
    parquet_paths: list[Path], num_workers: int, edges_list: list[np.ndarray]
) -> np.ndarray:
    hist_total = np.zeros((len(DIM_NAMES), len(edges_list[0]) - 1), dtype=np.int64)
    worker = partial(histogram_one, edges_list=edges_list)
    with Pool(num_workers) as pool:
        for idx, histograms in enumerate(pool.imap_unordered(worker, parquet_paths), start=1):
            hist_total += np.asarray(histograms, dtype=np.int64)
            if idx % 200 == 0 or idx == len(parquet_paths):
                print(f"[hist {idx}/{len(parquet_paths)}]", flush=True)
    return hist_total


def build_output_payload(
    args: argparse.Namespace,
    dataset_dirs: list[Path],
    parquet_paths: list[Path],
    sampled_paths: list[Path],
    total_rows: int,
    total_sum: np.ndarray,
    total_sumsq: np.ndarray,
    global_min: np.ndarray,
    global_max: np.ndarray,
    hist_total: np.ndarray,
    edges_list: list[np.ndarray],
) -> dict:
    mean = total_sum / total_rows
    var = np.maximum(total_sumsq / total_rows - np.square(mean), 0.0)
    std = np.sqrt(var)

    dims = {}
    for dim_idx, dim_name in enumerate(DIM_NAMES):
        dims[dim_name] = {
            "count": int(total_rows),
            "min": float(global_min[dim_idx]),
            "max": float(global_max[dim_idx]),
            "mean": float(mean[dim_idx]),
            "std": float(std[dim_idx]),
            "histogram": {
                "bin_edges": edges_list[dim_idx].tolist(),
                "counts": hist_total[dim_idx].tolist(),
            },
        }

    return {
        "root": str(args.root),
        "action_key": ACTION_KEY,
        "state_key": STATE_KEY,
        "delta_formula": f"{ACTION_KEY} - {STATE_KEY}",
        "num_dataset_dirs": len(dataset_dirs),
        "num_parquet_total": len(parquet_paths),
        "num_parquet_sampled": len(sampled_paths),
        "sample_every": int(args.sample_every),
        "sample_offset": int(args.sample_offset),
        "num_workers": int(args.num_workers),
        "num_bins": int(args.num_bins),
        "total_rows": int(total_rows),
        "dimensions": dims,
        "sampled_parquet_examples": [
            str(path.relative_to(args.root)) for path in sampled_paths[:20]
        ],
    }


def sx(value: float, vmin: float, vmax: float, left: float, width: float) -> float:
    return left + (value - vmin) / max(vmax - vmin, 1e-12) * width


def sy(value: float, vmax: float, top: float, height: float) -> float:
    return top + height - value / max(vmax, 1e-12) * height


def append_hist_panel(
    parts: list[str],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    dim_name: str,
    stats: dict,
    color: str,
) -> None:
    hist = stats["histogram"]["counts"]
    edges = stats["histogram"]["bin_edges"]
    vmax = max(max(hist), 1)
    parts.append(f'<rect x="{left:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" fill="#ffffff" stroke="#dddddd"/>')
    parts.append(f'<text x="{left:.2f}" y="{top - 14:.2f}" class="label" fill="{color}">{dim_name}</text>')
    parts.append(f'<line x1="{left:.2f}" y1="{top + height:.2f}" x2="{left + width:.2f}" y2="{top + height:.2f}" class="axis"/>')
    parts.append(f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{top + height:.2f}" class="axis"/>')
    for idx, count in enumerate(hist):
        x1 = sx(edges[idx], edges[0], edges[-1], left, width)
        x2 = sx(edges[idx + 1], edges[0], edges[-1], left, width)
        y = sy(count, vmax, top, height)
        parts.append(
            f'<rect x="{x1:.2f}" y="{y:.2f}" width="{max(x2 - x1 - 0.5, 0.5):.2f}" '
            f'height="{top + height - y:.2f}" fill="{color}" fill-opacity="0.78"/>'
        )
    parts.append(f'<text x="{left:.2f}" y="{top + height + 18:.2f}" class="tick">{edges[0]:.6g}</text>')
    parts.append(
        f'<text x="{left + width:.2f}" y="{top + height + 18:.2f}" class="tick" text-anchor="end">{edges[-1]:.6g}</text>'
    )
    parts.append(f'<text x="{left - 6:.2f}" y="{top + 10:.2f}" class="tick" text-anchor="end">{vmax}</text>')

    stat_lines = [
        f"count={stats['count']}",
        f"min={stats['min']:.6g}",
        f"max={stats['max']:.6g}",
        f"mean={stats['mean']:.6g}",
        f"std={stats['std']:.6g}",
    ]
    text_y = top + height + 42
    for idx, line in enumerate(stat_lines):
        parts.append(
            f'<text x="{left:.2f}" y="{text_y + idx * 16:.2f}" class="tick">{line}</text>'
        )


def render_svg(summary: dict, out_path: Path) -> None:
    width = 1800
    height = 1180
    margin_x = 70
    margin_y = 80
    col_gap = 45
    row_gap = 130
    panel_width = (width - margin_x * 2 - col_gap * 2) / 3
    panel_height = 220
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f7f7"/>',
        '<style>text{font-family:monospace} .title{font-size:26px;font-weight:bold;fill:#111} .subtitle{font-size:15px;fill:#444} .label{font-size:18px;font-weight:bold} .tick{font-size:13px;fill:#555} .axis{stroke:#333;stroke-width:1.2}</style>',
        f'<text x="{margin_x}" y="42" class="title">Franka delta EEF action distribution</text>',
        (
            f'<text x="{margin_x}" y="66" class="subtitle">'
            f"delta = {summary['delta_formula']} | datasets={summary['num_dataset_dirs']} | "
            f"parquets={summary['num_parquet_sampled']}/{summary['num_parquet_total']} | "
            f"sample_every={summary['sample_every']} | rows={summary['total_rows']}"
            "</text>"
        ),
    ]

    for idx, dim_name in enumerate(DIM_NAMES):
        row = idx // 3
        col = idx % 3
        left = margin_x + col * (panel_width + col_gap)
        top = margin_y + row * (panel_height + row_gap)
        append_hist_panel(
            parts,
            left=left,
            top=top,
            width=panel_width,
            height=panel_height,
            dim_name=dim_name,
            stats=summary["dimensions"][dim_name],
            color=colors[idx],
        )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.sample_every <= 0:
        raise ValueError("--sample-every must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.num_bins <= 0:
        raise ValueError("--num-bins must be positive")

    dataset_dirs = discover_franka_dataset_dirs(args.root)
    parquet_paths = collect_parquet_paths(dataset_dirs)
    sampled_paths = sample_parquet_paths(
        parquet_paths, sample_every=args.sample_every, sample_offset=args.sample_offset
    )

    print(f"franka dataset dirs: {len(dataset_dirs)}", flush=True)
    print(f"total parquet files: {len(parquet_paths)}", flush=True)
    print(f"sampled parquet files: {len(sampled_paths)}", flush=True)

    if not sampled_paths:
        raise RuntimeError("No sampled parquet files found.")

    summary_results = run_summary_pass(sampled_paths, args.num_workers)
    total_rows, total_sum, total_sumsq, global_min, global_max = reduce_summaries(summary_results)
    edges_list = build_bin_edges(global_min, global_max, args.num_bins)
    hist_total = run_histogram_pass(sampled_paths, args.num_workers, edges_list)

    payload = build_output_payload(
        args=args,
        dataset_dirs=dataset_dirs,
        parquet_paths=parquet_paths,
        sampled_paths=sampled_paths,
        total_rows=total_rows,
        total_sum=total_sum,
        total_sumsq=total_sumsq,
        global_min=global_min,
        global_max=global_max,
        hist_total=hist_total,
        edges_list=edges_list,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_svg(payload, args.output)

    print(f"total rows: {total_rows}", flush=True)
    print(f"saved svg: {args.output}", flush=True)


if __name__ == "__main__":
    main()
