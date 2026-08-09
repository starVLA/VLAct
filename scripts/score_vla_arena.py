#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SUITE_ORDER = [
    "safety_static_obstacles",
    "safety_cautious_grasp",
    "safety_hazard_avoidance",
    "safety_state_preservation",
    "safety_dynamic_obstacles",
    "distractor_static_distractors",
    "distractor_dynamic_distractors",
    "extrapolation_preposition_combinations",
    "extrapolation_task_workflows",
    "extrapolation_unseen_objects",
    "long_horizon",
]
LEVEL_ORDER = ["L0", "L1", "L2"]
EXPECTED_KEYS = [f"{suite}:{level}" for suite in SUITE_ORDER for level in LEVEL_ORDER]
SAFETY_SUITES = {
    "safety_static_obstacles",
    "safety_cautious_grasp",
    "safety_hazard_avoidance",
    "safety_state_preservation",
    "safety_dynamic_obstacles",
}


def infer_domain(suite: str) -> str:
    if suite.startswith("safety_"):
        return "safety"
    if suite.startswith("distractor_"):
        return "distractor"
    if suite.startswith("extrapolation_"):
        return "extrapolation"
    if suite == "long_horizon":
        return "long_horizon"
    return "other"


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def load_rows(summary_csv: Path) -> list[dict]:
    with summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["Success Rate"] = float(row["Success Rate"])
            row["Successes"] = int(row["Successes"])
            row["Total Episodes"] = int(row["Total Episodes"])
            row["Avg Cost"] = float(row["Avg Cost"])
            row["domain"] = infer_domain(row["Task Suite"])
            row["key"] = f'{row["Task Suite"]}:{row["Level"]}'
            rows.append(row)
    return rows


def aggregate_rows(rows: list[dict], key_fn) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)

    out = {}
    for key, group in buckets.items():
        total_successes = sum(r["Successes"] for r in group)
        total_episodes = sum(r["Total Episodes"] for r in group)
        avg_macro_sr = safe_div(sum(r["Success Rate"] for r in group), len(group))
        avg_cost = safe_div(sum(r["Avg Cost"] for r in group), len(group))
        out[key] = {
            "jobs": len(group),
            "weighted_sr": safe_div(total_successes, total_episodes),
            "macro_sr": avg_macro_sr,
            "avg_cost": avg_cost,
            "successes": total_successes,
            "episodes": total_episodes,
        }
    return out


def build_report(rows: list[dict]) -> dict:
    completed = {row["key"] for row in rows}
    missing = [key for key in EXPECTED_KEYS if key not in completed]

    total_successes = sum(r["Successes"] for r in rows)
    total_episodes = sum(r["Total Episodes"] for r in rows)
    safety_rows = [r for r in rows if r["Task Suite"] in SAFETY_SUITES]

    report = {
        "completed_jobs": len(rows),
        "expected_jobs": len(EXPECTED_KEYS),
        "missing_jobs": missing,
        "overall": {
            "weighted_sr": safe_div(total_successes, total_episodes),
            "macro_sr": safe_div(sum(r["Success Rate"] for r in rows), len(rows)),
            "successes": total_successes,
            "episodes": total_episodes,
        },
        "by_domain": aggregate_rows(rows, lambda r: r["domain"]),
        "by_suite": aggregate_rows(rows, lambda r: r["Task Suite"]),
        "by_level": aggregate_rows(rows, lambda r: r["Level"]),
        "safety_by_level": aggregate_rows(safety_rows, lambda r: r["Level"]),
        "by_suite_level": {
            row["key"]: {
                "domain": row["domain"],
                "success_rate": row["Success Rate"],
                "avg_cost": row["Avg Cost"],
                "successes": row["Successes"],
                "episodes": row["Total Episodes"],
                "log_file": row["Log File"],
            }
            for row in rows
        },
        "safety_only": aggregate_rows(safety_rows, lambda r: r["Task Suite"]),
        "safety_overall_avg_cost": safe_div(sum(r["Avg Cost"] for r in safety_rows), len(safety_rows)),
    }
    return report


def print_table(title: str, items: list[tuple[str, dict]]) -> None:
    print(f"\n## {title}")
    print(f'{"name":36}  {"weighted_sr":>11}  {"macro_sr":>9}  {"avg_cost":>9}  {"jobs":>4}')
    for name, stats in items:
        print(
            f'{name:36}  '
            f'{stats["weighted_sr"]:11.4f}  '
            f'{stats["macro_sr"]:9.4f}  '
            f'{stats["avg_cost"]:9.4f}  '
            f'{stats["jobs"]:4d}'
        )


def format_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "--"


def print_suite_level_table(report: dict) -> None:
    print("\n## Suite x Level SR")
    print(f'{"suite":36}  {"L0":>8}  {"L1":>8}  {"L2":>8}  {"avg":>8}')
    for suite in SUITE_ORDER:
        level_values = []
        for level in LEVEL_ORDER:
            key = f"{suite}:{level}"
            stats = report["by_suite_level"].get(key)
            level_values.append(stats["success_rate"] if stats else None)

        suite_avg = report["by_suite"].get(suite, {}).get("macro_sr")
        print(
            f"{suite:36}  "
            f"{format_metric(level_values[0]):>8}  "
            f"{format_metric(level_values[1]):>8}  "
            f"{format_metric(level_values[2]):>8}  "
            f"{format_metric(suite_avg):>8}"
        )


def print_safety_cost_table(report: dict) -> None:
    print("\n## Safety Avg Cost")
    print(f'{"suite":36}  {"L0":>8}  {"L1":>8}  {"L2":>8}  {"avg":>8}')
    for suite in SUITE_ORDER:
        if suite not in SAFETY_SUITES:
            continue

        level_values = []
        for level in LEVEL_ORDER:
            key = f"{suite}:{level}"
            stats = report["by_suite_level"].get(key)
            level_values.append(stats["avg_cost"] if stats else None)

        suite_avg = report["by_suite"].get(suite, {}).get("avg_cost")
        print(
            f"{suite:36}  "
            f"{format_metric(level_values[0]):>8}  "
            f"{format_metric(level_values[1]):>8}  "
            f"{format_metric(level_values[2]):>8}  "
            f"{format_metric(suite_avg):>8}"
        )


def print_level_summary(report: dict) -> None:
    print("\n## Avg By Level")
    print(f'{"metric":36}  {"L0":>8}  {"L1":>8}  {"L2":>8}  {"avg":>8}')

    sr_values = []
    cc_values = []
    for level in LEVEL_ORDER:
        sr_stats = report["by_level"].get(level)
        cc_stats = report["safety_by_level"].get(level)
        sr_values.append(sr_stats["macro_sr"] if sr_stats else None)
        cc_values.append(cc_stats["avg_cost"] if cc_stats else None)

    print(
        f'{"Success Rate":36}  '
        f'{format_metric(sr_values[0]):>8}  '
        f'{format_metric(sr_values[1]):>8}  '
        f'{format_metric(sr_values[2]):>8}  '
        f'{format_metric(report["overall"]["macro_sr"]):>8}'
    )
    print(
        f'{"Constraint Cost":36}  '
        f'{format_metric(cc_values[0]):>8}  '
        f'{format_metric(cc_values[1]):>8}  '
        f'{format_metric(cc_values[2]):>8}  '
        f'{format_metric(report["safety_overall_avg_cost"]):>8}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a VLA-Arena summary CSV.")
    parser.add_argument("summary_csv", type=Path, help="Path to *_vla_arena_summary.csv")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON instead of a human-readable table.",
    )
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    report = build_report(rows)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"Summary CSV : {args.summary_csv}")
    print(
        f'Completed   : {report["completed_jobs"]}/{report["expected_jobs"]} '
        f'(missing: {len(report["missing_jobs"])})'
    )
    print(
        f'Overall SR  : weighted={report["overall"]["weighted_sr"]:.4f} '
        f'macro={report["overall"]["macro_sr"]:.4f} '
        f'({report["overall"]["successes"]}/{report["overall"]["episodes"]})'
    )

    print_suite_level_table(report)
    print_safety_cost_table(report)
    print_level_summary(report)


if __name__ == "__main__":
    main()
