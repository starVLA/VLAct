#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path


DOMINO_TASKS = [
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

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RESULT_RE = re.compile(
    r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%\s*\|\s*Avg MS:\s*([\d.]+)\s*\|\s*current seed:\s*(\d+)",
    re.IGNORECASE,
)
FALLBACK_SR_RE = re.compile(r"Success Rate:\s*([\d.]+)%\s*\((\d+)/(\d+)\)", re.IGNORECASE)
FALLBACK_MS_RE = re.compile(r"(?:Avg MS|Manipulation Score).*?([\d.]+)", re.IGNORECASE)


def target_for_mode(mode: str, target_episodes: int) -> int:
    if target_episodes > 0:
        return target_episodes
    if mode == "demo_random_dynamic":
        return 200
    return 50


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def infer_task_and_mode(log_path: Path) -> tuple[str, str]:
    name = log_path.name
    match = re.match(r"(.+)_(demo_(?:clean|random)_dynamic)_slot\d+_gpu.+_port\d+_eval\.log$", name)
    if match:
        return match.group(1), match.group(2)

    for mode in ("demo_clean_dynamic", "demo_random_dynamic"):
        marker = f"_{mode}_"
        if marker in name:
            return name.split(marker, 1)[0], mode

    return name.removesuffix(".log"), "unknown"


def parse_eval_log(log_path: Path, target_episodes: int) -> dict:
    task, mode = infer_task_and_mode(log_path)
    mode_target_episodes = target_for_mode(mode, target_episodes)
    content = strip_ansi(log_path.read_text(errors="ignore"))

    suc = total = seed = None
    rate = avg_ms = None

    matches = RESULT_RE.findall(content)
    if matches:
        last = matches[-1]
        suc = int(last[0])
        total = int(last[1])
        rate = float(last[2])
        avg_ms = float(last[3])
        seed = int(last[4])
    else:
        fallback = FALLBACK_SR_RE.findall(content)
        if fallback:
            last = fallback[-1]
            rate = float(last[0])
            suc = int(last[1])
            total = int(last[2])
        ms_match = FALLBACK_MS_RE.findall(content)
        if ms_match:
            avg_ms = float(ms_match[-1])

    # Some DOMINO expert-check failures are caught and the task keeps running.
    # Treat a log as failed only when an error appears after the latest result line.
    last_result_idx = max(content.rfind("Success rate:"), content.rfind("Success Rate:"))
    last_error_idx = max(content.rfind("[ERROR]"), content.rfind("Traceback (most recent call last)"))
    failed = last_error_idx != -1 and last_error_idx > last_result_idx
    done = total is not None and total >= mode_target_episodes

    return {
        "task": task,
        "mode": mode,
        "suc": suc,
        "total": total,
        "rate": rate,
        "avg_ms": avg_ms,
        "seed": seed,
        "done": done,
        "target_episodes": mode_target_episodes,
        "failed": failed,
        "log": str(log_path),
    }


def find_latest_eval_dir() -> Path | None:
    base = Path(__file__).resolve().parent.parent / "results" / "Checkpoints"
    candidates = [p for p in base.rglob("checkpoints/output_eval") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def collect_logs(eval_dir: Path) -> list[Path]:
    return sorted(eval_dir.rglob("*_eval.log"))


def print_mode_summary(mode: str, rows: list[dict], target_episodes: int, eval_dir: Path) -> None:
    rows = sorted(rows, key=lambda r: r["task"])
    parsed = [r for r in rows if r["total"] is not None]
    done = [r for r in parsed if r["done"]]
    running = [r for r in parsed if not r["done"]]
    seen_tasks = {r["task"] for r in rows}
    missing = [task for task in DOMINO_TASKS if task not in seen_tasks]
    mode_target_episodes = target_for_mode(mode, target_episodes)

    total_suc = sum(r["suc"] for r in parsed if r["suc"] is not None)
    total_eps = sum(r["total"] for r in parsed if r["total"] is not None)
    weighted_sr = total_suc / total_eps * 100 if total_eps else 0.0
    avg_done_sr = sum(r["rate"] for r in done if r["rate"] is not None) / len(done) if done else 0.0
    ms_rows = [r for r in parsed if r["avg_ms"] is not None]
    avg_ms = sum(r["avg_ms"] for r in ms_rows) / len(ms_rows) if ms_rows else None
    done_ms_rows = [r for r in done if r["avg_ms"] is not None]
    avg_done_ms = sum(r["avg_ms"] for r in done_ms_rows) / len(done_ms_rows) if done_ms_rows else 0.0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 66}")
    print(f"  DOMINO Eval Progress  |  {now}")
    print(f"  Dir: {eval_dir}")
    if mode != "unknown":
        print(f"  Mode: {mode}  Target episodes: {mode_target_episodes}")
    print(f"{'=' * 66}\n")

    print(f"  {'Task':<32} {'Suc':>5} {'Total':>6} {'SR':>8} {'MS':>7} {'Status':>8}")
    print(f"  {'-' * 70}")

    for row in rows:
        if row["total"] is None:
            print(f"  {row['task']:<32} {'--':>5} {'--':>6} {'--':>8} {'--':>7} {'...':>8}")
            continue

        status = "DONE" if row["done"] else "..."
        ms_text = f"{row['avg_ms']:.1f}" if row["avg_ms"] is not None else "--"
        print(
            f"  {row['task']:<32} {row['suc']:>5} {row['total']:>6} "
            f"{row['rate']:>7.1f}% {ms_text:>7} {status:>8}"
        )

    print(f"  {'-' * 70}")
    avg_ms_text = f"{avg_ms:.1f}" if avg_ms is not None else "--"
    print(f"  {'TOTAL':<32} {total_suc:>5} {total_eps:>6} {weighted_sr:>7.1f}% {avg_ms_text:>7}")
    print()
    print(
        f"  Completed tasks: {len(done)}/{len(DOMINO_TASKS)},  "
        f"avg SR (per task): {avg_done_sr:.1f}%,  avg MS (per task): {avg_done_ms:.1f}"
    )
    print(f"  Running tasks:   {len(running)}")
    print(f"  Waiting tasks:   {len(missing)}")


def print_progress(eval_dir: Path, target_episodes: int) -> None:
    logs = collect_logs(eval_dir)

    if not logs:
        print("No *_eval.log files found.")
        return

    rows_by_task: dict[tuple[str, str], dict] = {}
    for path in logs:
        row = parse_eval_log(path, target_episodes)
        key = (row["mode"], row["task"])
        previous = rows_by_task.get(key)
        if previous is None or path.stat().st_mtime > Path(previous["log"]).stat().st_mtime:
            rows_by_task[key] = row

    rows = list(rows_by_task.values())
    modes = sorted({row["mode"] for row in rows})
    for mode in modes:
        print_mode_summary(mode, [row for row in rows if row["mode"] == mode], target_episodes, eval_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DOMINO eval logs.")
    parser.add_argument("--eval_dir", type=Path, default=None, help="Path to output_eval or a DOMINO log subdirectory.")
    parser.add_argument("--watch", type=int, default=0, help="Auto-refresh interval in seconds.")
    parser.add_argument(
        "--target-episodes",
        type=int,
        default=0,
        help="Episodes expected per task. Default: auto (50 for clean, 200 for random).",
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir or find_latest_eval_dir()
    if eval_dir is None:
        raise SystemExit("Cannot find an output_eval directory. Pass --eval_dir explicitly.")
    if not eval_dir.is_dir():
        raise SystemExit(f"Eval dir does not exist: {eval_dir}")

    if args.watch > 0:
        try:
            while True:
                os.system("clear")
                print_progress(eval_dir, args.target_episodes)
                print(f"\n[Auto-refresh every {args.watch}s, Ctrl+C to stop]")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_progress(eval_dir, args.target_episodes)


if __name__ == "__main__":
    main()
