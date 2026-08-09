#!/usr/bin/env python3
"""RoboTwin Eval 实时成功率监控脚本

Usage:
    python scripts/robotwin_eval_progress.py [--eval_dir PATH] [--watch INTERVAL]

Examples:
    # 查看默认路径的当前进度
    python scripts/robotwin_eval_progress.py

    # 指定 eval log 目录
    python scripts/robotwin_eval_progress.py --eval_dir results/Checkpoints/0326_robotwin_32_qwen3OFT_interna1_frozenhalf_chunk32_ft/checkpoints/output_eval

    # 每 60 秒自动刷新
    python scripts/robotwin_eval_progress.py --watch 60
"""

import argparse
import glob
import os
import re
import time
from datetime import datetime
from pathlib import Path


def parse_eval_log(log_path):
    """从 eval log 文件中提取最新的成功率信息"""
    log_name = os.path.basename(log_path)
    task_name = re.sub(r"_demo_(?:clean|randomized)_seed\d+\.log$", "", log_name)
    task_name = re.sub(r"^(?:steps_\d+_)?pytorch_model_", "", task_name)

    suc, total, rate, seed = None, None, None, None
    with open(log_path, "r", errors="ignore") as f:
        content = f.read()

    matches = re.findall(
        r"Success rate:.*?(\d+)/(\d+).*?=>\s*.*?([\d.]+)%.*?current seed:.*?(\d+)",
        content,
    )
    if matches:
        last = matches[-1]
        suc, total = int(last[0]), int(last[1])
        rate = float(last[2])
        seed = int(last[3])

    return {
        "task": task_name,
        "suc": suc,
        "total": total,
        "rate": rate,
        "seed": seed,
        "done": total == 100 if total is not None else False,
    }


def find_eval_dir():
    """自动搜索最近的 output_eval 目录"""
    base = Path(__file__).resolve().parent.parent / "results" / "Checkpoints"
    candidates = sorted(base.glob("*/checkpoints/output_eval"), key=os.path.getmtime, reverse=True)
    if candidates:
        return str(candidates[0])
    return None


def print_progress(eval_dir):
    logs = sorted(glob.glob(os.path.join(eval_dir, "*.log")))
    if not logs:
        print(f"No log files found in {eval_dir}")
        return

    results = [parse_eval_log(f) for f in logs]
    results = [r for r in results if r["total"] is not None]
    results.sort(key=lambda r: r["task"])

    done_tasks = [r for r in results if r["done"]]
    running_tasks = [r for r in results if not r["done"]]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 66}")
    print(f"  RoboTwin Eval Progress  |  {now}")
    print(f"  Dir: {eval_dir}")
    print(f"{'=' * 66}\n")

    print(f"  {'Task':<32} {'Suc':>5} {'Total':>6} {'Rate':>8} {'Status':>8}")
    print(f"  {'-' * 62}")

    total_suc, total_test = 0, 0

    for r in results:
        status = "DONE" if r["done"] else "..."
        print(f"  {r['task']:<32} {r['suc']:>5} {r['total']:>6} {r['rate']:>7.1f}% {status:>8}")
        total_suc += r["suc"]
        total_test += r["total"]

    print(f"  {'-' * 62}")

    if total_test > 0:
        avg = total_suc / total_test * 100
        print(f"  {'TOTAL':<32} {total_suc:>5} {total_test:>6} {avg:>7.1f}%")

    # 已完成任务的平均成功率（每任务权重相同）
    if done_tasks:
        avg_per_task = sum(r["rate"] for r in done_tasks) / len(done_tasks)
        print(f"\n  Completed tasks: {len(done_tasks)}/50,  avg SR (per task): {avg_per_task:.1f}%")

    print(f"  Running tasks:   {len(running_tasks)}")
    print(f"  Waiting tasks:   {50 - len(results)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="RoboTwin eval progress monitor")
    parser.add_argument("--eval_dir", type=str, default=None, help="Path to output_eval directory")
    parser.add_argument("--watch", type=int, default=0, help="Auto-refresh interval in seconds (0 = run once)")
    args = parser.parse_args()

    eval_dir = args.eval_dir
    if eval_dir is None:
        eval_dir = find_eval_dir()
        if eval_dir is None:
            print("Error: cannot find output_eval directory. Use --eval_dir to specify.")
            return

    if args.watch > 0:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                print_progress(eval_dir)
                print(f"  [Auto-refresh every {args.watch}s, Ctrl+C to stop]")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_progress(eval_dir)


if __name__ == "__main__":
    main()
