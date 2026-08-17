#!/usr/bin/env python3
"""RoboTwin Eval 实时成功率监控脚本

Usage:
    python scripts/robotwin_eval_progress.py [--eval_dir PATH] [--watch INTERVAL] [--all-runs]

Examples:
    # 查看默认路径的当前进度
    python scripts/robotwin_eval_progress.py

    # 指定 output_eval 日志目录
    python scripts/robotwin_eval_progress.py --eval_dir results/Checkpoints/0317_robotwin_qwen3OFT_interna1_60k_unified_ft_lr_base/checkpoints/output_eval

    # 统计单个 RoboTwin 结果目录
    python scripts/robotwin_eval_progress_sq.py --eval_dir /project/vonneumann1/wcy/copy/starVLA-VLAct/results/Robotwin/0315_robotwin_qwen3OFT

    # 统计整个 eval_result 根目录，默认每个 task 只取最新一次
    python scripts/robotwin_eval_progress_sq.py --eval_dir /project/vonneumann1/sqyang/project/EM-LLaVA/simulated_env/RoboTwin/eval_result

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


SUCCESS_RE = re.compile(
    r"Success rate:.*?(\d+)/(\d+).*?=>\s*.*?([\d.]+)%.*?current seed:.*?(\d+)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def parse_eval_log(log_path):
    """从 eval log 文件中提取最新的成功率信息"""
    task_name = re.sub(
        r"^steps_\d+_pytorch_model_|_demo_clean_seed\d+\.log$|_demo_randomized_seed\d+\.log$",
        "",
        os.path.basename(log_path),
    )

    suc, total, rate, seed = None, None, None, None
    with open(log_path, "r", errors="ignore") as f:
        content = strip_ansi(f.read())

    matches = SUCCESS_RE.findall(content)
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
        "source": log_path,
        "run": None,
        "kind": "log",
    }


def infer_task_from_result_path(result_path):
    path = Path(result_path).resolve()
    parts = path.parts
    if "model2robotwin_interface" in parts:
        idx = parts.index("model2robotwin_interface")
        if idx > 0:
            return parts[idx - 1]
    if "eval_result" in parts:
        idx = parts.index("eval_result")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if len(path.parents) >= 5:
        return path.parents[4].name
    return path.parent.name


def parse_result_file(result_path):
    """从 RoboTwin eval_result 目录中的 _result.txt 提取成功率。"""
    result_path_obj = Path(result_path).resolve()
    with open(result_path, "r", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    rate = None
    for line in reversed(lines):
        try:
            value = float(line)
        except ValueError:
            continue
        rate = value * 100 if value <= 1.0 else value
        break

    run_dir = result_path_obj.parent.name
    episode_count = sum(1 for _ in result_path_obj.parent.glob("episode*.mp4"))
    return {
        "task": infer_task_from_result_path(result_path),
        "suc": None,
        "total": None,
        "rate": rate,
        "seed": None,
        "done": episode_count >= 100,
        "source": result_path,
        "run": run_dir,
        "kind": "result",
        "episodes": episode_count,
    }


def collect_result_files(target_path):
    path = Path(target_path).expanduser().resolve()
    if path.is_file() and path.name == "_result.txt":
        return [str(path)]
    if not path.is_dir():
        return []

    direct_file = path / "_result.txt"
    if direct_file.is_file():
        return [str(direct_file)]

    return sorted(str(p) for p in path.rglob("_result.txt"))


def keep_latest_per_task(records):
    latest = {}
    for record in records:
        current = latest.get(record["task"])
        if current is None or os.path.getmtime(record["source"]) > os.path.getmtime(current["source"]):
            latest[record["task"]] = record
    return sorted(latest.values(), key=lambda r: (r["task"], r["run"] or ""))


def count_runs_per_task(records):
    counts = {}
    for record in records:
        counts[record["task"]] = counts.get(record["task"], 0) + 1
    return counts


def load_records(eval_dir, keep_all_runs=False):
    path = Path(eval_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {eval_dir}")

    if path.is_file() and path.suffix == ".log":
        records = [parse_eval_log(str(path))]
        records = [r for r in records if r["total"] is not None]
        return "log", records

    if path.is_dir():
        logs = sorted(glob.glob(os.path.join(str(path), "*.log")))
        if logs:
            records = [parse_eval_log(f) for f in logs]
            records = [r for r in records if r["total"] is not None]
            return "log", sorted(records, key=lambda r: r["task"])

    result_files = collect_result_files(str(path))
    if result_files:
        records = [parse_result_file(f) for f in result_files]
        records = [r for r in records if r["rate"] is not None]
        run_counts = count_runs_per_task(records)
        if not keep_all_runs:
            records = keep_latest_per_task(records)
        else:
            records.sort(key=lambda r: (r["task"], r["run"] or ""))
        for record in records:
            record["runs"] = run_counts.get(record["task"], 0)
        return "result", records

    return None, []


def find_eval_dir():
    """自动搜索最近的 output_eval 目录"""
    base = Path(__file__).resolve().parent.parent / "results" / "Checkpoints"
    candidates = sorted(base.glob("*/checkpoints/output_eval"), key=os.path.getmtime, reverse=True)
    if candidates:
        return str(candidates[0])
    return None


def print_log_progress(eval_dir, results):
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


def print_result_progress(eval_dir, results, keep_all_runs=False):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 102}")
    print(f"  RoboTwin Eval Result Summary  |  {now}")
    print(f"  Dir: {eval_dir}")
    print(f"{'=' * 102}\n")

    print(f"  {'Task':<32} {'Rate':>8} {'Eps':>5} {'100?':>6} {'Runs':>6} {'Run':<23}")
    print(f"  {'-' * 91}")

    for r in results:
        run = r["run"] or "-"
        runs = r.get("runs", 1)
        episodes = r.get("episodes", 0)
        reached_100 = "YES" if episodes >= 100 else "NO"
        print(f"  {r['task']:<32} {r['rate']:>7.1f}% {episodes:>5} {reached_100:>6} {runs:>6} {run:<23}")

    print(f"  {'-' * 91}")

    if results:
        avg_rate = sum(r["rate"] for r in results) / len(results)
        full_success = sum(1 for r in results if r["rate"] >= 100.0)
        reached_100_tasks = sum(1 for r in results if r.get("episodes", 0) >= 100)
        total_runs = sum(r.get("runs", 1) for r in results) if not keep_all_runs else len(results)
        print(f"  Tasks counted:  {len(results)}")
        print(f"  Runs found:     {total_runs}")
        print(f"  Reached 100:    {reached_100_tasks}/{len(results)}")
        print(f"  Mean SR:        {avg_rate:.1f}%")
        print(f"  100% tasks:     {full_success}")
        if not keep_all_runs:
            print("  Mode:           latest run per task")
        else:
            print("  Mode:           all discovered runs")
    else:
        print("  No valid _result.txt files found.")
    print()


def print_progress(eval_dir, keep_all_runs=False):
    mode, results = load_records(eval_dir, keep_all_runs=keep_all_runs)
    if not results:
        print(f"No eval logs or _result.txt files found in {eval_dir}")
        return

    if mode == "log":
        print_log_progress(eval_dir, results)
        return

    print_result_progress(eval_dir, results, keep_all_runs=keep_all_runs)


def main():
    parser = argparse.ArgumentParser(description="RoboTwin eval progress monitor")
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=None,
        help="Path to output_eval logs, a RoboTwin result directory, or the eval_result root",
    )
    parser.add_argument("--watch", type=int, default=0, help="Auto-refresh interval in seconds (0 = run once)")
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="When scanning eval_result directories, include all discovered runs instead of only the latest per task",
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    if eval_dir is None:
        eval_dir = find_eval_dir()
        if eval_dir is None:
            print("Error: cannot find output_eval directory. Use --eval_dir to specify a target path.")
            return

    if args.watch > 0:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                print_progress(eval_dir, keep_all_runs=args.all_runs)
                print(f"  [Auto-refresh every {args.watch}s, Ctrl+C to stop]")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_progress(eval_dir, keep_all_runs=args.all_runs)


if __name__ == "__main__":
    main()
