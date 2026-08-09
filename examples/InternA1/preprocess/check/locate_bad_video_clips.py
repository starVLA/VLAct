#!/usr/bin/env python3
"""Stage 2: read flagged episodes and localize bad clips."""

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

SERVER_PORT = 8000
TP_SIZE = 8
VIDEO_FPS = 8
MEM_FRACTION_STATIC = 0.85
CONTEXT_LENGTH = 32768
REASONING_PARSER = "qwen3"
PROMPT = (
    'You see synchronized multi-view videos of one robot episode.\n'
    'Task: "{task}"\n'
    'Episode duration: about {duration_sec:.2f} seconds.\n'
    'Find bad time ranges that should NOT be used for training: irreversible mistakes, wrong placement, recovery/retry after failure, '
    'or useless off-task motion after the useful demo ends.\n'
    'Do not mark normal reaching, grasping, transport, or mildly awkward but still usable motion as bad.\n'
    'Return STRICT JSON only:\n'
    '{{"task_completed": <true or false>, "summary": "<short summary>", '
    '"bad_ranges": [{{"start_sec": <float>, "end_sec": <float>, "reason": "<short reason>"}}]}}'
)


def read_jsonl(path: Path):
    text = path.read_text().strip() if path.exists() else ""
    return [json.loads(line) for line in text.splitlines()] if text else []


def call_vlm_json(client, model: str, content, max_retries: int):
    for attempt in range(max_retries):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            temperature=0.0 if attempt == 0 else 0.4,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        match = re.search(r"\{.*\}", (resp.choices[0].message.content or "").strip(), flags=re.S)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def launch_sglang_server(model_path: str, server_python: str | None):
    python_bin = server_python or sys.executable
    cmd = [
        python_bin, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(SERVER_PORT),
        "--tp-size", str(TP_SIZE),
        "--mm-process-config", json.dumps({"video": {"fps": VIDEO_FPS}}),
        "--mem-fraction-static", str(MEM_FRACTION_STATIC),
        "--context-length", str(CONTEXT_LENGTH),
        "--reasoning-parser", REASONING_PARSER,
    ]
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    for _ in range(1800):
        time.sleep(2)
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang exited early. Check environment: {python_bin}")
        try:
            if requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=3).status_code == 200:
                return proc
        except requests.ConnectionError:
            pass
    proc.kill()
    raise RuntimeError("SGLang server failed to start within 3600s")


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_bad_clips(bad_ranges, duration_sec: float):
    clips = []
    for row in bad_ranges or []:
        start = safe_float(row.get("start_sec", 0.0), 0.0)
        end = safe_float(row.get("end_sec", duration_sec if duration_sec > 0 else start), start)
        start = max(0.0, min(start, duration_sec)) if duration_sec > 0 else max(0.0, start)
        end = max(0.0, min(end, duration_sec)) if duration_sec > 0 else max(0.0, end)
        if end > start:
            clips.append({
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
                "reason": str(row.get("reason", "")).strip(),
            })
    return clips


def locate_bad_clips(client, model: str, data_root: Path, episode, max_retries: int):
    content = []
    for view_name, video_path in sorted(episode["views"].items()):
        resolved_path = data_root / video_path
        content += [
            {"type": "text", "text": f"Video view: {view_name}"},
            {"type": "video_url", "video_url": {"url": "file://" + str(resolved_path.resolve())}},
        ]
    content.append({"type": "text", "text": PROMPT.format(task=episode["task"], duration_sec=episode["duration_sec"])})
    review = call_vlm_json(client, model, content, max_retries)
    bad_clips = normalize_bad_clips(review.get("bad_ranges", []), safe_float(episode.get("duration_sec", 0.0), 0.0))
    return {
        "subset": episode["subset"],
        "episode_index": episode["episode_index"],
        "task": episode["task"],
        "fps": episode.get("fps"),
        "num_frames": episode.get("num_frames"),
        "duration_sec": safe_float(episode.get("duration_sec", 0.0), 0.0),
        "views": episode["views"],
        "final_state_check": episode.get("final_state_check"),
        "range_review": {
            "task_completed": bool(review.get("task_completed", False)),
            "summary": str(review.get("summary", "")).strip(),
        },
        "bad_clips": bad_clips,
        "status": "fail" if bad_clips or not bool(review.get("task_completed", False)) else "pass",
    }


def run_localization(args, client):
    episodes = read_jsonl(Path(args.input_json))
    if args.limit:
        episodes = episodes[:args.limit]
    with open(args.output_json, "w", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool, tqdm(total=len(episodes), desc="Locate bad clips", unit="episode") as pbar:
            data_root = Path(args.data_root)
            futures = [pool.submit(locate_bad_clips, client, args.model_path, data_root, episode, args.max_retries) for episode in episodes]
            for fut in as_completed(futures):
                fout.write(json.dumps(fut.result()) + "\n")
                pbar.update(1)
    print(f"Processed flagged episodes: {len(episodes)}")
    print(f"Bad clip results written to {args.output_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-path", default="/project/vonneumann1/wcy/models/LLM/Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--server-python")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if OpenAI is None:
        raise ImportError("openai package is required")
    server_proc = launch_sglang_server(args.model_path, args.server_python)
    try:
        client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{SERVER_PORT}/v1", timeout=300)
        run_localization(args, client)
    finally:
        server_proc.kill()
        server_proc.wait()
        print("Server stopped.")


if __name__ == "__main__":
    main()
