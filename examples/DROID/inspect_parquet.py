#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PARQUET = Path(
    "/project/vonneumann1/wcy/copy/starVLA-VLAct/playground/Datasets/DROID/data/chunk-000/episode_000000.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one DROID parquet file.")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--max-rows", type=int, default=5)
    parser.add_argument("--full", action="store_true", help="Print all rows for each key.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.parquet)

    print(f"parquet: {args.parquet}")
    print(f"num_rows: {len(df)}")
    print(f"columns: {list(df.columns)}")

    for key in df.columns:
        print("\n" + "=" * 100)
        print(f"key: {key}")
        series = df[key]
        values = series.tolist() if args.full else series.head(args.max_rows).tolist()
        for idx, value in enumerate(values):
            print(f"[{idx}] {value}")
        if not args.full and len(series) > args.max_rows:
            print(f"... ({len(series) - args.max_rows} more rows)")


if __name__ == "__main__":
    main()
