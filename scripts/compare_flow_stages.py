#!/usr/bin/env python
"""Compare first-40-frame validation metrics across motion/flow runs.

Reads the latest ``metrics/val-step=*.csv`` written by each run's validation
loop and prints a comparison table.  Intended usage::

    python scripts/compare_flow_stages.py \
        outputs/flowml_baseline outputs/flowml_stage1 outputs/flowml_stage2

The CSV MEAN row averages the per-image metrics, so the numbers are directly
comparable as long as all runs validated on the same frame subset.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


KEY_METRICS = [
    "psnr",
    "lpips",
    "gray_loss",
    "ssim_loss",
    "w_psnr",
    "w_lpips",
    "w_l1",
    "w_ssim",
]


def _step_number(name: str) -> int:
    match = re.search(r"val-step=(\d+)\.csv$", name)
    return int(match.group(1)) if match else -1


def _float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def latest_metrics_csv(run_dir: Path) -> Path | None:
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        return None
    candidates = list(metrics_dir.glob("val-step=*.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: _step_number(p.name))


def read_mean_row(csv_path: Path) -> tuple[str, dict[str, float]]:
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))

    header = rows[0][1:]  # drop the leading "name" column
    for row in rows:
        if row and row[0] == "MEAN":
            return csv_path.stem, {
                name: _float(value)
                for name, value in zip(header, row[1:])
            }
    # fallback: no MEAN row (shouldn't happen)
    return csv_path.stem, {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run output directories")
    parser.add_argument("--metrics", nargs="*", default=None, help="metric columns to show (default: key set)")
    args = parser.parse_args()

    metrics = args.metrics or KEY_METRICS

    print(f"{'run':<32}" + "".join(f"{m:>14}" for m in metrics))
    print("-" * (32 + 14 * len(metrics)))

    found_any = False
    for run_dir in args.runs:
        run_dir = run_dir.resolve()
        csv_path = latest_metrics_csv(run_dir)
        if csv_path is None:
            print(f"{run_dir.name:<32}  (no val-step=*.csv found)")
            continue
        found_any = True
        stem, mean = read_mean_row(csv_path)
        cells = []
        for m in metrics:
            value = mean.get(m)
            cells.append(f"{value:>14.4f}" if value is not None else f"{'—':>14}")
        print(f"{run_dir.name:<32}" + "".join(cells))
        print(f"  (csv: {stem})")

    if not found_any:
        print("No validation CSVs found in any run directory.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
