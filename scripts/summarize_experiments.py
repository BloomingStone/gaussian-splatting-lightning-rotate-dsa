#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_ROOT = "outputs/experiments_asoca"
DEFAULT_OUTPUT_CSV = "master_results.csv"
DEFAULT_OUTPUT_JSON = "study_report.json"


@dataclass(frozen=True)
class FlatRunSummary:
    study: str
    run_id: str
    description: str
    num_cases: int
    num_success: int
    num_failed: int
    row: dict[str, Any]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Q1/Q2/Q3 run summaries into one table.")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--studies", default="Q1,Q2,Q3")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--primary-metric", default="3d_dice_mean")
    parser.add_argument("--secondary-metric", default="3d_hd95_mean")
    parser.add_argument("--secondary-minimize", action="store_true", help="Use lower-is-better for secondary metric")
    parser.add_argument("--write-q2-curve-csv", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return None


def _flatten_summary(summary: dict[str, Any]) -> FlatRunSummary:
    study = str(summary.get("study", ""))
    run_id = str(summary.get("run_id", ""))
    desc = str(summary.get("description", ""))
    num_cases = int(summary.get("num_cases", 0))
    num_success = int(summary.get("num_success", 0))
    num_failed = int(summary.get("num_failed", 0))

    row: dict[str, Any] = {
        "study": study,
        "run_id": run_id,
        "description": desc,
        "num_cases": num_cases,
        "num_success": num_success,
        "num_failed": num_failed,
    }

    metadata = summary.get("metadata", {})
    params = metadata.get("params", {}) if isinstance(metadata, dict) else {}
    if isinstance(params, dict):
        for k, v in params.items():
            row[f"param_{k}"] = v

    agg = summary.get("agg", {})
    agg_2d = agg.get("metrics2d", {}) if isinstance(agg, dict) else {}
    agg_3d = agg.get("metrics3d_pipeline_a", {}) if isinstance(agg, dict) else {}

    if isinstance(agg_2d, dict):
        for metric_name, stat in agg_2d.items():
            if isinstance(stat, dict):
                row[f"2d_{metric_name}_mean"] = stat.get("mean")
                row[f"2d_{metric_name}_std"] = stat.get("std")

    if isinstance(agg_3d, dict):
        for metric_name, stat in agg_3d.items():
            if isinstance(stat, dict):
                row[f"3d_{metric_name}_mean"] = stat.get("mean")
                row[f"3d_{metric_name}_std"] = stat.get("std")

    return FlatRunSummary(
        study=study,
        run_id=run_id,
        description=desc,
        num_cases=num_cases,
        num_success=num_success,
        num_failed=num_failed,
        row=row,
    )


def _discover_run_summaries(results_root: Path, studies: list[str]) -> list[Path]:
    files: list[Path] = []
    for study in studies:
        study_root = results_root / study
        if not study_root.exists():
            continue
        files.extend(sorted(study_root.glob("*/run_summary.json")))
    return files


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    fields = sorted(keys)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pick_best_runs(
    summaries: list[FlatRunSummary],
    primary_metric: str,
    secondary_metric: str,
    secondary_minimize: bool,
) -> dict[str, dict[str, Any]]:
    best_by_study: dict[str, dict[str, Any]] = {}

    groups: dict[str, list[FlatRunSummary]] = {}
    for s in summaries:
        groups.setdefault(s.study, []).append(s)

    for study, items in groups.items():
        scored: list[tuple[float, float, FlatRunSummary]] = []
        for item in items:
            p = _safe_float(item.row.get(primary_metric))
            q = _safe_float(item.row.get(secondary_metric))
            if p is None:
                continue
            if q is None:
                q = float("inf") if secondary_minimize else float("-inf")
            scored.append((p, q, item))

        if not scored:
            continue

        if secondary_minimize:
            scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        else:
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        best = scored[0][2]
        best_by_study[study] = {
            "study": study,
            "run_id": best.run_id,
            "description": best.description,
            "num_cases": best.num_cases,
            "num_success": best.num_success,
            "primary_metric": primary_metric,
            "primary_value": best.row.get(primary_metric),
            "secondary_metric": secondary_metric,
            "secondary_value": best.row.get(secondary_metric),
            "row": best.row,
        }

    return best_by_study


def _try_build_q2_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curve_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("study") != "Q2":
            continue
        ratio = row.get("param_train_ratio")
        mode = row.get("param_random_loader_mode")
        dice = row.get("3d_dice_mean")
        if ratio is None or mode is None or dice is None:
            continue
        angle_hint = None
        if str(mode) == "random-start":
            try:
                r = float(ratio)
                if abs(r - 0.8) < 1e-9:
                    angle_hint = 240
                elif abs(r - 0.5) < 1e-9:
                    angle_hint = 150
                elif abs(r - 0.3) < 1e-9:
                    angle_hint = 90
            except Exception:  # noqa: BLE001
                angle_hint = None

        curve_rows.append(
            {
                "study": "Q2",
                "random_loader_mode": mode,
                "train_ratio": ratio,
                "angle_hint_degree": angle_hint,
                "dice_mean": dice,
                "run_id": row.get("run_id"),
            }
        )

    curve_rows.sort(key=lambda x: (str(x["random_loader_mode"]), float(x["train_ratio"])))
    return curve_rows


def main() -> int:
    args = parse_args()

    results_root = Path(args.results_root).resolve()
    studies = [x.strip().upper() for x in args.studies.split(",") if x.strip()]

    run_files = _discover_run_summaries(results_root, studies)
    if not run_files:
        print(f"No run_summary.json found under {results_root}")
        return 1

    summaries: list[FlatRunSummary] = []
    for fp in run_files:
        try:
            summary = _read_json(fp)
            summaries.append(_flatten_summary(summary))
        except Exception as e:  # noqa: BLE001
            print(f"Skip invalid summary file: {fp}, error={e}")

    rows = [s.row for s in summaries]

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = results_root / output_csv

    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = results_root / output_json

    _write_csv(output_csv, rows)

    best_by_study = _pick_best_runs(
        summaries=summaries,
        primary_metric=args.primary_metric,
        secondary_metric=args.secondary_metric,
        secondary_minimize=args.secondary_minimize,
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results_root": str(results_root),
        "num_run_summaries": len(summaries),
        "studies": studies,
        "primary_metric": args.primary_metric,
        "secondary_metric": args.secondary_metric,
        "secondary_minimize": bool(args.secondary_minimize),
        "best_by_study": best_by_study,
        "master_csv": str(output_csv),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.write_q2_curve_csv:
        q2_rows = _try_build_q2_curve(rows)
        q2_csv = results_root / "q2_dice_curve.csv"
        _write_csv(q2_csv, q2_rows)
        print(f"Q2 curve CSV: {q2_csv}")

    print(f"Master CSV: {output_csv}")
    print(f"Study report: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
