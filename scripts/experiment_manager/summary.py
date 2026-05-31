"""Orchestration: study summary generation, statistics, and visualisation.

Provides:
  - generate_study_summary(): write study_summary.json (called from experiment pipeline)
  - run_summarize(): run full stats + plots pipeline (called from summarize CLI)
  - summarize_main(): argparse-based CLI entry point
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from ._logging import Logger
from ._types import RunSpec
from ._utils import _to_plain, _write_json
from .filesystem import _has_checkpoint
from .stats import _default_study_outputs, build_study_tables, write_study_csvs
from .visualize import generate_plots


# ---------------------------------------------------------------------------
# Study summary (called at end of experiment pipeline)
# ---------------------------------------------------------------------------


def generate_study_summary(
    run_specs: list[RunSpec],
    alias_to_canonical: dict[str, str],
    results_root: Path,
    study_name: str,
    cases: list[Path],
    logger: Logger,
) -> dict[str, Any]:
    """Write study_summary.json after all runs complete.

    Success/failure is determined by checking for checkpoints on disk.
    """
    study_root = results_root / study_name
    total = len(run_specs) * len(cases)
    success = 0

    for spec in run_specs:
        for case in cases:
            case_dir = study_root / spec.name / "cases" / case.name
            if _has_checkpoint(case_dir):
                success += 1

    summary = {
        "study": study_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_runs": len(run_specs),
        "num_cases": len(cases),
        "num_total": total,
        "num_success": success,
        "num_failed": total - success,
    }
    _write_json(study_root / "study_summary.json", summary)
    logger.info(f"Study summary: {success}/{total} cases successful")
    return summary


# ---------------------------------------------------------------------------
# Summarize pipeline (called from summarize_experiments CLI)
# ---------------------------------------------------------------------------


# Defaults (mirrored from original summarize_experiments.py)
DEFAULT_RESULTS_ROOT = "/media/F/sj/Data/ASOCA_recon"

DEFAULT_PLOT_METRICS = [
    "3d_dice",
    "3d_hd95",
    "3d_assd",
    "3d_by_gt_dice",
    "2d_psnr",
    "2d_lpips",
    "2d_ssim_loss",
    "2d_loss",
]


def run_summarize(
    results_root: Path,
    study: str,
    *,
    plot_metrics: list[str] | None = None,
    zero_as_missing_metrics: list[str] | None = None,
    max_outliers_per_run: int = 2,
    aggregate_csv: str = "",
    outlier_csv: str = "",
    run_issue_csv: str = "",
    report_json: str = "",
    plots_dir: str = "plots",
    plots_per_row: int = 5,
) -> int:
    """Run the full summarization pipeline: stats → CSV → plots → report JSON.

    Returns 0 on success, 1 on error.
    """
    study_root = results_root / study
    if not study_root.exists():
        print(f"Study root not found: {study_root}")
        return 1

    metrics = list(plot_metrics or DEFAULT_PLOT_METRICS)
    zero_as_missing = set(zero_as_missing_metrics or DEFAULT_PLOT_METRICS)

    # ── Build tables ──────────────────────────────────────────────
    aggregate_rows, outlier_rows, run_issue_rows = build_study_tables(
        study_root=study_root,
        study=study,
        metrics=metrics,
        zero_as_missing=zero_as_missing,
        max_outliers_per_run=max_outliers_per_run,
    )

    if not aggregate_rows:
        print(f"No run_summary.json found under {study_root}")
        return 1

    # ── Write CSVs ────────────────────────────────────────────────
    agg_csv_path, outlier_csv_path, run_issue_csv_path = write_study_csvs(
        study_root=study_root,
        study=study,
        aggregate_rows=aggregate_rows,
        outlier_rows=outlier_rows,
        run_issue_rows=run_issue_rows,
        agg_csv=Path(aggregate_csv) if aggregate_csv else None,
        outlier_csv=Path(outlier_csv) if outlier_csv else None,
        run_issue_csv=Path(run_issue_csv) if run_issue_csv else None,
    )

    # ── Generate plots ────────────────────────────────────────────
    _plots_dir = Path(plots_dir)
    if not _plots_dir.is_absolute():
        _plots_dir = study_root / _plots_dir
    plot_files = generate_plots(
        aggregate_rows, metrics, _plots_dir, plots_per_row
    )

    # ── Write report JSON ─────────────────────────────────────────
    _report_json = Path(report_json) if report_json else None
    if _report_json is None:
        _, _, _, default_report = _default_study_outputs(study)
        _report_json = Path(default_report)
    if not _report_json.is_absolute():
        _report_json = study_root / _report_json

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results_root": str(results_root),
        "study": study,
        "num_runs": len(aggregate_rows),
        "num_failed_runs": sum(
            1 for r in aggregate_rows if int(r.get("run_failed", 0)) == 1
        ),
        "num_outlier_cases": len(outlier_rows),
        "max_outliers_per_run": int(max_outliers_per_run),
        "metrics": metrics,
        "zero_as_missing_metrics": sorted(zero_as_missing),
        "aggregate_csv": str(agg_csv_path),
        "outlier_csv": str(outlier_csv_path),
        "run_issue_csv": str(run_issue_csv_path),
        "plots": plot_files,
    }

    _write_json(_report_json, report)

    print(f"Aggregate CSV: {agg_csv_path}")
    print(f"Outlier CSV:  {outlier_csv_path}")
    print(f"Run issue CSV: {run_issue_csv_path}")
    print(f"Report JSON:   {_report_json}")
    print(f"Plots dir:     {_plots_dir}")
    return 0


# ---------------------------------------------------------------------------
# CLI entry point (argparse, independent of Hydra)
# ---------------------------------------------------------------------------


def summarize_main(argv: list[str] | None = None) -> int:
    """Argparse-based CLI for the summarization sub-command.

    Usage: python scripts/summarize_experiments.py --study <name> [options]
    """
    parser = argparse.ArgumentParser(
        description="Aggregate study run summaries and draw trend plots."
    )
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--study", required=True)
    parser.add_argument("--aggregate-csv", default="")
    parser.add_argument("--outlier-csv", default="")
    parser.add_argument("--run-issue-csv", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--plots-per-row", type=int, default=5)
    parser.add_argument(
        "--plot-metrics",
        nargs="+",
        default=DEFAULT_PLOT_METRICS,
        help="Metric keys in case rows, e.g. 3d_dice 3d_hd95 2d_psnr 2d_lpips",
    )
    parser.add_argument(
        "--zero-as-missing-metrics",
        nargs="+",
        default=DEFAULT_PLOT_METRICS,
        help="Metrics treated as invalid when value is 0",
    )
    parser.add_argument("--max-outliers-per-run", type=int, default=2)

    args = parser.parse_args(argv)

    return run_summarize(
        results_root=Path(args.results_root).resolve(),
        study=str(args.study),
        plot_metrics=list(args.plot_metrics),
        zero_as_missing_metrics=list(args.zero_as_missing_metrics),
        max_outliers_per_run=int(args.max_outliers_per_run),
        aggregate_csv=str(args.aggregate_csv),
        outlier_csv=str(args.outlier_csv),
        run_issue_csv=str(args.run_issue_csv),
        report_json=str(args.report_json),
        plots_dir=str(args.plots_dir),
        plots_per_row=int(args.plots_per_row),
    )
