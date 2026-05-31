"""Study-level statistics: run metadata parsing, outlier detection, aggregation, CSV output.

Refactored from scripts/summarize_experiments.py.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from ._types import RunMeta
from ._utils import _read_json, _safe_float


# ---------------------------------------------------------------------------
# CSV writing (moved from _utils – only used here)
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(keys))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _discover_run_summaries(study_root: Path) -> list[Path]:
    """Find all run_summary.json files under a study root."""
    return sorted(study_root.glob("*/run_summary.json"))


def _default_study_outputs(study: str) -> tuple[str, str, str, str]:
    """Return default output filenames for a study."""
    study_prefix = study.lower()
    return (
        f"{study_prefix}_aggregate.csv",
        f"{study_prefix}_outliers.csv",
        f"{study_prefix}_run_issues.csv",
        f"{study_prefix}_report.json",
    )


# ---------------------------------------------------------------------------
# Run metadata parsing
# ---------------------------------------------------------------------------


def _parse_run_meta(run_name: str, params: dict[str, Any]) -> RunMeta:
    """Parse model name, sweep variable, and sweep value from run metadata.

    Handles two naming conventions:
      1. Structured:  <sweep>__sv-<path>-<value>__<token>
      2. Fallback:    infer model from class_path, sweep_var from param keys.
    """
    pattern = re.compile(
        r"^(?P<sweep_name>.+?)__sv-(?P<path>.+?)-(?P<value>.+)__(?P<token>[0-9a-f]{8})$"
    )
    m = pattern.match(run_name)
    if m:
        sweep_name = m.group("sweep_name")
        parts = sweep_name.split("_", 1)
        if len(parts) == 2 and re.fullmatch(r"q\d+", parts[0], flags=re.IGNORECASE):
            model = parts[1]
        else:
            model = sweep_name
        sweep_path = m.group("path")
        sweep_var = sweep_path.split(".")[-1]
        raw_value = m.group("value")
        parsed = _safe_float(raw_value)
        sweep_value: float | str = parsed if parsed is not None else raw_value
        return RunMeta(
            run_name=run_name,
            model=model,
            sweep_var=sweep_var,
            sweep_path=sweep_path,
            sweep_value=sweep_value,
        )

    # Fallback heuristic
    model = "unknown"
    class_path = str(
        params.get(
            "model.renderer.init_args.deform_model_config.class_path", ""
        )
    ).lower()
    if "hashgrid" in class_path:
        model = "hashgrid"
    elif "mlp" in class_path:
        model = "mlp"

    sweep_var = "unknown"
    sweep_path = "unknown"
    sweep_value: float | str = "unknown"
    for k in sorted(params.keys()):
        if (
            "max_resolution" in k
            or "t_multires" in k
            or k.endswith(".D")
            or "x_multires" in k
        ):
            sweep_var = k.split(".")[-1]
            sweep_path = k
            v = params[k]
            parsed = _safe_float(v)
            sweep_value = parsed if parsed is not None else str(v)
            break

    return RunMeta(
        run_name=run_name,
        model=model,
        sweep_var=sweep_var,
        sweep_path=sweep_path,
        sweep_value=sweep_value,
    )


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def _is_case_outlier(
    case_row: dict[str, Any],
    metrics: list[str],
    zero_as_missing: set[str],
) -> tuple[bool, list[str]]:
    """Check whether a case row is an outlier (missing or zero metric).

    Returns (is_outlier, list_of_reasons).
    """
    reasons: list[str] = []

    for metric in metrics:
        v = case_row.get(metric)
        fv = _safe_float(v)
        if fv is None:
            reasons.append(f"missing:{metric}")
            continue
        if metric in zero_as_missing and abs(fv) < 1e-12:
            reasons.append(f"zero:{metric}")

    return (len(reasons) > 0), reasons


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_valid_cases(
    valid_cases: list[dict[str, Any]],
    metrics: list[str],
) -> dict[str, Any]:
    """Compute per-metric mean, std, count across valid (non-outlier) cases."""
    out: dict[str, Any] = {}
    for metric in metrics:
        vals: list[float] = []
        for row in valid_cases:
            fv = _safe_float(row.get(metric))
            if fv is not None:
                vals.append(fv)

        if not vals:
            out[f"{metric}_mean"] = None
            out[f"{metric}_std"] = None
            out[f"{metric}_count"] = 0
            continue

        out[f"{metric}_mean"] = mean(vals)
        out[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out[f"{metric}_count"] = len(vals)
    return out


def _value_sort_key(v: Any) -> tuple[int, float | str]:
    """Sort key: numeric values first, then strings."""
    fv = _safe_float(v)
    if fv is not None:
        return (0, fv)
    return (1, str(v))


def _collect_varying_param_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Identify param_* keys that vary across rows."""
    all_keys: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k.startswith("param_"):
                all_keys.add(k)

    varying: list[str] = []
    for k in sorted(all_keys):
        values = {str(row.get(k)) for row in rows if row.get(k) is not None}
        if len(values) > 1:
            varying.append(k)
    return varying


def _unique_sorted_values(rows: list[dict[str, Any]], key: str) -> list[Any]:
    """Return unique, sorted values for a given column key."""
    seen: set[str] = set()
    vals: list[Any] = []
    for row in rows:
        v = row.get(key)
        if v is None:
            continue
        token = str(v)
        if token in seen:
            continue
        seen.add(token)
        vals.append(v)
    vals.sort(key=_value_sort_key)
    return vals


def _short_param_name(param_key: str) -> str:
    """Shorten a param_* key to its last dotted component."""
    if param_key.startswith("param_"):
        raw = param_key[len("param_"):]
    else:
        raw = param_key
    return raw.split(".")[-1]


def _fmt_axis_label(v: Any) -> str:
    """Format a value for use as an axis tick label."""
    fv = _safe_float(v)
    if fv is not None:
        if abs(fv - round(fv)) < 1e-9:
            return str(int(round(fv)))
        return f"{fv:.4g}"
    return str(v)


def _has_single_variable_run(rows: list[dict[str, Any]]) -> bool:
    """Check whether any row comes from a single-variable sweep."""
    run_names = {
        str(row.get("run_name", ""))
        for row in rows
        if str(row.get("run_name", ""))
    }
    return any("__sv-" in run_name for run_name in run_names)


# ---------------------------------------------------------------------------
# Public API: build study tables
# ---------------------------------------------------------------------------


def build_study_tables(
    study_root: Path,
    study: str,
    metrics: list[str],
    zero_as_missing: set[str],
    max_outliers_per_run: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Process all run_summary.json files and build aggregate/outlier/issue tables.

    Returns (aggregate_rows, outlier_rows, run_issue_rows).
    """
    run_files = _discover_run_summaries(study_root)

    aggregate_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    run_issue_rows: list[dict[str, Any]] = []

    for fp in run_files:
        summary = _read_json(fp)
        run_name = str(summary.get("run_name", fp.parent.name))
        params = (
            summary.get("params", {})
            if isinstance(summary.get("params", {}), dict)
            else {}
        )
        cases = (
            summary.get("cases", [])
            if isinstance(summary.get("cases", []), list)
            else []
        )

        meta = _parse_run_meta(run_name, params)

        valid_cases: list[dict[str, Any]] = []
        abnormal_count = 0
        abnormal_case_names: list[str] = []
        abnormal_case_reason_tokens: list[str] = []

        for c in cases:
            case_row = c if isinstance(c, dict) else {}
            is_outlier, reasons = _is_case_outlier(
                case_row, metrics, zero_as_missing
            )
            if is_outlier:
                abnormal_count += 1
                case_name = str(case_row.get("case_name", ""))
                abnormal_case_names.append(case_name)
                abnormal_case_reason_tokens.append(
                    f"{case_name}:{'|'.join(reasons)}"
                )
                outlier_rows.append(
                    {
                        "run_name": run_name,
                        "case_name": case_name,
                        "reasons": ";".join(reasons),
                        "status": case_row.get("status"),
                        "is_excluded": 1,
                    }
                )
            else:
                valid_cases.append(case_row)

        run_failed = abnormal_count > int(max_outliers_per_run)
        agg = (
            _aggregate_valid_cases(valid_cases, metrics)
            if not run_failed
            else _aggregate_valid_cases([], metrics)
        )

        row: dict[str, Any] = {
            "study": study,
            "run_name": run_name,
            "model": meta.model,
            "sweep_var": meta.sweep_var,
            "sweep_path": meta.sweep_path,
            "sweep_value": meta.sweep_value,
            "num_cases_total": len(cases),
            "num_cases_valid": len(valid_cases),
            "num_cases_abnormal": abnormal_count,
            "run_failed": int(run_failed),
            "failure_reason": "too_many_abnormal_cases" if run_failed else "",
            "excluded_case_names": ";".join(
                sorted(x for x in abnormal_case_names if x)
            ),
            "excluded_case_reasons": ";;".join(abnormal_case_reason_tokens),
        }
        row.update(agg)
        for k, v in params.items():
            row[f"param_{k}"] = v

        aggregate_rows.append(row)

        run_issue_rows.append(
            {
                "study": study,
                "run_name": run_name,
                "model": meta.model,
                "sweep_var": meta.sweep_var,
                "sweep_value": meta.sweep_value,
                "num_cases_total": len(cases),
                "num_cases_abnormal": abnormal_count,
                "num_cases_excluded": abnormal_count,
                "run_failed": int(run_failed),
                "failure_reason": "too_many_abnormal_cases" if run_failed else "",
                "abnormal_case_names": ";".join(
                    sorted(x for x in abnormal_case_names if x)
                ),
                "abnormal_case_reasons": ";;".join(abnormal_case_reason_tokens),
            }
        )

    # Link outliers to run failure status
    failed_run_names = {
        str(r.get("run_name", ""))
        for r in run_issue_rows
        if int(r.get("run_failed", 0)) == 1
    }
    for row in outlier_rows:
        row["triggers_run_failure"] = int(
            str(row.get("run_name", "")) in failed_run_names
        )

    # Stable sort
    aggregate_rows.sort(key=lambda r: str(r.get("run_name", "")))
    outlier_rows.sort(
        key=lambda r: (str(r.get("run_name", "")), str(r.get("case_name", "")))
    )
    run_issue_rows.sort(key=lambda r: str(r.get("run_name", "")))

    return aggregate_rows, outlier_rows, run_issue_rows


def write_study_csvs(
    study_root: Path,
    study: str,
    aggregate_rows: list[dict[str, Any]],
    outlier_rows: list[dict[str, Any]],
    run_issue_rows: list[dict[str, Any]],
    agg_csv: Path | None = None,
    outlier_csv: Path | None = None,
    run_issue_csv: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write aggregate, outlier, and run-issue tables to CSV files.

    Returns (agg_csv_path, outlier_csv_path, run_issue_csv_path).
    """
    default_agg, default_outlier, default_run_issue, _ = _default_study_outputs(
        study
    )

    _agg_csv = Path(agg_csv or default_agg)
    if not _agg_csv.is_absolute():
        _agg_csv = study_root / _agg_csv
    _outlier_csv = Path(outlier_csv or default_outlier)
    if not _outlier_csv.is_absolute():
        _outlier_csv = study_root / _outlier_csv
    _run_issue_csv = Path(run_issue_csv or default_run_issue)
    if not _run_issue_csv.is_absolute():
        _run_issue_csv = study_root / _run_issue_csv

    _write_csv(_agg_csv, aggregate_rows)
    _write_csv(_outlier_csv, outlier_rows)
    _write_csv(_run_issue_csv, run_issue_rows)

    return _agg_csv, _outlier_csv, _run_issue_csv
