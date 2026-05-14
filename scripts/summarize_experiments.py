#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DEFAULT_RESULTS_ROOT = "/media/data2/sj/Data/ASOCA_recon"

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


MODEL_COLOR_OVERRIDES: dict[str, str] = {
    "mlp": "#d55e00",
    "hashgrid": "#0072b2",
}


FALLBACK_MODEL_COLORS = [
    "#009e73",
    "#cc79a7",
    "#56b4e9",
    "#e69f00",
    "#f0e442",
    "#000000",
]


@dataclass(frozen=True)
class RunMeta:
    run_name: str
    model: str
    sweep_var: str
    sweep_path: str
    sweep_value: float | str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate study run summaries and draw trend plots.")
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
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _discover_run_summaries(study_root: Path) -> list[Path]:
    return sorted(study_root.glob("*/run_summary.json"))


def _default_study_outputs(study: str) -> tuple[str, str, str, str]:
    study_prefix = study.lower()
    return (
        f"{study_prefix}_aggregate.csv",
        f"{study_prefix}_outliers.csv",
        f"{study_prefix}_run_issues.csv",
        f"{study_prefix}_report.json",
    )


def _parse_run_meta(run_name: str, params: dict[str, Any]) -> RunMeta:
    pattern = re.compile(r"^(?P<sweep_name>.+?)__sv-(?P<path>.+?)-(?P<value>.+)__(?P<token>[0-9a-f]{8})$")
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

    model = "unknown"
    class_path = str(params.get("model.renderer.init_args.deform_model_config.class_path", "")).lower()
    if "hashgrid" in class_path:
        model = "hashgrid"
    elif "mlp" in class_path:
        model = "mlp"

    sweep_var = "unknown"
    sweep_path = "unknown"
    sweep_value: float | str = "unknown"
    for k in sorted(params.keys()):
        if "max_resolution" in k or "t_multires" in k or k.endswith(".D") or "t_multires" in k or "x_multires" in k:
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


def _is_case_outlier(case_row: dict[str, Any], metrics: list[str], zero_as_missing: set[str]) -> tuple[bool, list[str]]:
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


def _aggregate_valid_cases(valid_cases: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
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
    fv = _safe_float(v)
    if fv is not None:
        return (0, fv)
    return (1, str(v))


def _collect_varying_param_keys(rows: list[dict[str, Any]]) -> list[str]:
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
    if param_key.startswith("param_"):
        raw = param_key[len("param_") :]
    else:
        raw = param_key
    return raw.split(".")[-1]


def _fmt_axis_label(v: Any) -> str:
    fv = _safe_float(v)
    if fv is not None:
        if abs(fv - round(fv)) < 1e-9:
            return str(int(round(fv)))
        return f"{fv:.4g}"
    return str(v)


def _has_single_variable_run(rows: list[dict[str, Any]]) -> bool:
    run_names = {str(row.get("run_name", "")) for row in rows if str(row.get("run_name", ""))}
    return any("__sv-" in run_name for run_name in run_names)


def _write_bar_plots(rows: list[dict[str, Any]], metrics: list[str], out_dir: Path, plots_per_row: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def _model_color(model_name: str) -> str:
        key = model_name.lower().strip()
        if key in MODEL_COLOR_OVERRIDES:
            return MODEL_COLOR_OVERRIDES[key]
        idx = sum(ord(ch) for ch in key) % len(FALLBACK_MODEL_COLORS)
        return FALLBACK_MODEL_COLORS[idx]

    for metric in metrics:
        grouped: dict[tuple[str, str], list[tuple[float, float, float]]] = {}

        for row in rows:
            if int(row.get("run_failed", 0)) == 1:
                continue
            x = _safe_float(row.get("sweep_value"))
            y = _safe_float(row.get(f"{metric}_mean"))
            y_std = _safe_float(row.get(f"{metric}_std"))
            if x is None or y is None:
                continue
            if y_std is None or y_std < 0:
                y_std = 0.0

            key = (str(row.get("model", "unknown")), str(row.get("sweep_var", "unknown")))
            grouped.setdefault(key, []).append((x, y, y_std))

        if not grouped:
            continue

        # Build sweep-var facets with grouped bars and std error bars.
        facet_data: dict[str, dict[str, list[tuple[float, float, float]]]] = {}
        for (model, sweep_var), points in grouped.items():
            sorted_points = sorted(points, key=lambda p: p[0])
            facet_data.setdefault(sweep_var, {})[model] = sorted_points

        if not facet_data:
            continue

        sweep_vars = sorted(facet_data.keys())
        n_facets = len(sweep_vars)
        n_cols = min(max(1, int(plots_per_row)), n_facets)
        n_rows = ceil(n_facets / n_cols)

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(4 * n_cols, 5 * n_rows),
            dpi=130,
            sharey=False,
            squeeze=False,
        )

        legend_models: set[str] = set()

        for idx, sweep_var in enumerate(sweep_vars):
            ax = axes[idx // n_cols][idx % n_cols]
            series_by_model = facet_data[sweep_var]
            model_names = sorted(series_by_model.keys())

            facet_lowers: list[float] = []
            facet_uppers: list[float] = []
            for points in series_by_model.values():
                facet_lowers.extend((p[1] - p[2]) for p in points)
                facet_uppers.extend((p[1] + p[2]) for p in points)
            if facet_lowers and facet_uppers:
                local_min = min(facet_lowers)
                local_max = max(facet_uppers)
                if abs(local_max - local_min) < 1e-12:
                    local_pad = max(1e-3, abs(local_max) * 0.05 + 1e-3)
                else:
                    local_pad = (local_max - local_min) * 0.08
                y_low = local_min - local_pad
                y_high = local_max + local_pad
            else:
                y_low, y_high = 0.0, 1.0

            x_values = sorted({p[0] for points in series_by_model.values() for p in points})
            if not x_values:
                continue

            x_positions = list(range(len(x_values)))
            x_to_idx = {v: i for i, v in enumerate(x_values)}
            n_models = max(1, len(model_names))
            group_width = 0.82
            bar_width = group_width / n_models

            for model_idx, model in enumerate(model_names):
                points = series_by_model[model]
                means: list[float | None] = [None] * len(x_values)
                stds: list[float] = [0.0] * len(x_values)
                for x_val, y_val, y_std in points:
                    idx_in_group = x_to_idx[x_val]
                    means[idx_in_group] = y_val
                    stds[idx_in_group] = y_std

                bar_xs: list[float] = []
                bar_heights: list[float] = []
                bar_errs: list[float] = []
                for i, m in enumerate(means):
                    if m is None:
                        continue
                    centered = x_positions[i] - group_width / 2 + (model_idx + 0.5) * bar_width
                    bar_xs.append(centered)
                    bar_heights.append(m)
                    bar_errs.append(stds[i])

                if bar_xs:
                    ax.bar(
                        bar_xs,
                        bar_heights,
                        width=bar_width,
                        yerr=bar_errs,
                        capsize=3,
                        alpha=0.9,
                        color=_model_color(model),
                    )
                    legend_models.add(model)

            x_labels: list[str] = []
            for v in x_values:
                if abs(v - round(v)) < 1e-9:
                    x_labels.append(str(int(round(v))))
                else:
                    x_labels.append(f"{v:.4g}")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=25, ha="right")

            ax.set_title(f"sweep: {sweep_var}")
            ax.set_xlabel("sweep value")
            ax.set_ylim(y_low, y_high)
            ax.grid(True, alpha=0.25)

        # Hide empty subplot slots when facet count is not a multiple of grid width.
        for idx in range(n_facets, n_rows * n_cols):
            ax = axes[idx // n_cols][idx % n_cols]
            ax.set_visible(False)

        for r in range(n_rows):
            axes[r][0].set_ylabel(metric)

        if legend_models:
            handles = [Patch(facecolor=_model_color(m), label=m) for m in sorted(legend_models)]
            fig.legend(handles=handles, loc="upper center", ncols=min(6, len(handles)), frameon=False)

        fig.suptitle(f"{metric} trend by sweep", y=1.03)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        out_path = out_dir / f"trend_{metric}.png"
        fig.savefig(out_path)
        plt.close(fig)
        written.append(str(out_path))

    return written


def _write_grid_heatmaps(rows: list[dict[str, Any]], metrics: list[str], out_dir: Path, plots_per_row: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    valid_rows = [row for row in rows if int(row.get("run_failed", 0)) == 0]
    if not valid_rows:
        return written

    varying = _collect_varying_param_keys(valid_rows)
    dim = len(varying)
    if dim not in (2, 3):
        return written

    model_names = sorted({str(row.get("model", "unknown")) for row in valid_rows})

    for model in model_names:
        model_rows = [r for r in valid_rows if str(r.get("model", "unknown")) == model]
        if not model_rows:
            continue

        model_varying = _collect_varying_param_keys(model_rows)
        if len(model_varying) != dim:
            continue

        x_key = model_varying[0]
        y_key = model_varying[1]
        x_vals = _unique_sorted_values(model_rows, x_key)
        y_vals = _unique_sorted_values(model_rows, y_key)
        if not x_vals or not y_vals:
            continue

        x_idx = {str(v): i for i, v in enumerate(x_vals)}
        y_idx = {str(v): i for i, v in enumerate(y_vals)}

        if dim == 2:
            for metric in metrics:
                cell: dict[tuple[int, int], list[tuple[float, float]]] = {}
                for row in model_rows:
                    mean_val = _safe_float(row.get(f"{metric}_mean"))
                    std_val = _safe_float(row.get(f"{metric}_std"))
                    if mean_val is None:
                        continue
                    if std_val is None or std_val < 0:
                        std_val = 0.0
                    xv = row.get(x_key)
                    yv = row.get(y_key)
                    if xv is None or yv is None:
                        continue
                    if str(xv) not in x_idx or str(yv) not in y_idx:
                        continue
                    key = (y_idx[str(yv)], x_idx[str(xv)])
                    cell.setdefault(key, []).append((mean_val, std_val))

                if not cell:
                    continue

                matrix: list[list[float]] = [[float("nan") for _ in x_vals] for _ in y_vals]
                std_matrix: list[list[float]] = [[0.0 for _ in x_vals] for _ in y_vals]
                for (yi, xi), vals in cell.items():
                    means = [v[0] for v in vals]
                    stds = [v[1] for v in vals]
                    matrix[yi][xi] = mean(means)
                    std_matrix[yi][xi] = mean(stds)

                valid_metric_vals = [v for row_vals in matrix for v in row_vals if v == v]
                if not valid_metric_vals:
                    continue

                vmin = min(valid_metric_vals)
                vmax = max(valid_metric_vals)
                if abs(vmax - vmin) < 1e-12:
                    vmax = vmin + 1e-6

                fig, ax = plt.subplots(
                    figsize=(max(5.0, len(x_vals) * 0.9 + 2.0), max(4.0, len(y_vals) * 0.65 + 2.2)),
                    dpi=130,
                )
                im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
                cb = fig.colorbar(im, ax=ax)
                cb.set_label(metric)

                ax.set_xticks(list(range(len(x_vals))))
                ax.set_xticklabels([_fmt_axis_label(v) for v in x_vals], rotation=30, ha="right")
                ax.set_yticks(list(range(len(y_vals))))
                ax.set_yticklabels([_fmt_axis_label(v) for v in y_vals])
                ax.set_xlabel(_short_param_name(x_key))
                ax.set_ylabel(_short_param_name(y_key))
                ax.set_title(f"{metric} heatmap | model={model}")

                for yi in range(len(y_vals)):
                    for xi in range(len(x_vals)):
                        mv = matrix[yi][xi]
                        if mv != mv:
                            continue
                        sv = std_matrix[yi][xi]
                        ax.text(xi, yi, f"{mv:.3g}\n±{sv:.2g}", ha="center", va="center", color="white", fontsize=7)

                fig.tight_layout()
                out_path = out_dir / f"trend_{metric}__{model}__grid2d.png"
                fig.savefig(out_path)
                plt.close(fig)
                written.append(str(out_path))
            continue

        z_key = model_varying[2]
        z_vals = _unique_sorted_values(model_rows, z_key)
        if not z_vals:
            continue

        z_idx = {str(v): i for i, v in enumerate(z_vals)}

        for metric in metrics:
            slice_cell: dict[int, dict[tuple[int, int], list[tuple[float, float]]]] = {}
            for row in model_rows:
                mean_val = _safe_float(row.get(f"{metric}_mean"))
                std_val = _safe_float(row.get(f"{metric}_std"))
                if mean_val is None:
                    continue
                if std_val is None or std_val < 0:
                    std_val = 0.0

                xv = row.get(x_key)
                yv = row.get(y_key)
                zv = row.get(z_key)
                if xv is None or yv is None or zv is None:
                    continue
                if str(xv) not in x_idx or str(yv) not in y_idx or str(zv) not in z_idx:
                    continue

                zi = z_idx[str(zv)]
                key = (y_idx[str(yv)], x_idx[str(xv)])
                slice_cell.setdefault(zi, {}).setdefault(key, []).append((mean_val, std_val))

            if not slice_cell:
                continue

            matrices: dict[int, list[list[float]]] = {}
            std_matrices: dict[int, list[list[float]]] = {}
            valid_metric_vals: list[float] = []
            for zi in range(len(z_vals)):
                matrix = [[float("nan") for _ in x_vals] for _ in y_vals]
                std_matrix = [[0.0 for _ in x_vals] for _ in y_vals]
                for (yi, xi), vals in slice_cell.get(zi, {}).items():
                    means = [v[0] for v in vals]
                    stds = [v[1] for v in vals]
                    matrix[yi][xi] = mean(means)
                    std_matrix[yi][xi] = mean(stds)
                matrices[zi] = matrix
                std_matrices[zi] = std_matrix
                valid_metric_vals.extend([v for row_vals in matrix for v in row_vals if v == v])

            if not valid_metric_vals:
                continue

            vmin = min(valid_metric_vals)
            vmax = max(valid_metric_vals)
            if abs(vmax - vmin) < 1e-12:
                vmax = vmin + 1e-6

            n_cols = min(max(1, int(plots_per_row)), len(z_vals))
            n_rows = ceil(len(z_vals) / n_cols)
            fig, axes = plt.subplots(
                nrows=n_rows,
                ncols=n_cols,
                figsize=(max(5.0, n_cols * 3.6), max(4.0, n_rows * 3.2)),
                dpi=130,
                squeeze=False,
            )

            im_ref = None
            for zi, zv in enumerate(z_vals):
                ax = axes[zi // n_cols][zi % n_cols]
                matrix = matrices[zi]
                std_matrix = std_matrices[zi]
                im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
                if im_ref is None:
                    im_ref = im

                ax.set_xticks(list(range(len(x_vals))))
                ax.set_xticklabels([_fmt_axis_label(v) for v in x_vals], rotation=30, ha="right")
                ax.set_yticks(list(range(len(y_vals))))
                ax.set_yticklabels([_fmt_axis_label(v) for v in y_vals])
                ax.set_xlabel(_short_param_name(x_key))
                ax.set_ylabel(_short_param_name(y_key))
                ax.set_title(f"{_short_param_name(z_key)}={_fmt_axis_label(zv)}")

                for yi in range(len(y_vals)):
                    for xi in range(len(x_vals)):
                        mv = matrix[yi][xi]
                        if mv != mv:
                            continue
                        sv = std_matrix[yi][xi]
                        ax.text(xi, yi, f"{mv:.3g}\n±{sv:.2g}", ha="center", va="center", color="white", fontsize=6)

            for idx in range(len(z_vals), n_rows * n_cols):
                axes[idx // n_cols][idx % n_cols].set_visible(False)

            if im_ref is not None:
                fig.colorbar(im_ref, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label=metric)
            fig.suptitle(f"{metric} heatmap slices | model={model}", y=1.02)
            fig.tight_layout(rect=(0, 0, 1, 0.98))

            out_path = out_dir / f"trend_{metric}__{model}__grid3d.png"
            fig.savefig(out_path)
            plt.close(fig)
            written.append(str(out_path))

    return written


def _write_plots(rows: list[dict[str, Any]], metrics: list[str], out_dir: Path, plots_per_row: int) -> list[str]:
    valid_rows = [row for row in rows if int(row.get("run_failed", 0)) == 0]
    if not valid_rows:
        return []

    if _has_single_variable_run(valid_rows):
        return _write_bar_plots(rows, metrics, out_dir, plots_per_row)

    varying = _collect_varying_param_keys(valid_rows)
    dim = len(varying)

    if dim in (2, 3):
        return _write_grid_heatmaps(rows, metrics, out_dir, plots_per_row)
    if dim > 3:
        print(f"Skip plotting: detected grid dimension={dim} (>3)")
        return []
    return _write_bar_plots(rows, metrics, out_dir, plots_per_row)


def main() -> int:
    args = parse_args()

    results_root = Path(args.results_root).resolve()
    study = str(args.study)
    study_root = results_root / study

    if not study_root.exists():
        print(f"Study root not found: {study_root}")
        return 1

    run_files = _discover_run_summaries(study_root)
    if not run_files:
        print(f"No run_summary.json found under {study_root}")
        return 1

    metrics = [str(x) for x in args.plot_metrics]
    zero_as_missing = set(str(x) for x in args.zero_as_missing_metrics)

    aggregate_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    run_issue_rows: list[dict[str, Any]] = []

    for fp in run_files:
        summary = _read_json(fp)
        run_name = str(summary.get("run_name", fp.parent.name))
        params = summary.get("params", {}) if isinstance(summary.get("params", {}), dict) else {}
        cases = summary.get("cases", []) if isinstance(summary.get("cases", []), list) else []

        meta = _parse_run_meta(run_name, params)

        valid_cases: list[dict[str, Any]] = []
        abnormal_count = 0
        abnormal_case_names: list[str] = []
        abnormal_case_reason_tokens: list[str] = []

        for c in cases:
            case_row = c if isinstance(c, dict) else {}
            is_outlier, reasons = _is_case_outlier(case_row, metrics, zero_as_missing)
            if is_outlier:
                abnormal_count += 1
                case_name = str(case_row.get("case_name", ""))
                abnormal_case_names.append(case_name)
                abnormal_case_reason_tokens.append(f"{case_name}:{'|'.join(reasons)}")
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

        run_failed = abnormal_count > int(args.max_outliers_per_run)
        agg = _aggregate_valid_cases(valid_cases, metrics) if not run_failed else _aggregate_valid_cases([], metrics)

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
            "excluded_case_names": ";".join(sorted(x for x in abnormal_case_names if x)),
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
                "abnormal_case_names": ";".join(sorted(x for x in abnormal_case_names if x)),
                "abnormal_case_reasons": ";;".join(abnormal_case_reason_tokens),
            }
        )

    failed_run_names = {str(r.get("run_name", "")) for r in run_issue_rows if int(r.get("run_failed", 0)) == 1}
    for row in outlier_rows:
        row["triggers_run_failure"] = int(str(row.get("run_name", "")) in failed_run_names)

    aggregate_rows.sort(key=lambda r: str(r.get("run_name", "")))
    outlier_rows.sort(key=lambda r: (str(r.get("run_name", "")), str(r.get("case_name", ""))))
    run_issue_rows.sort(key=lambda r: str(r.get("run_name", "")))

    default_agg_csv, default_outlier_csv, default_run_issue_csv, default_report_json = _default_study_outputs(study)

    agg_csv = Path(args.aggregate_csv or default_agg_csv)
    if not agg_csv.is_absolute():
        agg_csv = study_root / agg_csv
    outlier_csv = Path(args.outlier_csv or default_outlier_csv)
    if not outlier_csv.is_absolute():
        outlier_csv = study_root / outlier_csv
    run_issue_csv = Path(args.run_issue_csv or default_run_issue_csv)
    if not run_issue_csv.is_absolute():
        run_issue_csv = study_root / run_issue_csv
    report_json = Path(args.report_json or default_report_json)
    if not report_json.is_absolute():
        report_json = study_root / report_json
    plots_dir = Path(args.plots_dir)
    if not plots_dir.is_absolute():
        plots_dir = study_root / plots_dir

    _write_csv(agg_csv, aggregate_rows)
    _write_csv(outlier_csv, outlier_rows)
    _write_csv(run_issue_csv, run_issue_rows)
    plot_files = _write_plots(aggregate_rows, metrics, plots_dir, int(args.plots_per_row))

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results_root": str(results_root),
        "study": study,
        "num_runs": len(aggregate_rows),
        "num_failed_runs": sum(1 for r in aggregate_rows if int(r.get("run_failed", 0)) == 1),
        "num_outlier_cases": len(outlier_rows),
        "max_outliers_per_run": int(args.max_outliers_per_run),
        "metrics": metrics,
        "zero_as_missing_metrics": sorted(zero_as_missing),
        "aggregate_csv": str(agg_csv),
        "outlier_csv": str(outlier_csv),
        "run_issue_csv": str(run_issue_csv),
        "plots": plot_files,
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Aggregate CSV: {agg_csv}")
    print(f"Outlier CSV: {outlier_csv}")
    print(f"Run issue CSV: {run_issue_csv}")
    print(f"Report JSON: {report_json}")
    print(f"Plots dir: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
