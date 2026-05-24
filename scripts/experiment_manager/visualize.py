"""Visualization: bar plots, heatmaps, and trend visualisation for study results.

Refactored from scripts/summarize_experiments.py.
Requires matplotlib (heavy dependency, isolated to this module).
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ._utils import _safe_float
from .stats import (
    _collect_varying_param_keys,
    _fmt_axis_label,
    _has_single_variable_run,
    _short_param_name,
    _unique_sorted_values,
)

# ---------------------------------------------------------------------------
# Colour configuration
# ---------------------------------------------------------------------------

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


def _model_color(model_name: str) -> str:
    """Return a consistent colour for a model name."""
    key = model_name.lower().strip()
    if key in MODEL_COLOR_OVERRIDES:
        return MODEL_COLOR_OVERRIDES[key]
    idx = sum(ord(ch) for ch in key) % len(FALLBACK_MODEL_COLORS)
    return FALLBACK_MODEL_COLORS[idx]


# ---------------------------------------------------------------------------
# Bar plots (single-variable sweeps)
# ---------------------------------------------------------------------------


def _write_bar_plots(
    rows: list[dict[str, Any]],
    metrics: list[str],
    out_dir: Path,
    plots_per_row: int,
) -> list[str]:
    """Generate grouped bar charts for single-variable sweep results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

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

            key = (
                str(row.get("model", "unknown")),
                str(row.get("sweep_var", "unknown")),
            )
            grouped.setdefault(key, []).append((x, y, y_std))

        if not grouped:
            continue

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

            x_values = sorted(
                {p[0] for points in series_by_model.values() for p in points}
            )
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
                    centered = (
                        x_positions[i]
                        - group_width / 2
                        + (model_idx + 0.5) * bar_width
                    )
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

        # Hide empty subplot slots
        for idx in range(n_facets, n_rows * n_cols):
            ax = axes[idx // n_cols][idx % n_cols]
            ax.set_visible(False)

        for r in range(n_rows):
            axes[r][0].set_ylabel(metric)

        if legend_models:
            handles = [
                Patch(facecolor=_model_color(m), label=m)
                for m in sorted(legend_models)
            ]
            fig.legend(
                handles=handles,
                loc="upper center",
                ncols=min(6, len(handles)),
                frameon=False,
            )

        fig.suptitle(f"{metric} trend by sweep", y=1.03)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        out_path = out_dir / f"trend_{metric}.png"
        fig.savefig(out_path)
        plt.close(fig)
        written.append(str(out_path))

    return written


# ---------------------------------------------------------------------------
# Heatmap plots (2D / 3D grid sweeps)
# ---------------------------------------------------------------------------


def _write_grid_heatmaps(
    rows: list[dict[str, Any]],
    metrics: list[str],
    out_dir: Path,
    plots_per_row: int,
) -> list[str]:
    """Generate heatmap visualisations for 2D and 3D grid sweep results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    valid_rows = [row for row in rows if int(row.get("run_failed", 0)) == 0]
    if not valid_rows:
        return written

    varying = _collect_varying_param_keys(valid_rows)
    dim = len(varying)
    if dim not in (2, 3):
        return written

    model_names = sorted(
        {str(row.get("model", "unknown")) for row in valid_rows}
    )

    for model in model_names:
        model_rows = [
            r for r in valid_rows if str(r.get("model", "unknown")) == model
        ]
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

        # ---- 2D grid ----
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

                matrix: list[list[float]] = [
                    [float("nan") for _ in x_vals] for _ in y_vals
                ]
                std_matrix: list[list[float]] = [
                    [0.0 for _ in x_vals] for _ in y_vals
                ]
                for (yi, xi), vals in cell.items():
                    means_list = [v[0] for v in vals]
                    stds_list = [v[1] for v in vals]
                    matrix[yi][xi] = mean(means_list)
                    std_matrix[yi][xi] = mean(stds_list)

                valid_metric_vals = [
                    v for row_vals in matrix for v in row_vals if v == v
                ]
                if not valid_metric_vals:
                    continue

                vmin = min(valid_metric_vals)
                vmax = max(valid_metric_vals)
                if abs(vmax - vmin) < 1e-12:
                    vmax = vmin + 1e-6

                fig, ax = plt.subplots(
                    figsize=(
                        max(5.0, len(x_vals) * 0.9 + 2.0),
                        max(4.0, len(y_vals) * 0.65 + 2.2),
                    ),
                    dpi=130,
                )
                im = ax.imshow(
                    matrix,
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
                cb = fig.colorbar(im, ax=ax)
                cb.set_label(metric)

                ax.set_xticks(list(range(len(x_vals))))
                ax.set_xticklabels(
                    [_fmt_axis_label(v) for v in x_vals],
                    rotation=30,
                    ha="right",
                )
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
                        ax.text(
                            xi, yi,
                            f"{mv:.3g}\n±{sv:.2g}",
                            ha="center", va="center",
                            color="white", fontsize=7,
                        )

                fig.tight_layout()
                out_path = out_dir / f"trend_{metric}__{model}__grid2d.png"
                fig.savefig(out_path)
                plt.close(fig)
                written.append(str(out_path))
            continue

        # ---- 3D grid ----
        z_key = model_varying[2]
        z_vals = _unique_sorted_values(model_rows, z_key)
        if not z_vals:
            continue

        z_idx = {str(v): i for i, v in enumerate(z_vals)}

        for metric in metrics:
            slice_cell: dict[
                int, dict[tuple[int, int], list[tuple[float, float]]]
            ] = {}
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
                if (
                    str(xv) not in x_idx
                    or str(yv) not in y_idx
                    or str(zv) not in z_idx
                ):
                    continue

                zi = z_idx[str(zv)]
                key = (y_idx[str(yv)], x_idx[str(xv)])
                slice_cell.setdefault(zi, {}).setdefault(
                    key, []
                ).append((mean_val, std_val))

            if not slice_cell:
                continue

            matrices: dict[int, list[list[float]]] = {}
            std_matrices: dict[int, list[list[float]]] = {}
            valid_metric_vals: list[float] = []
            for zi in range(len(z_vals)):
                matrix = [[float("nan") for _ in x_vals] for _ in y_vals]
                std_matrix = [[0.0 for _ in x_vals] for _ in y_vals]
                for (yi, xi), vals in slice_cell.get(zi, {}).items():
                    means_list = [v[0] for v in vals]
                    stds_list = [v[1] for v in vals]
                    matrix[yi][xi] = mean(means_list)
                    std_matrix[yi][xi] = mean(stds_list)
                matrices[zi] = matrix
                std_matrices[zi] = std_matrix
                valid_metric_vals.extend(
                    [v for row_vals in matrix for v in row_vals if v == v]
                )

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
                im = ax.imshow(
                    matrix,
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
                if im_ref is None:
                    im_ref = im

                ax.set_xticks(list(range(len(x_vals))))
                ax.set_xticklabels(
                    [_fmt_axis_label(v) for v in x_vals],
                    rotation=30,
                    ha="right",
                )
                ax.set_yticks(list(range(len(y_vals))))
                ax.set_yticklabels([_fmt_axis_label(v) for v in y_vals])
                ax.set_xlabel(_short_param_name(x_key))
                ax.set_ylabel(_short_param_name(y_key))
                ax.set_title(
                    f"{_short_param_name(z_key)}={_fmt_axis_label(zv)}"
                )

                for yi in range(len(y_vals)):
                    for xi in range(len(x_vals)):
                        mv = matrix[yi][xi]
                        if mv != mv:
                            continue
                        sv = std_matrix[yi][xi]
                        ax.text(
                            xi, yi,
                            f"{mv:.3g}\n±{sv:.2g}",
                            ha="center", va="center",
                            color="white", fontsize=6,
                        )

            for idx in range(len(z_vals), n_rows * n_cols):
                axes[idx // n_cols][idx % n_cols].set_visible(False)

            if im_ref is not None:
                fig.colorbar(
                    im_ref,
                    ax=axes.ravel().tolist(),
                    fraction=0.02,
                    pad=0.02,
                    label=metric,
                )
            fig.suptitle(
                f"{metric} heatmap slices | model={model}", y=1.02
            )
            fig.tight_layout(rect=(0, 0, 1, 0.98))

            out_path = out_dir / f"trend_{metric}__{model}__grid3d.png"
            fig.savefig(out_path)
            plt.close(fig)
            written.append(str(out_path))

    return written


def _write_plots(
    rows: list[dict[str, Any]],
    metrics: list[str],
    out_dir: Path,
    plots_per_row: int,
) -> list[str]:
    """Orchestrate plot generation: bar charts for single-variable,
    heatmaps for 2D/3D grids."""
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_plots(
    aggregate_rows: list[dict[str, Any]],
    metrics: list[str],
    plots_dir: Path,
    plots_per_row: int = 5,
) -> list[str]:
    """Generate all trend plots for a study and return the list of output paths."""
    return _write_plots(aggregate_rows, metrics, plots_dir, plots_per_row)
