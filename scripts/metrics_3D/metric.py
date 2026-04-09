from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, cast, Literal, TYPE_CHECKING, Protocol
from skimage.morphology import skeletonize
import numpy as np
from tqdm import tqdm
import cupy as cp
from cupyx.scipy import ndimage as ndi

if TYPE_CHECKING:
    class Array3D(Protocol):
        ndim: int
        size: int
        shape: tuple[int, ...]
        def any(self) -> bool: ...
        def ravel(self) -> Array3D: ...
        def sum(self) -> int: ...
        def astype(self, dtype) -> Array3D: ...
        def get(self) -> np.ndarray: ...
        def max(self) -> float: ...
        def __array__(self, dtype=None):...
        def __and__(self, other): ...
        def __gt__(self, other): ...
        def __eq__(self, Array3D) -> Any: ...
        def __invert__(self) -> Array3D: ...
        def __getitem__(self, key) -> Array3D: ...
        def __add__(self, other) -> Array3D: ...
else:
    Array3D = cp.ndarray

Spacing3D = tuple[float, float, float]

Metrics = Literal["dice", "precision", "recall", "hd95", "assd", "cldice"]


logger = logging.getLogger(__name__)




@dataclass(frozen=True)
class FragiParams:
    scale_min: float = 0.5
    scale_max: float = 2.0
    scale_step: float = 0.5
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 15.0

@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for both segmentation and evaluation.

    Notes:
    - `threshold` must be shared by Pipeline A and Pipeline B.
    - `connectivity` in {1, 2, 3} for 3D connected components.
    """

    threshold: float = 0.0344
    connectivity: int = 1
    closing_radius_vox: int = 1
    oracle_gt_dilation_radius_mm: float = 2.0
    min_component_size_vox: int = 10*10*10
    visualize: bool = False
    use_fragi_filter: bool = False  # TODO: too slow for now, consider optimizing or removing
    fragi_params: FragiParams = field(default_factory=FragiParams)


def _validate_inputs(volume: Array3D, gt_label: Array3D, spacing: Sequence[float]) -> Spacing3D:
    if volume.ndim != 3 or gt_label.ndim != 3:
        raise ValueError("Both volume and gt_label must be 3D arrays.")
    if volume.shape != gt_label.shape:
        raise ValueError("volume and gt_label must share the same shape.")
    if len(spacing) != 3:
        raise ValueError("spacing must be length-3.")

    spacing_tuple = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    if any(s <= 0 for s in spacing_tuple):
        raise ValueError("All spacing values must be > 0.")
    return spacing_tuple


def _to_bool(mask: Array3D) -> Array3D:
    return cp.asarray(mask > 0, dtype=bool)


def _make_mm_ball_footprint(radius_mm: float, spacing: Spacing3D) -> Array3D:
    if radius_mm <= 0:
        return cp.ones((1, 1, 1), dtype=bool)

    rz = int(cp.ceil(radius_mm / spacing[0]))
    ry = int(cp.ceil(radius_mm / spacing[1]))
    rx = int(cp.ceil(radius_mm / spacing[2]))

    zz, yy, xx = cp.meshgrid(                       # type: ignore
        cp.arange(-rz, rz + 1, dtype=cp.float32),
        cp.arange(-ry, ry + 1, dtype=cp.float32),
        cp.arange(-rx, rx + 1, dtype=cp.float32),
        indexing="ij",
    )
    dist_mm_sq = (zz * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (xx * spacing[2]) ** 2
    return dist_mm_sq <= (radius_mm**2)


def _make_vox_ball_footprint(radius_vox: int) -> Array3D:
    if radius_vox <= 0:
        return cp.ones((1, 1, 1), dtype=bool)
    r = int(radius_vox)
    zz, yy, xx = cp.meshgrid(                       # type: ignore
        cp.arange(-r, r + 1),
        cp.arange(-r, r + 1),
        cp.arange(-r, r + 1),
        indexing="ij",
    )
    return (zz**2 + yy**2 + xx**2) <= (r**2)


def _largest_component(mask: Array3D, connectivity: int = 1) -> Array3D:
    mask = _to_bool(mask)
    if not mask.any():
        return cp.zeros_like(mask, dtype=bool)

    structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)
    labeled, n_comp = cast(tuple[Array3D, int], ndi.label(mask, structure=structure))
    if n_comp == 0:
        return cp.zeros_like(mask, dtype=bool)

    sizes = cp.bincount(labeled.ravel())
    sizes[0] = 0
    best = int(cp.argmax(sizes))
    return labeled == best

def _central_weight_map(shape: tuple[int, ...], max_value: float) -> Array3D:
    zz, yy, xx = cp.meshgrid(                       # type: ignore
        cp.arange(shape[0], dtype=cp.float32) - shape[0] / 2,
        cp.arange(shape[1], dtype=cp.float32) - shape[1] / 2,
        cp.arange(shape[2], dtype=cp.float32) - shape[2] / 2,
        indexing="ij",
    )
    dist_sq = zz**2 + yy**2 + xx**2
    max_dist_sq = (cp.array(shape, dtype=cp.float32) / 2).dot(cp.array(shape, dtype=cp.float32) / 2)
    weight_map = (1.0 - (dist_sq / max_dist_sq)) * max_value
    return weight_map

def _score_components_by_density_sum(
    labeled: Array3D,
    n_comp: int,
    volume: Array3D,
    min_component_size_vox: int = 0,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    if n_comp == 0:
        return scores

    labels = labeled.ravel()
    vals = volume.ravel() + _central_weight_map(volume.shape, volume.max()).ravel()
    
    counts = cp.bincount(labels, minlength=n_comp + 1)
    sums = cp.bincount(labels, weights=vals, minlength=n_comp + 1)

    for idx in range(1, n_comp + 1):
        if counts[idx] >= min_component_size_vox:
            scores[idx] = float(sums[idx]/counts[idx])
    return scores


def _binary_closing(mask: Array3D, radius_vox: int) -> Array3D:
    mask = _to_bool(mask)
    if radius_vox <= 0:
        return mask
    footprint = _make_vox_ball_footprint(radius_vox)
    return ndi.binary_closing(mask, structure=footprint)


def _optional_vesselness(volume: Array3D, config: EvaluationConfig) -> Array3D:
    if not config.use_fragi_filter:
        return volume

    try:
        from skimage.filters import frangi
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Frangi vesselness requested but unavailable: %s", exc)
        return volume

    p = config.fragi_params
    logger.info("Applying Frangi vesselness before thresholding.")
    vesselness = frangi(
        volume.get(),
        sigmas=np.arange(p.scale_min, p.scale_max, p.scale_step),  # type: ignore
        alpha=p.alpha,
        beta=p.beta,
        gamma=p.gamma,
        black_ridges=False,
    )
    return vesselness.astype(cp.float32, copy=False)


def pipeline_a_main(
    volume: Array3D,
    config: EvaluationConfig
) -> tuple[Array3D, dict[str, object]]:
    """GT-free deployable pipeline.

    Steps:
    1) Threshold
    2) Connected components
    3) Score each component by density sum
    4) Select argmax component
    5) Morphological closing
    """
    proc_volume = _optional_vesselness(volume, config)
    binary = proc_volume > config.threshold
    structure = ndi.generate_binary_structure(rank=3, connectivity=config.connectivity)
    labeled, n_comp = cast(tuple[Array3D, int], ndi.label(binary, structure=structure))

    comp_scores = _score_components_by_density_sum(
        labeled=labeled,
        n_comp=n_comp,
        volume=proc_volume,
        min_component_size_vox=config.min_component_size_vox,
    )

    if not comp_scores:
        logger.warning("Pipeline A found no valid components after thresholding.")
        selected = cp.zeros_like(binary, dtype=bool)
        best_label = 0
    else:
        best_label = max(comp_scores.keys(), key=lambda k: comp_scores[k])
        selected = labeled == best_label

    closed = _binary_closing(selected, radius_vox=config.closing_radius_vox)
    info = {
        "num_components": int(n_comp),
        "component_scores": comp_scores,
        "selected_component": int(best_label),
        "threshold": float(config.threshold),
    }
    return closed.astype(bool), info


def pipeline_b_oracle(
    volume: Array3D,
    gt_label: Array3D,
    spacing: Spacing3D,
    config: EvaluationConfig
) -> tuple[Array3D, dict[str, object]]:
    """Oracle upper bound pipeline using GT-dilated ROI (evaluation-only)."""
    logger.debug("Running Pipeline B (oracle).")

    gt = _to_bool(gt_label)
    roi_footprint = _make_mm_ball_footprint(config.oracle_gt_dilation_radius_mm, spacing)
    roi = cp.asarray(ndi.binary_dilation(gt, structure=roi_footprint), dtype=bool)

    proc_volume = _optional_vesselness(volume, config)
    binary = (proc_volume > config.threshold) & roi
    cc = _largest_component(binary, connectivity=config.connectivity)
    closed = _binary_closing(cc, radius_vox=config.closing_radius_vox)

    info = {
        "threshold": float(config.threshold),
        "roi_voxels": int(roi.sum()),
        "gt_dilation_radius_mm": float(config.oracle_gt_dilation_radius_mm),
    }
    return closed.astype(bool), info


def _dice(pred: Array3D, gt: Array3D) -> float:
    pred_b = _to_bool(pred)
    gt_b = _to_bool(gt)
    inter = int(cp.logical_and(pred_b, gt_b).sum())
    denom = int(pred_b.sum() + gt_b.sum())
    if denom == 0:
        return 1.0
    return (2.0 * inter) / denom


def _precision_recall(pred: Array3D, gt: Array3D) -> tuple[float, float]:
    pred_b = _to_bool(pred)
    gt_b = _to_bool(gt)

    tp = int(cp.logical_and(pred_b, gt_b).sum())
    fp = int(cp.logical_and(pred_b, ~gt_b).sum())
    fn = int(cp.logical_and(~pred_b, gt_b).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if gt_b.sum() == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return precision, recall


def _surface_mask(mask: Array3D) -> Array3D:
    mask = _to_bool(mask)
    if not mask.any():
        return cp.zeros_like(mask, dtype=bool)
    eroded = cp.asarray(
        ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1), border_value=0),
        dtype=bool,
    )
    return cp.logical_and(mask, ~eroded)


def _surface_distances(src_surface: Array3D, dst_surface: Array3D, spacing: Spacing3D) -> Array3D:
    if not src_surface.any():
        return cp.array([], dtype=cp.float64)
    if not dst_surface.any():
        return cp.full(int(src_surface.sum()), cp.inf, dtype=cp.float64)

    # Distance to nearest destination surface voxel in physical units.
    dst_dt = cast(Array3D, ndi.distance_transform_edt(~dst_surface, sampling=spacing))
    return dst_dt[src_surface]


def _hd95_assd(pred: Array3D, gt: Array3D, spacing: Spacing3D) -> tuple[float, float]:
    pred_b = _to_bool(pred)
    gt_b = _to_bool(gt)

    if not pred_b.any() and not gt_b.any():
        return 0.0, 0.0
    if (not pred_b.any()) ^ (not gt_b.any()):
        return float("inf"), float("inf")

    pred_s = _surface_mask(pred_b)
    gt_s = _surface_mask(gt_b)

    d_pg = _surface_distances(pred_s, gt_s, spacing)
    d_gp = _surface_distances(gt_s, pred_s, spacing)

    all_d = cp.concatenate([d_pg, d_gp])
    hd95 = float(cp.percentile(all_d, 95)) if all_d.size > 0 else 0.0

    mean_pg = float(cp.mean(d_pg)) if d_pg.size > 0 else 0.0
    mean_gp = float(cp.mean(d_gp)) if d_gp.size > 0 else 0.0
    assd = 0.5 * (mean_pg + mean_gp)
    return hd95, assd


def _cldice(pred: Array3D, gt: Array3D) -> float:
    pred_b = _to_bool(pred)
    gt_b = _to_bool(gt)

    if not pred_b.any() and not gt_b.any():
        return 1.0
    if (not pred_b.any()) ^ (not gt_b.any()):
        return 0.0

    # skimage skeletonize runs on CPU arrays.
    pred_np = cp.asnumpy(pred_b)
    gt_np = cp.asnumpy(gt_b)

    skel_pred = skeletonize(pred_np)
    skel_gt = skeletonize(gt_np)

    skel_pred_sum = int(skel_pred.sum())
    skel_gt_sum = int(skel_gt.sum())
    if skel_pred_sum == 0 or skel_gt_sum == 0:
        return 0.0

    tprec = float(np.logical_and(skel_pred, gt_np).sum()) / skel_pred_sum
    tsens = float(np.logical_and(skel_gt, pred_np).sum()) / skel_gt_sum
    denom = tprec + tsens
    if denom == 0.0:
        return 0.0
    return (2.0 * tprec * tsens) / denom


def evaluate_segmentation(
    pred: Array3D,
    gt: Array3D,
    spacing: Spacing3D
) -> dict[Metrics, float]:
    """Compute all requested metrics for one prediction mask."""
    precision, recall = _precision_recall(pred, gt)
    hd95, assd = _hd95_assd(pred, gt, spacing)

    return {
        "dice": float(_dice(pred, gt)),
        "precision": float(precision),
        "recall": float(recall),
        "hd95": float(hd95),
        "assd": float(assd),
        "cldice": float(_cldice(pred, gt)),
    }


def determine_best_threshold(
    volume: Array3D,
    gt_label: Array3D,
    spacing: Sequence[float],
    thresholds: Iterable[float],
    config: Optional[EvaluationConfig] = None,
    objective: Metrics = "dice"
) -> dict[str, Any]:
    """Evaluate Pipeline A over candidate thresholds and return best result.

    This helper is intended for future validation studies.
    It uses GT only for model selection/evaluation, never for deployable inference.
    """
    base_config = config or EvaluationConfig()
    spacing_t = _validate_inputs(volume, gt_label, spacing)

    results = []
    with tqdm(total=len(list(thresholds)), desc="Evaluating thresholds") as pbar:
        for thr in thresholds:
            cfg = EvaluationConfig(
                threshold=float(thr),
                connectivity=base_config.connectivity,
                closing_radius_vox=base_config.closing_radius_vox,
                oracle_gt_dilation_radius_mm=base_config.oracle_gt_dilation_radius_mm,
                min_component_size_vox=base_config.min_component_size_vox,
                visualize=base_config.visualize,
                use_fragi_filter=base_config.use_fragi_filter,
                fragi_params=base_config.fragi_params,
            )
            pred_a, info_a = pipeline_a_main(volume, cfg)
            if base_config.visualize:
                visualize_overlay_slices(volume, pred_a, gt_label)
            metrics_a = evaluate_segmentation(pred_a, gt_label, spacing_t)

            results.append(
                {
                    "threshold": float(thr),
                    "metrics": metrics_a,
                    "pipeline_info": info_a,
                }
            )
            pbar.update(1)
            pbar.set_postfix({"threshold": f"{thr:.3f}", objective: f"{metrics_a[objective]:.3f}"})

    if not results:
        raise ValueError("No threshold candidates provided.")

    if objective not in results[0]["metrics"]:
        raise ValueError(f"Unknown objective '{objective}'.")

    best = max(results, key=lambda x: x["metrics"][objective])
    return {
        "objective": objective,
        "best_threshold": float(best["threshold"]),
        "best_metrics": best["metrics"],
        "all_results": results,
    }


def run_evaluation(
    volume: Array3D,
    gt_label: Array3D,
    spacing: Sequence[float],
    config: Optional[EvaluationConfig] = None,
) -> tuple[dict[str, Any], Array3D, Array3D]:
    """Run both pipelines and return metrics for fair comparison."""
    cfg = config or EvaluationConfig()

    spacing_t = _validate_inputs(volume, gt_label, spacing)
    logger.debug("Starting evaluation with threshold=%.5f", cfg.threshold)

    pred_a, info_a = pipeline_a_main(volume=volume, config=cfg)
    pred_b, info_b = pipeline_b_oracle(
        volume=volume,
        gt_label=gt_label,
        spacing=spacing_t,
        config=cfg,
    )
    if cfg.visualize:
        visualize_overlay_slices(volume, pred_a, gt_label)
        visualize_overlay_slices(volume, pred_b, gt_label)

    metrics_a = evaluate_segmentation(pred_a, gt_label, spacing_t)
    metrics_b = evaluate_segmentation(pred_b, gt_label, spacing_t)
    consistency = {
        "dice_a_vs_b": float(_dice(pred_a, pred_b)),
    }

    out = {
        "config": {
            "threshold": cfg.threshold,
            "connectivity": cfg.connectivity,
            "closing_radius_vox": cfg.closing_radius_vox,
            "oracle_gt_dilation_radius_mm": cfg.oracle_gt_dilation_radius_mm,
            "min_component_size_vox": cfg.min_component_size_vox,
            "use_vesselness": cfg.use_fragi_filter,
        },
        "pipeline_a": {
            "metrics": metrics_a,
            "info": info_a,
        },
        "pipeline_b": {
            "metrics": metrics_b,
            "info": info_b,
        },
        "consistency": consistency,
    }
    return out, pred_a, pred_b


def visualize_overlay_slices(
    volume: Array3D,
    pred: Array3D,
    gt: Array3D,
    axis: int = 0,
    slice_index: Optional[int] = None,
) -> None:
    """Optional quick visualization utility for debug."""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')
    
    idx = int(slice_index if slice_index is not None else (volume.shape[axis] // 2))
    
    vol_slice = cp.take(volume, indices=idx, axis=axis).get()
    pred_slice = cp.take(pred, indices=idx, axis=axis).get()
    gt_slice = cp.take(gt, indices=idx, axis=axis).get()

    plt.figure()
    plt.imshow(vol_slice, cmap="gray")
    plt.contour(gt_slice.astype(cp.uint8), levels=[0.5], colors="lime", linewidths=1.0)
    plt.contour(pred_slice.astype(cp.uint8), levels=[0.5], colors="red", linewidths=1.0)
    plt.title(f"Axis={axis}, Slice={idx} | GT=green, Pred=red")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
