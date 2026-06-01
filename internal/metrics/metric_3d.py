"""3D segmentation metric utilities (NumPy / SciPy).

Ported from ``scripts/metrics_3D/metric.py`` but using numpy/scipy instead of cupy,
since the volume rasterized from Gaussians is already a CPU numpy array.

Key difference from the original Pipeline B:
    Uses **AABB** (axis-aligned bounding box) of the GT label as ROI instead of
    dilation, which is simpler and more appropriate for vasculature evaluation.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_bool(mask: np.ndarray) -> np.ndarray:
    """Binarize a mask in-place (copy)."""
    return np.asarray(mask, dtype=bool)


def _make_vox_ball_footprint(radius_vox: int) -> np.ndarray:
    """3D ball-shaped binary structuring element with given voxel radius."""
    if radius_vox <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    r = int(radius_vox)
    zz, yy, xx = np.meshgrid(
        np.arange(-r, r + 1),
        np.arange(-r, r + 1),
        np.arange(-r, r + 1),
        indexing="ij",
    )
    return (zz ** 2 + yy ** 2 + xx ** 2) <= (r ** 2)


def _largest_component(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Keep only the largest connected component of *mask*."""
    mask = _to_bool(mask)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)
    labeled, n_comp = ndi.label(mask, structure=structure)
    if n_comp == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    best = int(np.argmax(sizes))
    return labeled == best


def _binary_closing(mask: np.ndarray, radius_vox: int) -> np.ndarray:
    """Morphological closing with a ball footprint."""
    mask = _to_bool(mask)
    if radius_vox <= 0:
        return mask
    footprint = _make_vox_ball_footprint(radius_vox)
    return ndi.binary_closing(mask, structure=footprint)


# ---------------------------------------------------------------------------
# AABB ROI (replaces dilation-based ROI from the original Pipeline B)
# ---------------------------------------------------------------------------

def get_aabb_roi(gt_label: np.ndarray) -> np.ndarray:
    """Return a boolean mask covering the axis-aligned bounding box of *gt_label*.

    The returned mask has the same shape as *gt_label*.
    """
    gt = _to_bool(gt_label)
    if not gt.any():
        return np.zeros_like(gt, dtype=bool)

    coords = np.argwhere(gt)  # (N, 3)
    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)

    roi = np.zeros_like(gt, dtype=bool)
    roi[
        min_xyz[0]: max_xyz[0] + 1,
        min_xyz[1]: max_xyz[1] + 1,
        min_xyz[2]: max_xyz[2] + 1,
    ] = True
    return roi


# ---------------------------------------------------------------------------
# Segmentation pipeline (adapted Pipeline B with AABB)
# ---------------------------------------------------------------------------

def segment_volume_with_roi(
    volume: np.ndarray,
    threshold: float,
    aabb_roi: np.ndarray,
    connectivity: int = 1,
    closing_radius_vox: int = 1,
    min_component_size_vox: int = 0,
) -> np.ndarray:
    """Threshold *volume*, restrict to *aabb_roi*, keep largest component, close.

    Parameters
    ----------
    volume: np.ndarray
        Predicted density/absorption volume (float).
    threshold: float
        Absolute threshold value.
    aabb_roi: np.ndarray
        Boolean AABB mask (same shape as *volume*).
    connectivity: int
        3D connectivity (1, 2, or 3).
    closing_radius_vox: int
        Radius (in voxels) for morphological closing.
    min_component_size_vox: int
        Minimum component size in voxels (0 = no filtering).

    Returns
    -------
    pred_mask: np.ndarray (bool)
    """
    binary = (volume > threshold) & aabb_roi
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)

    pred = _largest_component(binary, connectivity=connectivity)

    if min_component_size_vox > 0 and pred.sum() < min_component_size_vox:
        return np.zeros_like(pred, dtype=bool)

    pred = _binary_closing(pred, radius_vox=closing_radius_vox)
    return pred


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sørensen–Dice coefficient."""
    pred_b = _to_bool(pred)
    gt_b = _to_bool(gt)
    inter = int(np.logical_and(pred_b, gt_b).sum())
    denom = int(pred_b.sum() + gt_b.sum())
    if denom == 0:
        return 1.0
    return (2.0 * inter) / denom


def compute_precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Return (precision, recall)."""
    pred_b = _to_bool(pred)
    gt_b = _to_bool(gt)

    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if gt_b.sum() == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return precision, recall


def _surface_mask(mask: np.ndarray) -> np.ndarray:
    """Extract surface voxels (mask AND NOT eroded)."""
    mask = _to_bool(mask)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(
        mask,
        structure=ndi.generate_binary_structure(3, 1),
        border_value=0,
    )
    return np.logical_and(mask, ~eroded)


def _surface_distances(
    src_surface: np.ndarray,
    dst_surface: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """Distance (mm) from each src surface voxel to the nearest dst surface voxel."""
    if not src_surface.any():
        return np.array([], dtype=np.float64)
    if not dst_surface.any():
        return np.full(int(src_surface.sum()), np.inf, dtype=np.float64)

    dst_dt = ndi.distance_transform_edt(~dst_surface, sampling=spacing)
    return dst_dt[src_surface]


def compute_hd95_assd(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[float, float]:
    """Return (HD95, ASSD) in mm."""
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

    all_d = np.concatenate([d_pg, d_gp])
    hd95 = float(np.percentile(all_d, 95)) if all_d.size > 0 else 0.0

    mean_pg = float(np.mean(d_pg)) if d_pg.size > 0 else 0.0
    mean_gp = float(np.mean(d_gp)) if d_gp.size > 0 else 0.0
    assd = 0.5 * (mean_pg + mean_gp)
    return hd95, assd


def compute_cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Centerline Dice (clDice)."""
    pred_b = _to_bool(pred)
    gt_b = _to_bool(gt)

    if not pred_b.any() and not gt_b.any():
        return 1.0
    if (not pred_b.any()) ^ (not gt_b.any()):
        return 0.0

    skel_pred = skeletonize(pred_b)
    skel_gt = skeletonize(gt_b)

    skel_pred_sum = int(skel_pred.sum())
    skel_gt_sum = int(skel_gt.sum())
    if skel_pred_sum == 0 or skel_gt_sum == 0:
        return 0.0

    tprec = float(np.logical_and(skel_pred, gt_b).sum()) / skel_pred_sum
    tsens = float(np.logical_and(skel_gt, pred_b).sum()) / skel_gt_sum
    denom = tprec + tsens
    if denom == 0.0:
        return 0.0
    return (2.0 * tprec * tsens) / denom


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

_METRIC_NAMES = ("dice", "precision", "recall", "hd95", "assd", "cldice")


def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, float]:
    """Compute all six segmentation metrics for one prediction mask."""
    precision, recall = compute_precision_recall(pred, gt)
    hd95, assd = compute_hd95_assd(pred, gt, spacing)

    return {
        "dice": float(compute_dice(pred, gt)),
        "precision": float(precision),
        "recall": float(recall),
        "hd95": float(hd95),
        "assd": float(assd),
        "cldice": float(compute_cldice(pred, gt)),
    }
