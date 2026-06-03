"""3D segmentation metric utilities (MONAI / PyTorch).

Uses MONAI metrics for GPU-accelerated computation.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
from monai.losses.cldice import SoftclDiceLoss
from monai.metrics import (
    compute_average_precision,          # type: ignore[import]
    compute_confusion_matrix_metric,    # type: ignore[import]
    compute_dice,                       # type: ignore[import]
    compute_hausdorff_distance,         # type: ignore[import]
    compute_roc_auc,                    # type: ignore[import]
    get_confusion_matrix,               # type: ignore[import]
)   

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Segmentation pipeline
# ---------------------------------------------------------------------------


def segment_volume_with_roi(
    volume: torch.Tensor,
    threshold: float,
    aabb_roi: torch.Tensor,
) -> torch.Tensor:
    """Threshold *volume*, restrict to *aabb_roi*.

    *volume* and *aabb_roi* must be on the same device.
    """
    aabb = torch.as_tensor(aabb_roi, dtype=torch.bool, device=volume.device)
    return (volume > threshold) & aabb


# ---------------------------------------------------------------------------
# MONAI-based metrics
# ---------------------------------------------------------------------------

_soft_cldice_fn = SoftclDiceLoss(iter_=3, smooth=1.0)


def compute_all_metrics(
    pred: torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    spacing: tuple[float, float, float],
) -> dict[str, float]:
    """Compute segmentation metrics using MONAI.

    Args:
        pred: (D, H, W) bool tensor (may be on GPU).
        gt: (D, H, W) bool array (numpy or torch).
        spacing: Voxel spacing in mm.

    Returns:
        dict with keys: ``dice``, ``precision``, ``recall``, ``hd95``, ``cldice``.
    """
    # --- Convert gt to torch on the same device as pred ---
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt).to(device=pred.device, dtype=torch.bool)
    else:
        gt = gt.to(device=pred.device, dtype=torch.bool)

    pred = pred.to(dtype=torch.bool)

    # --- Edge cases ---
    pred_sum = pred.sum().item()
    gt_sum = gt.sum().item()

    both_empty = (pred_sum == 0) and (gt_sum == 0)
    one_empty = (pred_sum == 0) != (gt_sum == 0)

    if both_empty:
        return {
            "dice": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "hd95": 0.0,
            "cldice": 1.0,
        }

    # --- Add batch/channel dims: (1, 1, D, H, W) ---
    pred_bc = pred.unsqueeze(0).unsqueeze(0).float()
    gt_bc = gt.unsqueeze(0).unsqueeze(0).float()

    # ── Dice ──
    dice = float(compute_dice(pred_bc, gt_bc, include_background=False).item())

    # ── Precision / Recall ──
    cm = get_confusion_matrix(y_pred=pred_bc, y=gt_bc, include_background=False)
    precision = float(compute_confusion_matrix_metric("precision", cm).item())
    recall = float(compute_confusion_matrix_metric("recall", cm).item())
    # Fix NaN from zero-division
    if math.isnan(precision):
        precision = 1.0 if gt_sum == 0 else 0.0
    if math.isnan(recall):
        recall = 1.0 if pred_sum == 0 else 0.0

    # ── HD95 ──
    if one_empty:
        hd95 = float("inf")
    else:
        hd95 = float(
            compute_hausdorff_distance(
                pred_bc,
                gt_bc,
                percentile=95,
                spacing=spacing,
            ).item()
        )

    # ── Soft clDice (MONAI SoftclDiceLoss, inverted to get metric) ──
    # SoftclDiceLoss expects (B, 2, ...) one-hot: [background, foreground]
    pred_oh = torch.cat([1.0 - pred_bc, pred_bc], dim=1)
    gt_oh = torch.cat([1.0 - gt_bc, gt_bc], dim=1)
    cldice = float((1.0 - _soft_cldice_fn(gt_oh, pred_oh)).item())

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "hd95": hd95,
        "cldice": cldice,
    }


# ---------------------------------------------------------------------------
# Density-based metrics (threshold-free)
# ---------------------------------------------------------------------------


def compute_density_based_metrics(
    density: torch.Tensor,
    gt: torch.Tensor,
    roi: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute threshold-free metrics directly from density and binary label.

    Args:
        density: (D, H, W) float tensor of predicted density values.
        gt: (D, H, W) bool tensor of ground-truth binary labels.
        roi: Optional (D, H, W) bool mask restricting computation region.
             If None, all voxels are used.

    Returns:
        dict with keys: ``soft_dice``, ``roc_auc``, ``pr_auc``.
    """
    # --- Restrict to ROI if provided ---
    if roi is not None:
        roi = roi.to(dtype=torch.bool, device=density.device)
        d_flat = density[roi].float()
        g_flat = gt[roi].float()
    else:
        d_flat = density.flatten().float()
        g_flat = gt.flatten().float()

    # --- Soft Dice ---
    # Normalize density to [0, 1]
    d_min = d_flat.min()
    d_max = d_flat.max()
    denom_range = d_max - d_min
    if denom_range > 1e-8:
        I = (d_flat - d_min) / denom_range
    else:
        I = torch.zeros_like(d_flat)

    numerator = 2.0 * (I * g_flat).sum()
    denominator = (I * I).sum() + (g_flat * g_flat).sum() + 1e-8
    soft_dice = float((numerator / denominator).item())

    # --- ROC-AUC (MONAI) ---
    # compute_roc_auc expects y_pred as confidence values [N] or [N, 1],
    # y as binary labels [N] or [N, 1].
    try:
        roc_auc = float(compute_roc_auc(d_flat, g_flat))    # type: ignore
    except Exception:
        roc_auc = float("nan")

    # --- PR-AUC / Average Precision (MONAI) ---
    try:
        pr_auc = float(compute_average_precision(d_flat, g_flat))   # type: ignore
    except Exception:
        pr_auc = float("nan")

    return {
        "soft_dice": soft_dice,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
