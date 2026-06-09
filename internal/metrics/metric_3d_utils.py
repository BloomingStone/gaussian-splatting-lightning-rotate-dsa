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
from scipy.spatial import KDTree

# skeletonize 兼容处理
try:
    from skimage.morphology import skeletonize
    HAS_SKELETON = True
except Exception:
    HAS_SKELETON = False


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


# ---------------------------------------------------------------------------
# Helper: 3D patch-wise SSIM (torch)
# ---------------------------------------------------------------------------


def ssim_3d_patchwise_strict(
    pred: torch.Tensor,
    gt: torch.Tensor,
    patch_size: int = 7,
    stride: int = 3,
    min_fg_voxels: int = 20,
    data_range: float = 1.0,
) -> float:
    """3D patch SSIM for binary volumes, torch implementation.

    Args:
        pred: (D, H, W) float tensor (values 0/1).
        gt: (D, H, W) float tensor (values 0/1).

    Returns:
        Mean SSIM over foreground patches.
    """
    assert pred.shape == gt.shape, "pred 和 gt shape 必须一致"

    # Normalize to [0, 1]
    def _normalize(v: torch.Tensor) -> torch.Tensor:
        v_min, v_max = v.min(), v.max()
        if v_max > v_min:
            v = (v - v_min) / (v_max - v_min)
        return v

    pred = _normalize(pred)
    gt = _normalize(gt)

    D, H, W = pred.shape
    ps = patch_size

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_vals: list[float] = []

    for z in range(0, D - ps + 1, stride):
        for y in range(0, H - ps + 1, stride):
            for x in range(0, W - ps + 1, stride):
                p_patch = pred[z : z + ps, y : y + ps, x : x + ps]
                g_patch = gt[z : z + ps, y : y + ps, x : x + ps]

                if g_patch.sum() < min_fg_voxels:
                    continue

                mu_x = p_patch.mean()
                mu_y = g_patch.mean()
                sigma_x2 = p_patch.var()
                sigma_y2 = g_patch.var()
                sigma_xy = ((p_patch - mu_x) * (g_patch - mu_y)).mean()

                num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
                den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)

                val = (num / den).item() if den != 0 else 0.0
                ssim_vals.append(val)

    return float(np.mean(ssim_vals)) if ssim_vals else 0.0


# ---------------------------------------------------------------------------
# Helper: point-cloud / geometry-level metrics
# ---------------------------------------------------------------------------


def _nonzero_points(t: torch.Tensor) -> np.ndarray:
    """Return (N, 3) numpy array of non-zero voxel coordinates."""
    return t.nonzero().cpu().numpy().astype(np.float64)


def chamfer_distance(
    pred: torch.Tensor,
    gt: torch.Tensor,
    trim_ratio: float = 0.95,
) -> float:
    """Chamfer distance squared (mean) between two binary volumes."""
    pred_pts = _nonzero_points(pred)
    gt_pts = _nonzero_points(gt)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")

    tree_pred = KDTree(pred_pts)
    tree_gt = KDTree(gt_pts)

    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)

    if trim_ratio < 1.0:
        keep_p = max(1, int(len(d_pred_to_gt) * trim_ratio))
        keep_g = max(1, int(len(d_gt_to_pred) * trim_ratio))
        d_pred_to_gt = np.sort(d_pred_to_gt)[:keep_p]
        d_gt_to_pred = np.sort(d_gt_to_pred)[:keep_g]

    return float(np.mean(d_pred_to_gt**2) + np.mean(d_gt_to_pred**2))


def point_cloud_precision_recall_f1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 1.0,
) -> tuple[float, float, float]:
    """Point-cloud precision, recall, F1 at a given distance threshold."""
    pred_pts = _nonzero_points(pred)
    gt_pts = _nonzero_points(gt)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan"), float("nan"), float("nan")

    tree_gt = KDTree(gt_pts)
    tree_pred = KDTree(pred_pts)

    d_pred, _ = tree_gt.query(pred_pts)
    d_gt, _ = tree_pred.query(gt_pts)

    precision = float(np.mean(d_pred <= threshold))
    recall = float(np.mean(d_gt <= threshold))
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)

    return precision, recall, f1


# ---------------------------------------------------------------------------
# Helper: skeleton-based metrics
# ---------------------------------------------------------------------------


def skeleton_chamfer_distance(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Chamfer distance computed on skeletonized volumes."""
    if not HAS_SKELETON:
        return float("nan")

    pred_np = pred.cpu().numpy().astype(np.uint8)
    gt_np = gt.cpu().numpy().astype(np.uint8)

    if pred_np.sum() == 0 or gt_np.sum() == 0:
        return float("nan")

    pred_skel = skeletonize(pred_np > 0)
    gt_skel = skeletonize(gt_np > 0)

    pred_pts = np.array(np.nonzero(pred_skel)).T.astype(np.float64)
    gt_pts = np.array(np.nonzero(gt_skel)).T.astype(np.float64)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")

    tree_pred = KDTree(pred_pts)
    tree_gt = KDTree(gt_pts)

    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)

    return float(np.mean(d_pred_to_gt**2) + np.mean(d_gt_to_pred**2))


def cr_skeleton(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
    """Completeness Ratio on GT skeleton: fraction of GT skeleton voxels covered by pred."""
    if not HAS_SKELETON:
        return float("nan")

    gt_np = gt.cpu().numpy().astype(np.uint8)

    if gt_np.sum() == 0:
        return float("nan")

    gt_skel = skeletonize(gt_np > 0)
    skel_total = gt_skel.sum()

    if skel_total == 0:
        return float("nan")

    pred_np = pred.cpu().numpy().astype(np.uint8)
    skel_intersection = (pred_np * gt_skel).sum()

    return float((skel_intersection + eps) / (skel_total + eps))


# ---------------------------------------------------------------------------
# Main metrics entrypoint
# ---------------------------------------------------------------------------


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
            "iou": 1.0,
            "vol_ratio": 1.0,
            "ssim3d": 1.0,
            "chamfer": 0.0,
            "pc_precision": 1.0,
            "pc_recall": 1.0,
            "pc_f1": 1.0,
            "skel_chamfer": 0.0,
            "cr_skel": 1.0,
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

    # --- IoU (using MONAI confusion matrix, "ts" = threat score = IoU in binary) ---
    iou = float(compute_confusion_matrix_metric("ts", cm).item())
    if math.isnan(iou):
        iou = 1.0 if both_empty else 0.0

    # --- Volume ratio ---
    vol_ratio = float((pred_sum + 1e-6) / (gt_sum + 1e-6))

    # --- SSIM 3D (on the original (D, H, W) tensors) ---
    ssim3d = ssim_3d_patchwise_strict(
        pred.float(),
        gt.float(),
        patch_size=7,
        stride=3,
        min_fg_voxels=20,
        data_range=1.0,
    )

    # --- Chamfer distance ---
    chamfer = chamfer_distance(pred, gt)

    # --- Point-cloud Precision / Recall / F1 ---
    pc_p, pc_r, pc_f1 = point_cloud_precision_recall_f1(pred, gt, threshold=1.0)

    # --- Skeleton metrics ---
    skel_cd = skeleton_chamfer_distance(pred, gt)
    cr_skel_val = cr_skeleton(pred, gt)

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "hd95": hd95,
        "cldice": cldice,
        "iou": iou,
        "vol_ratio": vol_ratio,
        "ssim3d": ssim3d,
        "chamfer": chamfer,
        "pc_precision": pc_p,
        "pc_recall": pc_r,
        "pc_f1": pc_f1,
        "skel_chamfer": skel_cd,
        "cr_skel": cr_skel_val,
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
