"""Compute 3D segmentation metrics for an exported density NIfTI vs the label.

Both inputs are assumed to be in the SAME NIfTI order / centering affine
(gaussians_to_volume_by_Rasterizer already applied the permute/flip), so no
extra alignment is needed.

Run in the GS-contrast-flow environment.
"""
import argparse

import numpy as np
import nibabel as nib
from scipy.spatial import KDTree
from skimage.morphology import skeletonize


def dice(pred, gt):
    inter = int((pred & gt).sum())
    total = int(pred.sum()) + int(gt.sum())
    return 2.0 * inter / total if total > 0 else 0.0


def hd95(pred, gt, spacing):
    a = np.argwhere(pred).astype(np.float64) * np.array(spacing)
    b = np.argwhere(gt).astype(np.float64) * np.array(spacing)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    ta, tb = KDTree(a), KDTree(b)
    d = np.concatenate([ta.query(b)[0], tb.query(a)[0]])
    return float(np.percentile(d, 95))


def cldice(pred, gt, sk_gt):
    sk_pred = skeletonize(pred, method="lee") > 0
    tprec, tsens = int(sk_pred.sum()), int(sk_gt.sum())
    if tprec == 0 or tsens == 0:
        return float("nan")
    prec = int((sk_pred & gt).sum()) / tprec
    recall = int((sk_gt & pred).sum()) / tsens
    return 2.0 * prec * recall / (prec + recall)


def density_metrics(vol, gt):
    y = gt.ravel().astype(np.int64)
    x = vol.ravel().astype(np.float64)
    if y.sum() == 0 or y.sum() == y.size:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "soft_dice": float("nan")}
    order = np.argsort(x)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(1, x.size + 1)
    n_pos, n_neg = int(y.sum()), y.size - int(y.sum())
    roc_auc = (rank[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    y_sorted = y[np.argsort(x)[::-1]]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, y.size + 1)
    recall = tp / n_pos
    pr_auc = float(np.trapezoid(precision, recall))
    soft_dice = 2.0 * x[y == 1].sum() / (x.sum() + y.sum())
    return {"roc_auc": float(roc_auc), "pr_auc": pr_auc, "soft_dice": float(soft_dice)}


def main(args):
    vol = nib.load(args.vol).get_fdata().astype(np.float32)
    lab_img = nib.load(args.label)
    gt = lab_img.get_fdata().astype(np.float32) > 0.5
    assert vol.shape == gt.shape, f"{vol.shape} vs {gt.shape}"
    assert np.allclose(lab_img.affine, nib.load(args.vol).affine, atol=1e-3), "affine mismatch"

    spacing = tuple(abs(x) for x in np.diag(lab_img.affine)[:3])

    # AABB of label
    idx = np.argwhere(gt)
    lo = idx.min(0) - 4
    hi = idx.max(0) + 5
    lo = np.clip(lo, 0, None)
    hi = np.clip(hi, None, np.array(gt.shape))
    aabb = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    vol_r = vol[aabb]
    gt_r = gt[aabb]

    print(f"AABB: {[f'{s.start}:{s.stop}' for s in aabb]}, label voxels {int(gt_r.sum())}")
    print("Skeletonising GT ...")
    sk_gt = skeletonize(gt_r, method="lee") > 0

    print("\n=== Segmentation metrics ===")
    best = None
    for thr in args.thresholds:
        pred = vol_r > thr
        if pred.sum() == 0:
            print(f"thr {thr}: empty")
            continue
        d = dice(pred, gt_r)
        tp = int((pred & gt_r).sum())
        prec = tp / int(pred.sum())
        rec = tp / int(gt_r.sum())
        h = hd95(pred, gt_r, spacing)
        cl = cldice(pred, gt_r, sk_gt)
        print(f"thr {thr}: dice={d:.4f} precision={prec:.4f} recall={rec:.4f} hd95={h:.3f} cldice={cl:.4f}")
        if best is None or d > best[0]:
            best = (d, thr)

    print("\n=== Density metrics (threshold-free, AABB) ===")
    dm = density_metrics(vol_r, gt_r)
    print(f"roc_auc={dm['roc_auc']:.4f} pr_auc={dm['pr_auc']:.4f} soft_dice={dm['soft_dice']:.4f}")

    inside = vol_r[gt_r]
    outside = vol_r[~gt_r]
    print(f"inside mean {inside.mean():.5f} / outside mean {outside.mean():.5f} ratio {inside.mean()/outside.mean():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vol", default="/media/F/sj/Code/r2_gaussian/output/lca_vis/r2g_density_deformed_phase0.0.nii.gz")
    p.add_argument("--label", default="/media/F/sj/Code/GS-contrast-flow/data/multi_layers/rbf_reader_flow_contrast_LCA/coronary_label.nii.gz")
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    args = p.parse_args()
    main(args)
