"""Voxelise each phase's deformed pcd, then evaluate per-phase 3D metrics.

Reports dice / precision / recall / hd95 / clDice at the best threshold per
phase, plus tolerance dice (NSD, surface distance with tolerance tau). Finds
the best phase and plots dice vs phase.

Run in the GS-contrast-flow environment.

Usage:
    pixi run python scripts/voxelize_and_eval_multi.py
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import KDTree
from skimage.morphology import skeletonize

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from internal.savers.x_ray_saver import gaussians_to_volume_by_Rasterizer


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


def nsd(pred, gt, tau, spacing):
    """Normalized Surface Dice: fraction of surface voxels within tau mm of the
    other mask's surface."""
    struct = generate_binary_structure(3, 1)
    surf_p = np.argwhere(pred & ~binary_erosion(pred, structure=struct)).astype(np.float64)
    surf_g = np.argwhere(gt & ~binary_erosion(gt, structure=struct)).astype(np.float64)
    if len(surf_p) == 0 or len(surf_g) == 0:
        return float("nan")
    sp = np.array(spacing, dtype=np.float64)
    tp, tg = KDTree(surf_p * sp), KDTree(surf_g * sp)
    d_p = tp.query(surf_g * sp)[0]  # gt surface -> pred surface
    d_g = tg.query(surf_p * sp)[0]  # pred surface -> gt surface
    s_p = int((d_p <= tau).sum())
    s_g = int((d_g <= tau).sum())
    return (s_p + s_g) / (len(surf_p) + len(surf_g))


def main(args):
    lab_img = nib.load(args.label)
    gt_full = lab_img.get_fdata().astype(np.float32) > 0.5
    spacing = tuple(abs(float(x)) for x in np.diag(lab_img.affine)[:3])

    # AABB around label
    idx = np.argwhere(gt_full)
    lo = np.clip(idx.min(0) - 4, 0, None)
    hi = np.clip(idx.max(0) + 5, None, np.array(gt_full.shape))
    aabb = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    gt = gt_full[aabb]
    print(f"AABB: {[f'{s.start}:{s.stop}' for s in aabb]}, label voxels {int(gt.sum())}")
    sk_gt = skeletonize(gt, method="lee") > 0

    meta = json.load(open(args.meta_json))
    ca = np.array(meta["lca_centering_affine"], dtype=np.float64)
    shape = tuple(int(x) for x in meta["volume_size"])

    phases = np.linspace(0.0, 1.0, args.n_phases, endpoint=True)
    rows = []
    for p in phases:
        pkl = os.path.join(args.pcd_dir, f"deformed_pcd_phase{p:.3f}.pkl")
        if not os.path.exists(pkl):
            print(f"missing {pkl}, skip")
            continue
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        xyz = torch.tensor(d["xyz_mm"], dtype=torch.float32, device="cuda")
        scale = torch.tensor(d["scale_mm"], dtype=torch.float32, device="cuda")
        rot = torch.tensor(d["rotation"], dtype=torch.float32, device="cuda")
        dens = torch.tensor(d["density"], dtype=torch.float32, device="cuda")

        vol_full = gaussians_to_volume_by_Rasterizer(
            xyz, scale, rot, dens, shape, ca, to_cpu=True)
        vol = vol_full[aabb]

        best = None
        for thr in args.thresholds:
            pred = vol > thr
            if pred.sum() == 0:
                continue
            dd = dice(pred, gt)
            if best is None or dd > best[0]:
                best = (dd, thr)
        if best is None:
            print(f"phase {p:.2f}: all thresholds empty")
            continue
        thr = best[1]
        pred = vol > thr
        tp = int((pred & gt).sum())
        prec = tp / int(pred.sum())
        rec = tp / int(gt.sum())
        h = hd95(pred, gt, spacing)
        cl = cldice(pred, gt, sk_gt)
        n1 = nsd(pred, gt, args.tau[0], spacing) if args.tau else float("nan")
        rows.append({"phase": p, "thr": thr, "dice": best[0], "prec": prec,
                     "rec": rec, "hd95": h, "cldice": cl, "nsd1": n1})
        print(f"phase {p:.2f}: thr {thr:.2f} dice {best[0]:.4f} prec {prec:.4f} "
              f"rec {rec:.4f} hd95 {h:.3f} cldice {cl:.4f} nsd{args.tau[0]:.0f} {n1:.4f}")

    if not rows:
        print("no phases evaluated")
        return

    # summary
    best_row = max(rows, key=lambda r: r["dice"])
    print(f"\nbest phase {best_row['phase']:.2f}: dice {best_row['dice']:.4f} "
          f"(phase0 reference: see table)")
    ph0 = next((r for r in rows if abs(r["phase"]) < 1e-6), None)
    if ph0:
        print(f"phase 0.00: dice {ph0['dice']:.4f} (best {best_row['dice']:.4f} at "
              f"phase {best_row['phase']:.2f})")

    # plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ph = [r["phase"] for r in rows]
    ax1.plot(ph, [r["dice"] for r in rows], "o-", label="dice")
    ax1.plot(ph, [r["cldice"] for r in rows], "s-", label="clDice")
    ax1.plot(ph, [r["nsd1"] for r in rows], "^-", label=f"NSD(tau={args.tau[0]}mm)")
    ax1.set_xlabel("phase")
    ax1.set_ylabel("score")
    ax1.set_title("3D metrics vs phase")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "dice_vs_phase.png")
    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pcd_dir", default="/media/F/sj/Code/r2_gaussian/output/lca_vis/multi_phase")
    p.add_argument("--label",
                   default="/media/F/sj/Code/GS-contrast-flow/data/multi_layers/rbf_reader_flow_contrast_LCA/coronary_label.nii.gz")
    p.add_argument("--meta_json",
                   default="/media/F/sj/Code/GS-contrast-flow/data/multi_layers/rbf_reader_flow_contrast_LCA/rotate_dsa.json")
    p.add_argument("--n_phases", default=11, type=int)
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    p.add_argument("--tau", nargs="+", type=float, default=[1.0])
    p.add_argument("--out_dir", default="/media/F/sj/Code/r2_gaussian/output/lca_vis/multi_phase")
    args = p.parse_args()
    main(args)
