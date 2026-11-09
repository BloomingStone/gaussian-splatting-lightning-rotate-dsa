"""Compute artifact-sensitive image metrics that correlate with human judgement:

1. global PSNR (reference)
2. v_psnr / v_ssim / v_l1  : vessel-region (2D label) metrics
3. edge_sharpness          : mean |grad(pred)| / |grad(gt)| on GT strong vessel
                             edges — low value => misalignment / ghosting / blur
4. spike_score             : high-frequency energy ratio in the background
                             (pred vs gt) — high value => needle/spike artifacts

All images are J-domain, shape (H, W), range [0, 1]. `label` is a bool vessel mask.

Runs on frame_100 (phase 0.667): 10 grid views + 3 training frames (40/100/160).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, sobel
from skimage.metrics import structural_similarity

ROOT = Path("/media/F/sj/Code/GS-contrast-flow")
R2G_ROOT = Path("/media/F/sj/Code/r2_gaussian")

GRID_DIR = ROOT / "data/example_grids/grid/frame_100_phase_0.667"
ROTATE_DSD = ROOT / "data/example_grids/rotate_dsa"
TRAIN_LABEL_DIR = ROOT / "data/example_grids/label"
STAGE1_RENDERS = ROOT / "outputs/stage1_grid_eval/renders"
STAGE1_TRAIN = ROOT / "outputs/stage1_grid_eval/train_frame"
R2G_RENDERS = R2G_ROOT / "output/lca_grid_eval/renders"
R2G_TRAIN = R2G_ROOT / "output/lca_grid_eval/train_frame"
OUT_DIR = ROOT / "outputs/comparison"

EPS = 1e-8


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def load_label(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.uint8) > 0


def psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    return float("inf") if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))


def count_profile_peaks(profile, min_peak_frac=0.4):
    """Count significant local maxima in a 1D intensity profile."""
    p = np.asarray(profile, dtype=np.float64)
    if p.size < 3:
        return 1.0
    pmax = p.max()
    if pmax <= EPS:
        return 1.0
    thr = pmax * min_peak_frac
    peaks = 0
    for i in range(1, p.size - 1):
        if p[i] > p[i - 1] and p[i] >= p[i + 1] and p[i] >= thr:
            peaks += 1
    return float(max(peaks, 1))


def ghost_score(gt, pred, vessel, radius=10, edge_quantile=0.90, max_samples=800):
    """Ghosting / edge-widening score via the normal-direction gradient profile.

    At each strong GT vessel edge pixel, sample the gradient-magnitude profile of
    the prediction along the GT gradient (normal) direction and compute its
    second moment (width). Ghosting (a vessel edge appearing twice / smeared)
    widens the profile. Returns pred_width / gt_width (1.0 = no ghosting,
    >1 = ghosting / blur / misalignment).
    """
    gx_gt = sobel(gt, axis=1)
    gy_gt = sobel(gt, axis=0)
    gmag_gt = np.hypot(gx_gt, gy_gt)
    gx_p = sobel(pred, axis=1)
    gy_p = sobel(pred, axis=0)
    gmag_p = np.hypot(gx_p, gy_p)

    thr = np.quantile(gmag_gt, edge_quantile)
    edge = (gmag_gt > thr) & vessel
    ys, xs = np.nonzero(edge)
    if len(xs) == 0:
        return float("nan")

    step = max(1, len(xs) // max_samples)
    H, W = gt.shape
    rs = np.arange(-radius, radius + 1)

    def profile_width(img_gmag):
        widths = []
        for k in range(0, len(xs), step):
            y, x = ys[k], xs[k]
            gx0, gy0 = gx_gt[y, x], gy_gt[y, x]
            nm = np.hypot(gx0, gy0) + 1e-8
            nx, ny = gx0 / nm, gy0 / nm
            prof = []
            for r in rs:
                xx = int(round(x + nx * r))
                yy = int(round(y + ny * r))
                prof.append(img_gmag[yy, xx] if 0 <= yy < H and 0 <= xx < W else 0.0)
            prof = np.asarray(prof)
            s = prof.sum() + 1e-8
            c = float((rs * prof).sum() / s)
            widths.append(float(np.sqrt(((rs - c) ** 2 * prof).sum() / s)))
        return float(np.mean(widths)) if widths else 1.0

    w_gt = profile_width(gmag_gt)
    w_pred = profile_width(gmag_p)
    return float(w_pred / (w_gt + EPS))


def compute_metrics(gt, pred, label):
    vessel = label.astype(bool)
    bg = ~vessel

    out = {}

    # --- global ---
    out["psnr"] = psnr(pred, gt)

    # --- vessel-region (label-weighted) ---
    wsum = vessel.sum()
    if wsum > 0:
        diff = (pred - gt) ** 2
        out["v_mse"] = float(diff[vessel].mean())
        out["v_psnr"] = float(10 * np.log10(1.0 / (out["v_mse"] + EPS)))
        out["v_l1"] = float(np.abs(pred - gt)[vessel].mean())
        # vessel SSIM: average the local SSIM map over vessel pixels
        _, ssim_map = structural_similarity(gt, pred, data_range=1.0, full=True)
        out["v_ssim"] = float(ssim_map[vessel].mean())
    else:
        out["v_psnr"] = out["v_l1"] = out["v_ssim"] = float("nan")

    # --- edge sharpness (misalignment / ghosting / blur) ---
    g_gt = np.hypot(sobel(gt, axis=0), sobel(gt, axis=1))
    g_pred = np.hypot(sobel(pred, axis=0), sobel(pred, axis=1))
    thr = np.quantile(g_gt, 0.90)
    edge = (g_gt > thr) & vessel
    if edge.sum() > 0:
        out["edge_sharpness"] = float(g_pred[edge].mean() / (g_gt[edge].mean() + EPS))
    else:
        out["edge_sharpness"] = float("nan")

    # --- spike / needle artifact score (background high-frequency energy) ---
    hp_gt = gt - gaussian_filter(gt, sigma=1.0)
    hp_pred = pred - gaussian_filter(pred, sigma=1.0)
    e_gt = float((hp_gt[bg] ** 2).mean())
    e_pred = float((hp_pred[bg] ** 2).mean())
    out["spike_score"] = float(e_pred / (e_gt + EPS))

    # --- ghosting / double-edge score (normal-direction profile peaks) ---
    out["ghost_score"] = ghost_score(gt, pred, vessel)

    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rot_meta = json.load(open(ROOT / "data/example_grids/rotate_dsa.json"))
    rot_frames = {int(f["frame"]): f for f in rot_meta["frames"]}

    rows = []
    # grid views 0..9
    for i in range(10):
        gt = load_gray(GRID_DIR / "dsa" / f"{i:03d}.png")
        label = load_label(GRID_DIR / "label" / f"{i:03d}.png")
        s1 = load_gray(STAGE1_RENDERS / f"val_{50 + i:03d}.png")
        r2 = load_gray(R2G_RENDERS / f"frame_100_phase_0.667_{i:03d}.png")
        rows.append({
            "name": f"grid view {i}",
            "alpha": json.load(open(GRID_DIR / "meta.json"))["frames"][i]["alpha_degree"],
            "stage1": compute_metrics(gt, s1, label),
            "r2g": compute_metrics(gt, r2, label),
        })

    # training frames 40 / 100 / 160 (same phase 0.667)
    for f in [40, 100, 160]:
        gt = load_gray(ROTATE_DSD / f"{f:03d}.png")
        label = load_label(TRAIN_LABEL_DIR / f"{f:03d}.png")
        s1 = load_gray(STAGE1_TRAIN / f"stage1_frame{f}.png")
        r2 = load_gray(R2G_TRAIN / f"r2g_frame{f}.png")
        rows.append({
            "name": f"train frame {f}",
            "alpha": rot_frames[f]["alpha_degree"],
            "stage1": compute_metrics(gt, s1, label),
            "r2g": compute_metrics(gt, r2, label),
        })

    # sort by angle
    rows.sort(key=lambda r: r["alpha"])

    metrics = ["psnr", "v_psnr", "v_ssim", "v_l1", "edge_sharpness", "ghost_score", "spike_score"]
    header = f"{'view':<20}{'alpha':>8}" + "".join(f"{m:>13}" for m in metrics) + f"{'winner':>10}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for r in rows:
        line = f"{r['name']:<20}{r['alpha']:>8.1f}"
        for m in metrics:
            line += f"{r['stage1'][m]:>13.4f}"
        print(line + "    [stage1]")
        line = f"{'':<28}"
        for m in metrics:
            line += f"{r['r2g'][m]:>13.4f}"
        print(line + "    [r2g]")
        print("-" * len(header))

    # save
    with open(OUT_DIR / "artifact_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {OUT_DIR / 'artifact_metrics.json'}")


if __name__ == "__main__":
    main()
