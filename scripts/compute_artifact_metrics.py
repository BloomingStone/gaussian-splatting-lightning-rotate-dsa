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


def ghost_score(gt, pred, vessel, radius=8, edge_quantile=0.90,
                max_samples=800, sigma=0.5):
    """Double-edge / ghosting score via dark-valley counting on normal-direction
    intensity profiles.

    At each strong GT vessel edge pixel, sample the prediction intensity profile
    along the GT gradient (normal) direction and count local minima (dark
    valleys). A vessel edge appearing twice (ghosting) yields >=2 valleys on a
    single profile. Returns the fraction of edge pixels with >=2 valleys
    (0 = no ghosting, >0 = ghosting). Matches human judgement.
    """
    gx_gt = sobel(gt, axis=1)
    gy_gt = sobel(gt, axis=0)
    gmag = np.hypot(gx_gt, gy_gt)
    thr = np.quantile(gmag, edge_quantile)
    edge = (gmag > thr) & vessel
    ys, xs = np.nonzero(edge)
    if len(xs) == 0:
        return float("nan")

    step = max(1, len(xs) // max_samples)
    H, W = gt.shape
    dbl = 0
    tot = 0
    for k in range(0, len(xs), step):
        y, x = ys[k], xs[k]
        gx0, gy0 = gx_gt[y, x], gy_gt[y, x]
        nm = np.hypot(gx0, gy0) + 1e-8
        nx, ny = gx0 / nm, gy0 / nm
        prof = []
        for r in range(-radius, radius + 1):
            xx = int(round(x + nx * r))
            yy = int(round(y + ny * r))
            prof.append(pred[yy, xx] if 0 <= yy < H and 0 <= xx < W else 0.0)
        prof = gaussian_filter1d(np.asarray(prof), sigma)
        m = prof.mean()
        s = prof.std() + 1e-8
        valleys = sum(
            1 for j in range(1, len(prof) - 1)
            if prof[j] < prof[j - 1] and prof[j] <= prof[j + 1]
            and prof[j] < m - 0.5 * s
        )
        if valleys >= 2:
            dbl += 1
        tot += 1
    return float(dbl / tot)


def edge_contrast(gt, pred, vessel, radius=6, edge_quantile=0.90, max_samples=800):
    """Edge contrast ratio: mean intensity range (max - min) along the normal
    direction at GT vessel edges, pred / gt. <1 = under-reconstruction blur."""
    gx_gt = sobel(gt, axis=1)
    gy_gt = sobel(gt, axis=0)
    gmag = np.hypot(gx_gt, gy_gt)
    thr = np.quantile(gmag, edge_quantile)
    edge = (gmag > thr) & vessel
    ys, xs = np.nonzero(edge)
    if len(xs) == 0:
        return float("nan")

    step = max(1, len(xs) // max_samples)
    H, W = gt.shape

    def mean_range(img):
        ranges = []
        for k in range(0, len(xs), step):
            y, x = ys[k], xs[k]
            gx0, gy0 = gx_gt[y, x], gy_gt[y, x]
            nm = np.hypot(gx0, gy0) + 1e-8
            nx, ny = gx0 / nm, gy0 / nm
            prof = []
            for r in range(-radius, radius + 1):
                xx = int(round(x + nx * r))
                yy = int(round(y + ny * r))
                prof.append(img[yy, xx] if 0 <= yy < H and 0 <= xx < W else 0.0)
            ranges.append(float(np.max(prof) - np.min(prof)))
        return float(np.mean(ranges)) if ranges else 1.0

    return float(mean_range(pred) / (mean_range(gt) + EPS))


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

    # --- edge contrast ratio (under-reconstruction blur) ---
    out["edge_contrast"] = edge_contrast(gt, pred, vessel)

    return out


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rot_meta = json.load(open(ROOT / "data/example_grids/rotate_dsa.json"))
    rot_frames = {int(f["frame"]): f for f in rot_meta["frames"]}

    rows = []
    # grid views 0..9
    for i in range(10):
        gt = load_gray(args.grid_dir / "dsa" / f"{i:03d}.png")
        label = load_label(args.grid_dir / "label" / f"{i:03d}.png")
        s1 = load_gray(args.stage1_dir / "renders" / f"val_{50 + i:03d}.png")
        r2 = load_gray(args.r2g_dir / "renders" / f"frame_100_phase_0.667_{i:03d}.png")
        rows.append({
            "name": f"grid view {i}",
            "alpha": json.load(open(args.grid_dir / "meta.json"))["frames"][i]["alpha_degree"],
            "stage1": compute_metrics(gt, s1, label),
            "r2g": compute_metrics(gt, r2, label),
        })

    # training frames 40 / 100 / 160 (same phase 0.667)
    for f in [40, 100, 160]:
        gt = load_gray(ROTATE_DSD / f"{f:03d}.png")
        label = load_label(TRAIN_LABEL_DIR / f"{f:03d}.png")
        s1 = load_gray(args.stage1_dir / "train_frame" / f"stage1_frame{f}.png")
        r2 = load_gray(args.r2g_dir / "train_frame" / f"r2g_frame{f}.png")
        rows.append({
            "name": f"train frame {f}",
            "alpha": rot_frames[f]["alpha_degree"],
            "stage1": compute_metrics(gt, s1, label),
            "r2g": compute_metrics(gt, r2, label),
        })

    # sort by angle
    rows.sort(key=lambda r: r["alpha"])

    metrics = ["psnr", "v_psnr", "v_ssim", "v_l1", "edge_sharpness", "ghost_score", "edge_contrast", "spike_score"]
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
    with open(args.out_dir / "artifact_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {args.out_dir / 'artifact_metrics.json'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--grid_dir", default=GRID_DIR)
    p.add_argument("--stage1_dir", default=STAGE1_RENDERS.parent)
    p.add_argument("--r2g_dir", default=R2G_RENDERS.parent)
    p.add_argument("--out_dir", default=OUT_DIR)
    main(p.parse_args())
