"""Generate (1) a metric comparison table and (2) a frame-100 multi-view
comparison figure (GT / stage1 / r2g).

Figure layout: 11 rows x 3 columns.
  Row 0      : training-set frame 100 (red box)  — GT + stage1 + r2g
  Rows 1..10 : grid frame_100_phase_0.667 views 0..9 (multi-angle)
               views whose source frame is in the training set (>= 40) get a
               yellow box.

Each stage1 / r2g cell is annotated with its PSNR; each row is labelled with the
C-arm angle (alpha).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path("/media/F/sj/Code/GS-contrast-flow")
R2G_ROOT = Path("/media/F/sj/Code/r2_gaussian")

GRID_DIR = ROOT / "data/example_grids/grid/frame_100_phase_0.667"
ROTATE_DSD = ROOT / "data/example_grids/rotate_dsa"
STAGE1_RENDERS = ROOT / "outputs/stage1_grid_eval/renders"
STAGE1_TRAIN = ROOT / "outputs/stage1_grid_eval/train_frame"
R2G_RENDERS = R2G_ROOT / "output/lca_grid_eval/renders"
R2G_TRAIN = R2G_ROOT / "output/lca_grid_eval/train_frame"
OUT_DIR = ROOT / "outputs/comparison"

TRAIN_START_FRAME = 40


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def style_box(ax, color: str):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                           edgecolor=color, linewidth=3.0, clip_on=False))


def main(args):
    # local overrides from CLI args
    OUT_DIR = args.out_dir
    GRID_DIR = args.grid_dir
    STAGE1_RENDERS = args.stage1_dir / "renders"
    STAGE1_TRAIN = args.stage1_dir / "train_frame"
    R2G_RENDERS = args.r2g_dir / "renders"
    R2G_TRAIN = args.r2g_dir / "train_frame"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- metric table ----------
    stage1_csv = ROOT / "outputs/stage1_examplegrids_gridval/metrics/val-step=20000.csv"
    with open(stage1_csv) as f:
        lines = f.read().strip().splitlines()
    hdr = lines[0].split(",")
    for ln in lines:
        if ln.startswith("MEAN"):
            vals = ln.split(",")
            stage1 = {
                "psnr": float(vals[hdr.index("psnr")]),
                "ssim": 1.0 - float(vals[hdr.index("ssim_loss")]),
                "lpips": float(vals[hdr.index("lpips")]),
            }
            break

    with open(R2G_RENDERS.parent / "metrics.json") as f:
        r2g_metrics = json.load(f)
    r2g = {"psnr": r2g_metrics["mean"]["psnr"], "ssim": r2g_metrics["mean"]["ssim"]}

    print("=" * 46)
    print("grid 验证集 metric 对比 (90 张, phase_stride=20)")
    print("=" * 46)
    print(f"{'metric':<10}{'stage1':>14}{'r2g':>14}")
    print("-" * 46)
    print(f"{'PSNR':<10}{stage1['psnr']:>14.4f}{r2g['psnr']:>14.4f}")
    print(f"{'SSIM':<10}{stage1['ssim']:>14.4f}{r2g['ssim']:>14.4f}")
    print(f"{'LPIPS':<10}{stage1['lpips']:>14.4f}{'n/a':>14}")
    print("=" * 46)

    with open(OUT_DIR / "metrics_compare.json", "w") as f:
        json.dump({"stage1": stage1, "r2g": r2g}, f, indent=2)

    # ---------- frame-100 multi-view figure ----------
    meta = json.load(open(GRID_DIR / "meta.json"))
    frames = meta["frames"]
    grid_alphas = [f["alpha_degree"] for f in frames]
    grid_src = [int(f["frame"]) for f in frames]

    rot_meta = json.load(open(ROOT / "data/example_grids/rotate_dsa.json"))
    rot_frames = {int(f["frame"]): f for f in rot_meta["frames"]}
    train_frames = [40, 100, 160]
    train_alphas = {f: rot_frames[f]["alpha_degree"] for f in train_frames}

    stage1_grid = json.load(open(STAGE1_RENDERS.parent / "metrics.json"))
    stage1_psnr = {r["idx"]: r["psnr"] for r in stage1_grid["per_image"]}
    r2g_grid = json.load(open(R2G_RENDERS.parent / "metrics.json"))
    r2g_psnr = {}
    for k, v in r2g_grid["per_image"].items():
        view = int(k.rsplit("/", 1)[1].split(".")[0])
        r2g_psnr[view] = v["psnr"]

    s1_train_psnr = {f: json.load(open(STAGE1_TRAIN / f"stage1_frame{f}_psnr.json"))["psnr"]
                     for f in train_frames}
    r2g_train_psnr = {f: json.load(open(R2G_TRAIN / f"r2g_frame{f}_psnr.json"))["psnr"]
                      for f in train_frames}

    # Build rows sorted by C-arm angle: grid views + training frames (phase 0.667).
    rows = []
    for i in range(10):
        rows.append({"kind": "grid", "view": i, "alpha": grid_alphas[i]})
    for f in train_frames:
        rows.append({"kind": "train", "frame": f, "alpha": train_alphas[f]})
    rows.sort(key=lambda r: r["alpha"])

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 2.8 * n_rows))

    for r_i, row in enumerate(rows):
        if row["kind"] == "grid":
            i = row["view"]
            ims = [
                load_gray(GRID_DIR / "dsa" / f"{i:03d}.png"),
                load_gray(STAGE1_RENDERS / f"val_{50 + i:03d}.png"),
                load_gray(R2G_RENDERS / f"frame_100_phase_0.667_{i:03d}.png"),
            ]
            psnrs = [None, stage1_psnr[50 + i], r2g_psnr[i]]
            label = f"α={row['alpha']:+.1f}°\ngrid view {i} (src f{grid_src[i]})"
            box = None
        else:
            f = row["frame"]
            ims = [
                load_gray(ROTATE_DSD / f"{f:03d}.png"),
                load_gray(STAGE1_TRAIN / f"stage1_frame{f}.png"),
                load_gray(R2G_TRAIN / f"r2g_frame{f}.png"),
            ]
            psnrs = [None, s1_train_psnr[f], r2g_train_psnr[f]]
            label = f"α={row['alpha']:+.1f}°\ntrain frame {f} (phase 0.667)"
            box = "red" if f == 100 else "gold"

        for j, im in enumerate(ims):
            ax = axes[r_i, j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if box is not None:
                style_box(ax, box)
            if j == 0:
                ax.set_title(label, fontsize=9, pad=4)
            else:
                ax.set_title(f"PSNR={psnrs[j]:.2f}", fontsize=9, pad=4)

    # Column headers
    for j, name in enumerate(["GT", "stage1", "r2g"]):
        fig.text((j + 0.5) / 3.0, 0.985, name, ha="center", va="top",
                 fontsize=12, fontweight="bold")

    fig.suptitle(
        "frame_100 (phase 0.667): grid multi-view + training frames, sorted by angle\n"
        "red box = train frame 100   |   yellow box = train frames 40 / 160 (same phase 0.667)",
        fontsize=12, fontweight="bold", y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = OUT_DIR / f"frame_100_grid_compare_{args.exp_name}.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"\nsaved comparison figure -> {out_png}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--grid_dir", default=GRID_DIR, type=Path)
    p.add_argument("--stage1_dir", default=STAGE1_RENDERS.parent, type=Path)
    p.add_argument("--r2g_dir", default=R2G_RENDERS.parent, type=Path)
    p.add_argument("--out_dir", default=OUT_DIR, type=Path)
    p.add_argument("--exp_name", default="")
    main(p.parse_args())
