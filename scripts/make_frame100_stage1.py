"""Single stage-1 frame-100 comparison figure (GT / stage1, no r2g).

Used to visualise every sweep run. Layout: 13 rows x 2 columns
(grid views 0..9 + training frames 40/100/160), sorted by C-arm angle.
Red box = train frame 100, yellow box = train frames 40/160.
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
GRID_DIR = ROOT / "data/example_grids/grid/frame_100_phase_0.667"
ROTATE_DSD = ROOT / "data/example_grids/rotate_dsa"
TRAIN_FRAMES = [40, 100, 160]


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def style_box(ax, color: str):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                           edgecolor=color, linewidth=3.0, clip_on=False))


def main(args):
    stage1_dir = args.stage1_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.load(open(GRID_DIR / "meta.json"))
    grid_alphas = [f["alpha_degree"] for f in meta["frames"]]
    grid_src = [int(f["frame"]) for f in meta["frames"]]

    rot_meta = json.load(open(ROOT / "data/example_grids/rotate_dsa.json"))
    rot_frames = {int(f["frame"]): f for f in rot_meta["frames"]}
    train_alphas = {f: rot_frames[f]["alpha_degree"] for f in TRAIN_FRAMES}

    s1_grid = json.load(open(stage1_dir / "metrics.json"))
    s1_psnr = {r["idx"]: r["psnr"] for r in s1_grid["per_image"]}
    s1_train_psnr = {
        f: json.load(open(stage1_dir / "train_frame" / f"stage1_frame{f}_psnr.json"))["psnr"]
        for f in TRAIN_FRAMES
    }

    rows = []
    for i in range(10):
        rows.append({"kind": "grid", "view": i, "alpha": grid_alphas[i]})
    for f in TRAIN_FRAMES:
        rows.append({"kind": "train", "frame": f, "alpha": train_alphas[f]})
    rows.sort(key=lambda r: r["alpha"])

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7, 2.8 * n_rows))

    for r_i, row in enumerate(rows):
        if row["kind"] == "grid":
            i = row["view"]
            gt = load_gray(GRID_DIR / "dsa" / f"{i:03d}.png")
            pred = load_gray(stage1_dir / "renders" / f"val_{50 + i:03d}.png")
            p = s1_psnr[50 + i]
            label = f"α={row['alpha']:+.1f}°\ngrid view {i} (src f{grid_src[i]})"
            box = None
        else:
            f = row["frame"]
            gt = load_gray(ROTATE_DSD / f"{f:03d}.png")
            pred = load_gray(stage1_dir / "train_frame" / f"stage1_frame{f}.png")
            p = s1_train_psnr[f]
            label = f"α={row['alpha']:+.1f}°\ntrain frame {f} (phase 0.667)"
            box = "red" if f == 100 else "gold"

        for j, im in enumerate((gt, pred)):
            ax = axes[r_i, j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if box is not None:
                style_box(ax, box)
            if j == 0:
                ax.set_title(label, fontsize=9, pad=4)
            else:
                ax.set_title(f"PSNR={p:.2f}", fontsize=9, pad=4)

    for j, name in enumerate(["GT", "stage1"]):
        fig.text((j + 0.5) / 2.0, 0.985, name, ha="center", va="top",
                 fontsize=12, fontweight="bold")
    fig.suptitle(
        f"frame_100 (phase 0.667): {args.exp_name}\n"
        "red box = train frame 100   |   yellow box = train frames 40 / 160",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / f"frame_100_compare_{args.exp_name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=ROOT / "dev_grid_views/bspline-frangi")
    p.add_argument("--exp_name", default="exp")
    main(p.parse_args())
