"""hashgrid(stage1-spike) vs bspline(bs32-smooth-0.2-frangi) 的 frame-100 对比图。

三列：GT / hashgrid / bspline，13 行 = 10 grid views + 3 train frames(40/100/160)，
按 C-arm 角度排序。红框 = train frame 100，黄框 = train frames 40/160。
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

HG_DIR = ROOT / "outputs/dev_grid_views/stage1-spike"
BS_DIR = ROOT / "outputs/dev_grid_views/bspline-frangi/bs32-smooth-0.2-frangi"


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def style_box(ax, color: str):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                           edgecolor=color, linewidth=3.0, clip_on=False))


def main(args):
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.load(open(GRID_DIR / "meta.json"))
    grid_alphas = [f["alpha_degree"] for f in meta["frames"]]
    grid_src = [int(f["frame"]) for f in meta["frames"]]

    rot_meta = json.load(open(ROOT / "data/example_grids/rotate_dsa.json"))
    rot_frames = {int(f["frame"]): f for f in rot_meta["frames"]}
    train_alphas = {f: rot_frames[f]["alpha_degree"] for f in TRAIN_FRAMES}

    hg_metrics = json.load(open(HG_DIR / "metrics.json"))
    bs_metrics = json.load(open(BS_DIR / "metrics.json"))
    hg_psnr = {r["idx"]: r["psnr"] for r in hg_metrics["per_image"]}
    bs_psnr = {r["idx"]: r["psnr"] for r in bs_metrics["per_image"]}

    hg_train_psnr = {
        f: json.load(open(HG_DIR / "train_frame" / f"stage1_frame{f}_psnr.json"))["psnr"]
        for f in TRAIN_FRAMES
    }
    bs_train_psnr = {
        f: json.load(open(BS_DIR / "train_frame" / f"stage1_frame{f}_psnr.json"))["psnr"]
        for f in TRAIN_FRAMES
    }

    rows = []
    for i in range(10):
        rows.append({"kind": "grid", "view": i, "alpha": grid_alphas[i]})
    for f in TRAIN_FRAMES:
        rows.append({"kind": "train", "frame": f, "alpha": train_alphas[f]})
    rows.sort(key=lambda r: r["alpha"])

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(10, 2.8 * n_rows))

    for r_i, row in enumerate(rows):
        if row["kind"] == "grid":
            i = row["view"]
            gt = load_gray(GRID_DIR / "dsa" / f"{i:03d}.png")
            hg = load_gray(HG_DIR / "renders" / f"val_{50 + i:03d}.png")
            bs = load_gray(BS_DIR / "renders" / f"val_{50 + i:03d}.png")
            p_hg = hg_psnr[50 + i]
            p_bs = bs_psnr[50 + i]
            label = f"α={row['alpha']:+.1f}°\ngrid view {i} (src f{grid_src[i]})"
            box = None
        else:
            f = row["frame"]
            gt = load_gray(ROTATE_DSD / f"{f:03d}.png")
            hg = load_gray(HG_DIR / "train_frame" / f"stage1_frame{f}.png")
            bs = load_gray(BS_DIR / "train_frame" / f"stage1_frame{f}.png")
            p_hg = hg_train_psnr[f]
            p_bs = bs_train_psnr[f]
            label = f"α={row['alpha']:+.1f}°\ntrain frame {f} (phase 0.667)"
            box = "red" if f == 100 else "gold"

        ims = [gt, hg, bs]
        titles = [label, f"hashgrid\nPSNR={p_hg:.2f}", f"bspline\nPSNR={p_bs:.2f}"]
        for j, (im, t) in enumerate(zip(ims, titles)):
            ax = axes[r_i, j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if box is not None:
                style_box(ax, box)
            if j == 0:
                ax.set_title(t, fontsize=9, pad=4)
            else:
                ax.set_title(t, fontsize=9, pad=4)

    for j, name in enumerate(["GT", "hashgrid (stage1-spike)", "bspline (bs32-smooth-0.2-frangi)"]):
        fig.text((j + 0.5) / 3.0, 0.985, name, ha="center", va="top",
                 fontsize=12, fontweight="bold")
    fig.suptitle(
        "frame_100 (phase 0.667): hashgrid vs bspline best configs\n"
        "red box = train frame 100   |   yellow box = train frames 40 / 160",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / "frame_100_compare_hashgrid_vs_bspline.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=ROOT / "dev_grid_views/hashgrid-adaptive-smooth")
    main(p.parse_args())
