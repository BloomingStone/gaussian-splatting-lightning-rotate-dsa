"""把多个 stage2 实验的同一帧 grid 对比合并成一张大图 (GT + 各实验并排)。

用法：
    pixi run python scripts/make_stage2_grid_merged.py \
        --experiments stage2-base stage2-full stage2-full-smooth stage2-full-smooth-nogate \
        --frame 100 --out_dir dev_grid_views/stage2-flow
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("/media/F/sj/Code/GS-contrast-flow")
GRID_DIR = ROOT / "data/example_grids/grid"
STAGE2_OUT = ROOT / "outputs/stage2_grid_eval"


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def find_phase_dir(frame: int) -> Path:
    for d in sorted(GRID_DIR.iterdir()):
        if d.is_dir() and d.name.startswith(f"frame_{frame:03d}_"):
            return d
    raise FileNotFoundError(f"no grid phase dir for frame {frame}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", type=str, nargs="+", required=True)
    p.add_argument("--frame", type=int, default=100)
    p.add_argument("--out_dir", type=Path, default=ROOT / "dev_grid_views/stage2-flow")
    args = p.parse_args()

    frame = args.frame
    phase_dir = find_phase_dir(frame)
    meta = json.load(open(phase_dir / "meta.json"))
    alphas = [f["alpha_degree"] for f in meta["frames"]]
    n_views = len(alphas)
    idx_start = (frame // 20) * 10

    # 各实验的 pred + psnr
    exps = args.experiments
    preds = []      # [n_exp][n_views] np array
    psnrs = []      # [n_exp][n_views] float
    vpsnrs = []
    for exp in exps:
        ge = STAGE2_OUT / exp
        metrics = json.load(open(ge / "metrics.json"))
        p_by_idx = {r["idx"]: r["psnr"] for r in metrics["per_image"]}
        vp_by_idx = {r["idx"]: r.get("v_psnr", float("nan")) for r in metrics["per_image"]}
        pred_imgs = [load_gray(ge / "renders" / f"val_{idx_start + i:03d}.png") for i in range(n_views)]
        preds.append(pred_imgs)
        psnrs.append([p_by_idx.get(idx_start + i, float("nan")) for i in range(n_views)])
        vpsnrs.append([vp_by_idx.get(idx_start + i, float("nan")) for i in range(n_views)])

    # GT（所有实验共享，取第一个实验的 gt 目录）
    ge0 = STAGE2_OUT / exps[0]
    gt_imgs = [load_gray(ge0 / "gt" / f"val_{idx_start + i:03d}.png") for i in range(n_views)]

    n_cols = 1 + len(exps)
    fig, axes = plt.subplots(n_views, n_cols, figsize=(2.9 * n_cols, 2.8 * n_views))

    for i in range(n_views):
        # GT 列
        axes[i, 0].imshow(gt_imgs[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"α={alphas[i]:+.1f}°", fontsize=9, pad=4)
        # 各实验列
        for j, exp in enumerate(exps):
            ax = axes[i, 1 + j]
            ax.imshow(preds[j][i], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            ax.set_title(f"P={psnrs[j][i]:.2f}\nvP={vpsnrs[j][i]:.2f}", fontsize=8, pad=4)

    # 列标题
    col_names = ["GT"] + [e.replace("stage2-", "") for e in exps]
    for j, name in enumerate(col_names):
        fig.text((j + 0.5) / n_cols, 0.985, name, ha="center", va="top",
                 fontsize=11, fontweight="bold")
    fig.suptitle(f"grid frame {frame} (phase {meta['phase']:.3f})", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out_dir / f"frame_{frame}_grid_compare_merged.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
