"""phase 0 单视角 (指定 alpha) 的误差分析图。

对选定视角：
  1. 用 2D label 截取冠脉 mask
  2. 计算 mask 内 error = pred - gt
  3. 绘制 error map 并统计正/负误差（正=假阳性/多出的分支，负=漏检）

用法：
    pixi run python scripts/make_stage2_phase0_error.py \
        --frame 0 --alpha 120 \
        --experiments stage2-base stage2-full stage2-full-smooth stage2-full-smooth-nogate stage2-joint \
        --out_dir dev_grid_views/stage2-flow
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
EVAL_DIR = ROOT / "outputs/stage2_grid_eval"


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def find_phase_dir(frame: int) -> Path:
    for d in sorted(GRID_DIR.iterdir()):
        if d.is_dir() and d.name.startswith(f"frame_{frame:03d}_"):
            return d
    raise FileNotFoundError(f"no grid phase dir for frame {frame}")


def find_view_index(phase_dir: Path, alpha: float) -> int:
    meta = json.load(open(phase_dir / "meta.json"))
    for i, f in enumerate(meta["frames"]):
        if abs(f["alpha_degree"] - alpha) < 0.5:
            return i
    raise ValueError(f"alpha {alpha} not found in {phase_dir}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--alpha", type=float, default=120.0)
    p.add_argument("--experiments", type=str, nargs="+", required=True)
    p.add_argument("--out_dir", type=Path, default=ROOT / "dev_grid_views/stage2-flow")
    args = p.parse_args()

    phase_dir = find_phase_dir(args.frame)
    vi = find_view_index(phase_dir, args.alpha)
    idx = (args.frame // 20) * 10 + vi  # 验证集 batch_idx（phase_stride=20）

    gt = load_gray(phase_dir / "dsa" / f"{vi:03d}.png")
    label = load_gray(phase_dir / "label" / f"{vi:03d}.png")
    mask = label > 0.5
    print(f"frame={args.frame} alpha={args.alpha:+.1f} view_idx={vi} batch_idx={idx} "
          f"mask_pixels={mask.sum()} / {mask.size}")

    preds = {}
    for exp in args.experiments:
        preds[exp] = load_gray(EVAL_DIR / exp / "renders" / f"val_{idx:03d}.png")

    # ---------- 绘图 ----------
    n_exp = len(args.experiments)
    fig, axes = plt.subplots(2, 1 + n_exp, figsize=(2.9 * (1 + n_exp), 5.8))

    # 第一行：GT / label mask / 各实验 pred（mask 外灰）
    def show_masked(ax, im):
        disp = np.full_like(im, 0.5)
        disp[mask] = im[mask]
        ax.imshow(disp, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    axes[0, 0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("GT", fontsize=10)
    axes[0, 0].axis("off")

    axes[1, 0].imshow(label, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("2D label mask", fontsize=10)
    axes[1, 0].axis("off")

    # 第二行第一列：GT 的 mask 内放大（便于对比细节）
    for j, exp in enumerate(args.experiments):
        ax = axes[0, 1 + j]
        show_masked(ax, preds[exp])
        ax.set_title(exp.replace("stage2-", ""), fontsize=10)

        err = np.full_like(gt, np.nan)
        err[mask] = preds[exp][mask] - gt[mask]
        vmax = max(np.nanpercentile(np.abs(err), 99), 1e-3)
        eax = axes[1, 1 + j]
        im = eax.imshow(err, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        eax.axis("off")
        eax.set_title("error (red=+, blue=-)", fontsize=9)

        # 统计
        e = err[mask]
        pos = e[e > 0]
        neg = e[e < 0]
        print(f"\n[{exp}]")
        print(f"  mean_err  = {e.mean():+.5f}   |err|mean = {np.abs(e).mean():.5f}  rmse = {np.sqrt((e**2).mean()):.5f}")
        print(f"  pos: n={len(pos):6d} ({len(pos)/len(e)*100:5.1f}%)  sum={pos.sum():+.5f}  mean={pos.mean():+.5f}")
        print(f"  neg: n={len(neg):6d} ({len(neg)/len(e)*100:5.1f}%)  sum={neg.sum():+.5f}  mean={neg.mean():+.5f}")
        print(f"  |err|>0.05 pixels: {np.sum(np.abs(e) > 0.05)} ({np.sum(np.abs(e)>0.05)/len(e)*100:.1f}%)")

    fig.colorbar(im, ax=axes[1, 1 + n_exp - 1], fraction=0.046, pad=0.04)
    fig.suptitle(f"phase 0 (frame {args.frame}) alpha={args.alpha:+.1f}° — 冠脉 mask 内误差分析",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = args.out_dir / f"phase0_alpha{args.alpha:+.0f}_error_analysis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
