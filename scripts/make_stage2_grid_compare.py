"""stage2 grid 对比图：指定帧的多视角 (GT vs pred)，标注 PSNR + C-arm 角度。

用法：
    pixi run python scripts/make_stage2_grid_compare.py \
        --grid_eval_dir outputs/dev_grid_views/stage2/<exp>/grid_eval \
        --exp_name <exp> --frames 20 100 --out_dir dev_grid_views/stage2-flow
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("/media/F/sj/Code/GS-contrast-flow")
GRID_DIR = ROOT / "data/example_only_flow_grids/grid"


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def find_phase_dir(frame: int) -> Path:
    for d in sorted(GRID_DIR.iterdir()):
        if d.is_dir() and d.name.startswith(f"frame_{frame:03d}_"):
            return d
    raise FileNotFoundError(f"no grid phase dir for frame {frame}")


def make_figure(grid_eval_dir: Path, frame: int, exp_name: str, out_dir: Path):
    grid_eval_dir = Path(grid_eval_dir)
    phase_dir = find_phase_dir(frame)
    meta = json.load(open(phase_dir / "meta.json"))
    alphas = [f["alpha_degree"] for f in meta["frames"]]
    n_views = len(alphas)

    metrics = json.load(open(grid_eval_dir / "metrics.json"))
    psnr_by_idx = {r["idx"]: r["psnr"] for r in metrics["per_image"]}
    vpsnr_by_idx = {r["idx"]: r.get("v_psnr", float("nan")) for r in metrics["per_image"]}

    # grid 验证集按 phase_stride=20 采样，每 phase 10 视角
    idx_start = (frame // 20) * 10

    fig, axes = plt.subplots(n_views, 2, figsize=(7, 2.8 * n_views))
    for i in range(n_views):
        idx = idx_start + i
        gt = load_gray(grid_eval_dir / "gt" / f"val_{idx:03d}.png")
        pred = load_gray(grid_eval_dir / "renders" / f"val_{idx:03d}.png")
        p = psnr_by_idx.get(idx, float("nan"))
        vp = vpsnr_by_idx.get(idx, float("nan"))

        for j, im in enumerate((gt, pred)):
            ax = axes[i, j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if j == 0:
                ax.set_title(f"α={alphas[i]:+.1f}°", fontsize=9, pad=4)
            else:
                ax.set_title(f"PSNR={p:.2f}\nvPSNR={vp:.2f}", fontsize=9, pad=4)

    for j, name in enumerate(["GT", exp_name]):
        fig.text((j + 0.5) / 2.0, 0.985, name, ha="center", va="top",
                 fontsize=12, fontweight="bold")
    fig.suptitle(f"grid frame {frame} (phase {meta['phase']:.3f})", fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / f"frame_{frame}_grid_compare_{exp_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--grid_eval_dir", type=Path, required=True)
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--frames", type=int, nargs="+", default=[20, 100])
    p.add_argument("--out_dir", type=Path, default=ROOT / "dev_grid_views/stage2-flow")
    args = p.parse_args()
    for f in args.frames:
        make_figure(args.grid_eval_dir, f, args.exp_name, args.out_dir)


if __name__ == "__main__":
    main()
