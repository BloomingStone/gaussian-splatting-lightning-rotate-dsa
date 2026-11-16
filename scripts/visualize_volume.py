"""可视化 3D 体积：MIP（最大密度投影）三视图 + 中央切片。

用于直观检查重建体积的"毛刺"/细分支结构，对比 GT label 与预测体积。

用法：
    pixi run python scripts/visualize_volume.py \
        --vols GT:data/example_grids/coronary_label.nii.gz \
                spike:outputs/dev_grid_views/stage1-spike/vol3d/density_phase0.000.nii.gz \
        --out dev_grid_views/hashgrid-adaptive-smooth/volume_compare.png
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(vol_path: str) -> np.ndarray:
    img = nib.load(vol_path)
    return np.asarray(img.get_fdata(), dtype=np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return v
    return (v - lo) / (hi - lo)


def mip3(v: np.ndarray) -> list[np.ndarray]:
    return [v.max(axis=0), v.max(axis=1), v.max(axis=2)]


def main(args):
    specs = [s.split(":", 1) for s in args.vols]
    n = len(specs)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4.2 * n), squeeze=False)
    for i, (name, path) in enumerate(specs):
        v = load(path)
        vn = normalize(v)
        mips = mip3(vn)
        slices = [vn[vn.shape[0] // 2], vn[:, vn.shape[1] // 2], vn[:, :, vn.shape[2] // 2]]
        titles = ["MIP (z)", "MIP (y)", "MIP (x)"]
        for j in range(3):
            ax = axes[i, j]
            if args.mode == "mip":
                im = mips[j]
            else:
                im = slices[j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(titles[j], fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(name, fontsize=11, fontweight="bold", rotation=0, labelpad=40, va="center")

    fig.suptitle(f"3D volume comparison ({args.mode})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vols", nargs="+", required=True,
                   help="name:path pairs, e.g. GT:label.nii.gz spike:pred.nii.gz")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--mode", choices=["mip", "slice"], default="mip")
    main(p.parse_args())
