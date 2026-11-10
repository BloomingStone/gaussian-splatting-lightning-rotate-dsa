"""Visualise the artifact metrics (stage1 vs r2g) across frame-100 views,
ordered by C-arm angle. Highlights how spike_score flags stage1 needle artifacts
while v_ssim / edge_sharpness flag r2g vessel misalignment."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/media/F/sj/Code/GS-contrast-flow")
OUT_DIR = ROOT / "outputs/comparison"

PANELS = [
    ("spike_score", "spike_score  (needle/spike artifact, lower better)", "r"),
    ("v_ssim", "v_ssim  (vessel misalignment/ghost, higher better)", "b"),
    ("ghost_score", "ghost_score  (double-edge ghost, lower better)", "c"),
    ("edge_contrast", "edge_contrast  (under-reconstruction blur, higher better)", "y"),
    ("edge_sharpness", "edge_sharpness  (misalign/blur, higher better)", "g"),
    ("v_psnr", "v_psnr  (vessel intensity error, higher better)", "m"),
]


def main(args):
    rows = json.load(open(args.metrics_dir / "artifact_metrics.json"))
    names = [r["name"] for r in rows]
    x = np.arange(len(names))

    fig, axes = plt.subplots(6, 1, figsize=(14, 24))
    for ax, (key, title, color) in zip(axes, PANELS):
        s1 = [r["stage1"][key] for r in rows]
        r2 = [r["r2g"][key] for r in rows]
        w = 0.38
        ax.bar(x - w / 2, s1, w, label="stage1", color="steelblue")
        ax.bar(x + w / 2, r2, w, label="r2g", color="darkorange")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        # zoom the y-axis to the data range so differences are visible
        lo, hi = min(min(s1), min(r2)), max(max(s1), max(r2))
        pad = (hi - lo) * 0.12 + 1e-9
        ax.set_ylim(lo - pad, hi + pad)

    fig.suptitle("frame_100 (phase 0.667) artifact metrics: stage1 vs r2g",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = args.out_dir / f"artifact_metrics_compare_{args.exp_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", default=OUT_DIR, type=Path)
    p.add_argument("--out_dir", default=OUT_DIR, type=Path)
    p.add_argument("--exp_name", default="")
    main(p.parse_args())
