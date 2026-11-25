"""phase 0 单视角：用 flow 参考区分已充盈/未充盈分支，验证时间泄漏。

flow 参考 = example_only_flow_grids（连续造影剂浓度，暗=浓度高）。
flow_contrast = 1 - flow/255 ∈ [0,1]（1=已充盈，0=无造影剂）。

在 phase 0（造影剂刚流入）：
  - 已充盈主干 (flow_contrast 高)：GT 应暗；pred 偏亮 = 欠充盈（误差>0）
  - 未充盈末梢 (flow_contrast 低)：GT 应亮；pred 偏暗 = 错误充盈/时间泄漏（误差<0）

用法：
    pixi run python scripts/make_stage2_phase0_flowcheck.py \
        --frame 0 --alpha 120 \
        --experiments stage2-base stage2-full stage2-full-smooth stage2-full-smooth-nogate stage2-joint
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/media/F/sj/Code/GS-contrast-flow")
GRID_DIR = ROOT / "data/example_grids/grid"
FLOW_DIR = ROOT / "data/example_only_flow_grids/grid"
EVAL_DIR = ROOT / "outputs/stage2_grid_eval"

FILLED_TH = 0.5   # flow_contrast > 0.5 视为已充盈
EMPTY_TH = 0.2    # flow_contrast < 0.2 视为未充盈


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0


def find_phase_dir(base: Path, frame: int) -> Path:
    for d in sorted(base.iterdir()):
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
    args = p.parse_args()

    gd = find_phase_dir(GRID_DIR, args.frame)
    fd = find_phase_dir(FLOW_DIR, args.frame)
    vi = find_view_index(gd, args.alpha)
    idx = (args.frame // 20) * 10 + vi

    gt = load_gray(gd / "dsa" / f"{vi:03d}.png")
    label = load_gray(gd / "label" / f"{vi:03d}.png")
    flow = load_gray(fd / "dsa" / f"{vi:03d}.png")
    mask = label > 0.5

    fc = 1.0 - flow  # 造影剂浓度
    vessel = mask
    filled = vessel & (fc > FILLED_TH)
    empty = vessel & (fc < EMPTY_TH)

    print(f"frame={args.frame} alpha={args.alpha:+.1f} view={vi}")
    print(f"mask 内像素: {vessel.sum()}  已充盈(>0.5): {filled.sum()}  未充盈(<0.2): {empty.sum()}")
    print(f"GT: filled_mean={gt[filled].mean():.4f}  empty_mean={gt[empty].mean():.4f} "
          f"(filled 应暗、empty 应亮)")

    print(f"\n{'experiment':26s} {'filled_err':>11s} {'empty_err':>10s} {'empty暗比':>9s}")
    for exp in args.experiments:
        pred = load_gray(EVAL_DIR / exp / "renders" / f"val_{idx:03d}.png")
        # 已充盈主干：pred-gt，正=欠充盈(偏亮)
        filled_err = float((pred[filled] - gt[filled]).mean())
        # 未充盈末梢：pred-gt，负=错误充盈(偏暗)
        empty_err = float((pred[empty] - gt[empty]).mean())
        # 未充盈末梢中，pred 明显比 GT 暗的像素占比（错误充盈/时间泄漏）
        dark_ratio = float((pred[empty] < gt[empty] - 0.02).mean())
        print(f"{exp:26s} {filled_err:+11.4f} {empty_err:+10.4f} {dark_ratio:9.3f}")

    print("\nfilled_err>0 = 主干欠充盈(偏亮)；empty_err<0 或 dark_ratio 高 = 末梢被错误充盈(时间泄漏)")


if __name__ == "__main__":
    main()
