"""phase 0 单视角 (指定 alpha) 的整图 LPIPS + 冠脉/背景对比度。

验证假设：full 渲染整体偏亮，冠脉虽亮但与背景对比不明显；smooth 冠脉暗后
与背景对比更明显。LPIPS 更接近人眼响应，可能比 PSNR 更敏感于这种对比度差异。

用法：
    pixi run python scripts/make_stage2_phase0_lpips.py \
        --frame 0 --alpha 120 \
        --experiments stage2-base stage2-full stage2-full-smooth stage2-full-smooth-nogate stage2-joint
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT = Path("/media/F/sj/Code/GS-contrast-flow")
GRID_DIR = ROOT / "data/example_grids/grid"
EVAL_DIR = ROOT / "outputs/stage2_grid_eval"


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0


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


def to_rgb_tensor(img: np.ndarray) -> torch.Tensor:
    """灰度 [0,1] -> (1, 3, H, W)，3 通道复制以满足 LPIPS 的 RGB 输入。"""
    return torch.from_numpy(img)[None, None].repeat(1, 3, 1, 1)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--alpha", type=float, default=120.0)
    p.add_argument("--experiments", type=str, nargs="+", required=True)
    p.add_argument("--net_type", type=str, default="alex", choices=["vgg", "alex", "squeeze"])
    args = p.parse_args()

    phase_dir = find_phase_dir(args.frame)
    vi = find_view_index(phase_dir, args.alpha)
    idx = (args.frame // 20) * 10 + vi

    gt = load_gray(phase_dir / "dsa" / f"{vi:03d}.png")
    label = load_gray(phase_dir / "label" / f"{vi:03d}.png")
    mask = label > 0.5

    # GT 的冠脉/背景对比度（X-ray：血管暗、背景亮）
    gt_vessel = gt[mask].mean()
    gt_bg = gt[~mask].mean()
    gt_contrast = (gt_bg - gt_vessel) / gt_bg
    print(f"frame={args.frame} alpha={args.alpha:+.1f} view_idx={vi} batch_idx={idx}")
    print(f"GT: vessel_mean={gt_vessel:.4f} bg_mean={gt_bg:.4f} "
          f"contrast=(bg-vessel)/bg = {gt_contrast:.4f}")

    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type=args.net_type)

    print(f"\n{'experiment':26s} {'LPIPS':>8s}  {'vessel_mean':>12s} {'bg_mean':>9s} {'contrast':>9s}")
    for exp in args.experiments:
        pred = load_gray(EVAL_DIR / exp / "renders" / f"val_{idx:03d}.png")

        # 整图 LPIPS（不叠加 mask）
        with torch.no_grad():
            lp = float(lpips(to_rgb_tensor(pred), to_rgb_tensor(gt)).item())

        p_vessel = pred[mask].mean()
        p_bg = pred[~mask].mean()
        p_contrast = (p_bg - p_vessel) / p_bg

        print(f"{exp:26s} {lp:8.4f}  {p_vessel:12.4f} {p_bg:9.4f} {p_contrast:9.4f}")

    print(f"\nGT contrast = {gt_contrast:.4f}（越接近说明冠脉-背景对比越真实）")


if __name__ == "__main__":
    main()
