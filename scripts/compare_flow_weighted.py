#!/usr/bin/env python
"""用 flow-only DRR 作为权重，重新评估前 40 帧验证集上的 2D 投影指标。

数据约定：
- ``data/multi_layers/rbf_reader_flow_contrast_LCA/rotate_dsa/*.png``: 完整 DRR（背景+血管+流动造影剂）
- ``data/multi_layers/rbf_only_flow/rotate_dsa/*.png``: 仅流动造影剂的 DRR（白=背景，越黑密度越大）
- 渲染预测与完整 DRR 同为 [0,1]，白=背景、黑=密度。

权重 W = (255 - flow_img)/255，高密度(流动/血管)处接近 1，背景处为 0。
在 W 上计算加权 L1 / PSNR / SSIM。

用法::

    python scripts/compare_flow_weighted.py \
        outputs/flowml_baseline outputs/flowml_stage1 outputs/flowml_stage2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from internal.dataparsers.xray_dataparser.parser import XRayDataParserBuilder
from internal.dataparsers.xray_dataparser.meta import XRayMetaLoader
from internal.dataparsers.xray_dataparser.cloud_parsers import RandomCloudParser
from internal.dataparsers.xray_dataparser.splitters import FrameRangeSpliter
from internal.dataparsers.xray_dataparser.cameras_builder import RotateXRayCamerasBuilder
from internal.dataparsers.xray_dataparser.datasets import ImagesDatasetBuilder, ImagesDatasetConfig
from internal.renderers.deformabel_xray_renderer_with_flow import (
    DeformableXrayRendererWithFlow,
    DeformableXrayRendererWithFlowConfig,
)
from internal.deform_models.deform_with_flow.hashgrid_deform import HashGridDeformConfig
from internal.utils.gaussian_utils.gaussian_model_loader import GaussianModelLoader

DATA_DIR = Path("data/multi_layers/rbf_reader_flow_contrast_LCA")
FLOW_DIR = Path("data/multi_layers/rbf_only_flow")

# 每个 run 的 deform 配置开关（与训练 config 一致）
RUN_FLAGS = {
    "baseline": {"enable_flow": True, "freeze_motion": False},
    "stage1": {"enable_flow": False, "freeze_motion": False},
    "stage2": {"enable_flow": True, "freeze_motion": True},
}


def build_dataparser(data_dir: Path):
    parser = XRayDataParserBuilder(
        meta_loader=XRayMetaLoader(),
        cloud_parser=RandomCloudParser(num_points=10_000),
        spliter=FrameRangeSpliter(start_frame=0, end_frame=40, val_ratio=0.2),
        cameras_builder=RotateXRayCamerasBuilder(),
        dataset_builder=ImagesDatasetBuilder(dataset_config=ImagesDatasetConfig(image_uint8=False)),
        filter_visible_points=False,
    ).build()
    return parser.get_outputs(data_dir)


def search_checkpoint(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    assert candidates, f"no checkpoint found in {run_dir}"
    return candidates[-1]


def load_model_and_renderer(ckpt_path: Path, enable_flow: bool, freeze_motion: bool, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    gaussian_model = GaussianModelLoader.initialize_model_from_checkpoint(checkpoint, device)
    gaussian_model.eval()

    deform_cfg = HashGridDeformConfig(
        t_phase_combined=False,
        phase_multires=6,
        x_multires=13,
        log2_hashmap_size=19,
        max_resolution=2048,
        enable_flow=enable_flow,
        freeze_motion=freeze_motion,
    )
    renderer = DeformableXrayRendererWithFlow(
        optimization_config=DeformableXrayRendererWithFlowConfig(),
        deform_model_config=deform_cfg,
    )
    # 直接实例化 deform_model（eval 阶段无需 renderer.setup / 优化器）
    renderer.deform_model = deform_cfg.instantiate()
    renderer.deform_model.configure_trainable()

    prefix = "renderer.deform_model."
    deform_state = {
        key[len(prefix):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    assert deform_state, "no deform model state found in checkpoint"
    renderer.deform_model.load_state_dict(deform_state)

    renderer.to(device)
    renderer.eval()
    return gaussian_model, renderer


@torch.no_grad()
def render_frame(renderer, gaussian_model, camera, device):
    camera = camera.to_device(device)
    outputs = renderer(
        camera,
        gaussian_model,
        bg_color=torch.zeros(3, device=device),
    )
    # 原始灰度（未经过 visualizer 归一化）
    pred = outputs.gray_image.cpu()  # (1,1,H,W)
    return pred.squeeze().numpy().astype(np.float64)


def load_gray(path: Path) -> np.ndarray:
    img = np.array(Image.open(path), dtype=np.float64)
    if img.ndim == 3:
        img = img[..., 0]
    return img / 255.0


def load_flow_weight(flow_dir: Path, frame_idx: int) -> np.ndarray:
    path = flow_dir / "rotate_dsa" / f"{frame_idx:03d}.png"
    img = load_gray(path)
    weight = 1.0 - img  # 白背景→0，黑(流动)→1
    return weight


def weighted_metrics(pred: np.ndarray, gt: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    w = weight.astype(np.float64)
    w_sum = w.sum() + 1e-12

    diff = pred - gt
    w_l1 = float((np.abs(diff) * w).sum() / w_sum)

    mse = float(((diff ** 2) * w).sum() / w_sum)
    w_psnr = float(-10.0 * np.log10(mse + 1e-12))

    ssim_map = structural_similarity(
        gt, pred, data_range=1.0, full=True, gradient=False
    )[1]
    w_ssim = float((ssim_map * w).sum() / w_sum)

    return {"w_l1": w_l1, "w_psnr": w_psnr, "w_ssim": w_ssim, "w_sum": float(w_sum)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="run 输出目录（名称需含 baseline/stage1/stage2）")
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--flow-data", type=Path, default=FLOW_DIR)
    ap.add_argument("--save-preds", action="store_true", help="把渲染预测保存为 .npy")
    args = ap.parse_args()

    device = torch.device("cuda")

    outputs = build_dataparser(args.data)
    val_set = outputs.val_set
    print(f"val frames: {val_set.image_names}")

    # 每个 val frame 的 GT（完整 DRR）与 flow 权重
    frame_gt = {}
    frame_weight = {}
    for i, name in enumerate(val_set.image_names):
        frame_idx = int(name.split(".")[0])
        gt = val_set[i][1][1]  # (3,H,W) float [0,1]
        gt = gt[0].numpy().astype(np.float64)
        frame_gt[frame_idx] = gt
        frame_weight[frame_idx] = load_flow_weight(args.flow_data, frame_idx)

    rows = {}
    for run_dir in args.runs:
        run_dir = run_dir.resolve()
        name = run_dir.name
        flags = None
        for key, f in RUN_FLAGS.items():
            if key in name:
                flags = f
                break
        assert flags is not None, f"cannot infer run flags from {name} (expected baseline/stage1/stage2)"

        ckpt = search_checkpoint(run_dir)
        gaussian_model, renderer = load_model_and_renderer(
            ckpt, flags["enable_flow"], flags["freeze_motion"], device
        )

        agg = {"w_l1": 0.0, "w_psnr": 0.0, "w_ssim": 0.0, "w_sum": 0.0}
        per_frame = []
        for i, cam_item in enumerate(val_set):
            frame_idx = int(val_set.image_names[i].split(".")[0])
            camera = cam_item[0]
            pred = render_frame(renderer, gaussian_model, camera, device)
            if args.save_preds:
                np.save(run_dir / f"pred_{frame_idx:03d}.npy", pred)
            m = weighted_metrics(pred, frame_gt[frame_idx], frame_weight[frame_idx])
            per_frame.append((frame_idx, m))
            for k in ("w_l1", "w_psnr", "w_ssim", "w_sum"):
                agg[k] += m[k]

        n = len(per_frame)
        rows[name] = {k: v / n for k, v in agg.items()}
        print(f"\n[{name}] ckpt={ckpt.name}")
        print(f"  {'frame':>6} {'w_l1':>10} {'w_psnr':>9} {'w_ssim':>9}")
        for frame_idx, m in per_frame:
            print(f"  {frame_idx:>6} {m['w_l1']:>10.4f} {m['w_psnr']:>9.3f} {m['w_ssim']:>9.4f}")

    print("\n==== 汇总（flow 加权，前 40 帧 valid 帧均值）====")
    print(f"{'run':<20} {'w_l1':>10} {'w_psnr':>9} {'w_ssim':>9}")
    for name, m in rows.items():
        print(f"{name:<20} {m['w_l1']:>10.4f} {m['w_psnr']:>9.3f} {m['w_ssim']:>9.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
