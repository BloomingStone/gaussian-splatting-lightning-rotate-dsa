from pathlib import Path
from types import SimpleNamespace

import torch
from matplotlib import pyplot as plt
import numpy as np

from internal.dataparsers.dataparser import ImageItemT, ItemT
from internal.dataparsers.xray_dataparser.cameras_builder import RotateXRayCamerasBuilder
from internal.dataparsers.xray_dataparser.datasets import (
    FrangiImagesDatasetBuilder,
    FrangiImagesDatasetConfig,
)
from internal.dataparsers.xray_dataparser.meta import XRayMetaLoader
from internal.metrics.rotate_xray_metrics_weight_patch import (
    RotateXrayMetricsWeightPatch,
    RotateXrayMetricsWeightPatchImpl,
)
from internal.metrics.ssim import ssim


def _make_batch(
    gt_image: torch.Tensor,
    pred_image: torch.Tensor,
    weight_map: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> ItemT:
    if mask is None:
        mask = torch.ones(3, *gt_image.shape[-2:], dtype=torch.bool)
    extra = {"weight_map": weight_map} if weight_map is not None else None
    return ItemT(
        camera=None,
        image=ImageItemT(image_name="test.png", gt_image=gt_image, mask=mask),
        extra_data=extra,
    )


def _make_outputs(gray_image: torch.Tensor):
    return SimpleNamespace(gray_image=gray_image)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_patch_metric_raises_no_weight_map():
    """Without a weight map the patch loss should be zero."""
    metric = RotateXrayMetricsWeightPatch(num_patches=5, patch_divisor=16)
    impl = RotateXrayMetricsWeightPatchImpl(metric)
    impl.setup("train", None)
    impl.ssim = ssim

    gt_image = torch.zeros(1, 64, 64)
    pred_image = torch.ones(1, 64, 64)

    batch = _make_batch(gt_image, pred_image, weight_map=None)
    outputs = _make_outputs(pred_image)

    metrics, _ = impl._weight_patch_metrics(batch, outputs)
    assert metrics["patch_gray_loss"] == 0.0
    assert metrics["patch_ssim_loss"] == 0.0


def test_patch_metric_samples_in_high_weight_region():
    """Patches should be drawn from the region with high weight."""
    metric = RotateXrayMetricsWeightPatch(num_patches=20, patch_divisor=16)
    impl = RotateXrayMetricsWeightPatchImpl(metric)
    impl.setup("train", None)
    impl.ssim = ssim

    gt_image = torch.zeros(1, 64, 64)
    pred_image = torch.ones(1, 64, 64)

    # weight map: only the centre 16×16 is high
    wmap = torch.zeros(64, 64)
    wmap[24:40, 24:40] = 10.0

    batch = _make_batch(gt_image, pred_image, weight_map=wmap)
    outputs = _make_outputs(pred_image)

    cy, cx = impl._sample_patches(wmap, patch_size=8, num_patches=100, weight_power=1.0)
    # all sampled centres should be inside the high-weight zone
    assert all(24 <= y < 40 for y in cy.tolist())
    assert all(24 <= x < 40 for x in cx.tolist())


def test_patch_metric():
    """Run the metric on a synthetic image and plot the sampled patch boxes."""
    metric = RotateXrayMetricsWeightPatch(
        num_patches=10,
        patch_divisor=8,
        w_gray_loss=1.0,
        w_ssim_loss=1.0,
        w_patch_loss=1.0,        
        weight_power=1.0,
    )
    impl = RotateXrayMetricsWeightPatchImpl(metric)
    impl.setup("train", None)
    impl.ssim = ssim

    # 128×128 image with a dark vertical bar (simulated coronary)
    gt_image = torch.full((1, 128, 128), 0.85)
    gt_image[:, :, 56:72] = 0.15
    pred_image = gt_image.clone()  # perfect prediction → zero loss

    # weight map — strongest on the dark bar
    wmap = torch.zeros(128, 128)
    wmap[:, 52:76] = 5.0  # slightly wider than the dark bar

    batch = _make_batch(gt_image, pred_image, weight_map=wmap)
    outputs = _make_outputs(pred_image)

    metrics, _ = impl._weight_patch_metrics(batch, outputs)     # type: ignore

    # With perfect prediction all losses should be near zero
    assert torch.isclose(metrics["gray_loss"], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(metrics["patch_gray_loss"], torch.tensor(0.0), atol=1e-6)


def test_patch_metric_with_real_fixture_data(test_xray_data_no_flow_root: Path, output_root: Path):
    """Build a real dataset with Frangi weight map and visualise sampled patches."""
    from matplotlib import pyplot as plt

    # ── build real dataset ────────────────────────────────────────────
    meta = XRayMetaLoader().load(test_xray_data_no_flow_root)
    cameras = RotateXRayCamerasBuilder().build_cameras(meta)
    indices = list(range(0, len(cameras), 40))  # thin out for speed

    builder = FrangiImagesDatasetBuilder(
        image_dir_name="rotate_dsa",
        image_suffix="*.png",
        dataset_config=FrangiImagesDatasetConfig(
            image_uint8=False,
            frangi_threshold=0.03,
        ),
    )
    dataset = builder.build_dataset(test_xray_data_no_flow_root, cameras, meta, indices, "train")

    # ── metric setup ──────────────────────────────────────────────────
    metric = RotateXrayMetricsWeightPatch(num_patches=30)
    impl = RotateXrayMetricsWeightPatchImpl(metric)
    impl.setup("val", None)
    impl.ssim = ssim

    # ── run metric on the first item ──────────────────────────────────
    item = dataset[0]
    _, gt_image, _ = item.image
    assert item.extra_data is not None and "weight_map" in item.extra_data

    # perfect prediction => all losses should be (near) zero
    outputs = SimpleNamespace(gray_image=gt_image.unsqueeze(0))  # (C, H, W) → (1, C, H, W)
    batch = ItemT(
        camera=item.camera,
        image=ImageItemT(
            image_name=item.image.image_name,
            gt_image=gt_image,
            mask=item.image.mask,
        ),
        extra_data=item.extra_data,
    )

    metrics, _ = impl._weight_patch_metrics(batch, outputs)

    assert torch.isclose(metrics["gray_loss"], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(metrics["ssim_loss"], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(metrics["patch_gray_loss"], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(metrics["patch_ssim_loss"], torch.tensor(0.0), atol=1e-6)
    assert metrics["num_patches"] > 0

    # ── visualise ─────────────────────────────────────────────────────
    extra = item.extra_data
    gt_np = gt_image[0].cpu().numpy()
    wmap_np = extra["weight_map"].cpu().numpy()  # type: ignore[index]
    frangi_np = extra["weight_frangi"].cpu().numpy()  # type: ignore[index]

    patch_side = max(gt_image.shape[1], gt_image.shape[2]) // metric.patch_divisor
    half = patch_side // 2
    cy, cx = impl._sample_patches(torch.from_numpy(wmap_np), patch_side, metric.num_patches, weight_power=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # top-left: Frangi
    axes[0, 0].imshow(frangi_np, cmap="hot")
    axes[0, 0].set_title("Frangi vesselness")
    axes[0, 0].axis("off")
    # top-right: weight map
    axes[0, 1].imshow(wmap_np, cmap="hot")
    axes[0, 1].set_title("Weight map")
    axes[0, 1].axis("off")
    # bottom-left: GT + patches
    axes[1, 0].imshow(gt_np, cmap="gray")
    axes[1, 0].set_title("GT + patches")
    for y, x in zip(cy.tolist(), cx.tolist()):
        rect = plt.Rectangle((x - half, y - half), patch_side, patch_side,
                             fill=False, edgecolor="red", linewidth=1)
        axes[1, 0].add_patch(rect)
    axes[1, 0].axis("off")
    # bottom-right: overlay
    axes[1, 1].imshow(gt_np, cmap="gray", alpha=0.6)
    axes[1, 1].imshow(wmap_np, cmap="hot", alpha=0.4)
    axes[1, 1].set_title("Overlay")
    axes[1, 1].axis("off")

    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    out_dir = output_root / "weight_patch_metric_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "real_patch_sampling.png", bbox_inches="tight")
    plt.close(fig)
