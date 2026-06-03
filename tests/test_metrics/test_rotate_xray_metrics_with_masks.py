from types import SimpleNamespace

import torch

from internal.dataparsers.dataparser import ImageItemT, ItemT
from internal.metrics.rotate_xray_metrics_frangi_masks import RotateXrayMetricsWithMasks, RotateXrayMetricsWithMasksImpl
from internal.metrics.ssim import ssim


def _soft_mask_from_mask(hard_mask: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Simplified soft mask for testing: Gaussian blur of the hard mask."""
    from scipy.ndimage import distance_transform_edt
    import numpy as np
    mask_np = hard_mask[0].cpu().numpy().astype(bool)
    dist = distance_transform_edt(~mask_np)
    weight = np.where(mask_np, 1.0, np.exp(-dist ** 2 / (2.0 * sigma ** 2)))
    return torch.from_numpy(weight).float()


def test_masked_metric_ignores_outside_mask_errors():
    """Fallback path: no extra_data → uses image mask."""
    metric = RotateXrayMetricsWithMasks()
    impl = RotateXrayMetricsWithMasksImpl(metric)
    impl.ssim = ssim

    gt_image = torch.zeros(1, 16, 16)
    pred_image = torch.ones(1, 16, 16)
    mask = torch.zeros(3, 16, 16, dtype=torch.bool)
    mask[:, 4:12, 4:12] = True

    batch = ItemT(
        camera=None,
        image=ImageItemT(image_name="test.png", gt_image=gt_image, mask=mask),
        extra_data=None,
    )
    outputs = SimpleNamespace(gray_image=pred_image * (~mask[:1]).float())

    metrics, _ = impl._masked_basic_metrics(batch, outputs)

    assert torch.isclose(metrics["mask_gray_loss"], torch.tensor(0.0))
    assert torch.isclose(metrics["mask_ssim_loss"], torch.tensor(0.0))


def test_masked_metric_uses_frangi_soft_mask():
    """When extra_data provides frangi_soft_mask, it takes precedence over image mask."""
    metric = RotateXrayMetricsWithMasks()
    impl = RotateXrayMetricsWithMasksImpl(metric)
    impl.ssim = ssim

    gt_image = torch.zeros(1, 16, 16)
    pred_image = torch.ones(1, 16, 16)

    # hard mask covers top-left quadrant
    hard_mask = torch.zeros(3, 16, 16, dtype=torch.bool)
    hard_mask[:, :8, :8] = True
    # soft mask covers the whole image (uniform weight → same as no mask)
    soft_mask = torch.ones(16, 16)  # no soft falloff → all 1

    batch = ItemT(
        camera=None,
        image=ImageItemT(image_name="test.png", gt_image=gt_image, mask=hard_mask),
        extra_data={"frangi_soft_mask": soft_mask},
    )
    # pred outside hard mask → would be wrong if hard mask were used
    outputs = SimpleNamespace(gray_image=torch.full((1, 16, 16), 0.5))

    metrics, _ = impl._masked_basic_metrics(batch, outputs)

    # soft_mask is all 1 → gray_loss = |0.5 - 0| = 0.5
    assert torch.isclose(metrics["mask_gray_loss"], torch.tensor(0.5), atol=1e-6)
    # hard mask would mask out pred → loss = 0, but frangi_soft_mask takes priority
    assert not torch.isclose(metrics["mask_gray_loss"], torch.tensor(0.0))