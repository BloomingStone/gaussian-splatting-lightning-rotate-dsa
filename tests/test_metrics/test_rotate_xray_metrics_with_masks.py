from types import SimpleNamespace

import torch

from internal.dataparsers.dataparser import ImageItemT, ItemT
from internal.metrics.rotate_xray_metrics_with_masks import RotateXrayMetricsWithMasks, RotateXrayMetricsWithMasksImpl
from internal.metrics.ssim import ssim


def test_masked_metric_ignores_outside_mask_errors():
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