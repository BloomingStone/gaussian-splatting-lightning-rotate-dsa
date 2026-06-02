from dataclasses import dataclass
from typing import Dict, Literal, Any, Tuple

import torch
from pytorch_lightning import LightningModule

from internal.dataparsers.dataparser import BatchT
from .metric import Metric, MetricImpl, CommonImageMetricImpl
from ..renderers.deformabel_xray_renderer_coronary_props import XrayRendererOuputs


@dataclass
class RotateXrayMetricsWithMasks(Metric):
    w_gray_loss: float = 1.0
    w_ssim_loss: float = 1.0

    rgb_diff_loss: Literal["l1", "l2"] = "l1"

    lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex"
    fused_ssim: bool = True

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return RotateXrayMetricsWithMasksImpl(self)


class RotateXrayMetricsWithMasksImpl(CommonImageMetricImpl):
    config: RotateXrayMetricsWithMasks

    @staticmethod
    def _mask_to_nchw(mask: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(ref)
        mask = mask.to(device=ref.device)
        mask = CommonImageMetricImpl._ensure_gray_nchw(mask)
        return mask.to(dtype=ref.dtype)

    def _masked_basic_metrics(
        self,
        batch: BatchT,
        outputs: XrayRendererOuputs,
    ) -> tuple[dict[str, torch.Tensor], dict[str, bool]]:
        _, image_info, _ = batch
        _, gt_image, mask = image_info

        gt_image = self._ensure_gray_nchw(gt_image)
        pred_gray = self._ensure_gray_nchw(outputs.gray_image)
        mask_nchw = self._mask_to_nchw(mask, pred_gray)

        mask_sum = mask_nchw.sum().clamp_min(1.0)
        gray_loss = torch.abs(pred_gray - gt_image) * mask_nchw
        gray_loss = gray_loss.sum() / mask_sum

        masked_pred = pred_gray * mask_nchw
        masked_gt = gt_image * mask_nchw
        ssim_metric = self.ssim(masked_pred, masked_gt)
        ssim_loss = 1.0 - ssim_metric

        loss = self.config.w_gray_loss * gray_loss + self.config.w_ssim_loss * ssim_loss

        assert not torch.isnan(loss), "Loss is NaN!"

        metrics = {
            "loss": loss,
            "mask_gray_loss": gray_loss,
            "mask_ssim_loss": ssim_loss,
            "mask_ratio": mask_nchw.mean(),
        }
        prog_bar = {
            "loss": True,
            "mask_gray_loss": True,
            "mask_ssim_loss": True,
            "mask_ratio": False,
        }
        return metrics, prog_bar

    def get_train_metrics(
        self,
        pl_module: LightningModule,
        gaussian_model,
        step: int,
        batch: BatchT,
        outputs: XrayRendererOuputs,
    ) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        del pl_module, gaussian_model, step
        metrics_base, prog_bar_base = self._masked_basic_metrics(batch, outputs)
        metrics_mask, prog_bar_mask = self._masked_basic_metrics(batch, outputs)
        metrics = {**metrics_base, **metrics_mask}
        prog_bar = {**prog_bar_base, **prog_bar_mask}
        metrics["loss"] = metrics_base["loss"] + metrics_mask["loss"]
        prog_bar["loss"] = True
        return metrics, prog_bar

    def get_validate_metrics(
        self,
        pl_module: LightningModule,
        gaussian_model,
        batch: BatchT,
        outputs: XrayRendererOuputs,
    ) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        del pl_module, gaussian_model
        metrics_base, prog_bar_base = self._masked_basic_metrics(batch, outputs)
        metrics_mask, prog_bar_mask = self._masked_basic_metrics(batch, outputs)
        metrics = {**metrics_base, **metrics_mask}
        prog_bar = {**prog_bar_base, **prog_bar_mask}
        metrics["loss"] = metrics_base["loss"] + metrics_mask["loss"]
        prog_bar["loss"] = True
        

        _, image_info, _ = batch
        _, gt_image, _ = image_info
        gt_image = self._ensure_gray_nchw(gt_image)

        self.add_image_validation_metrics(metrics, prog_bar, outputs.gray_image, gt_image)
        return metrics, prog_bar
