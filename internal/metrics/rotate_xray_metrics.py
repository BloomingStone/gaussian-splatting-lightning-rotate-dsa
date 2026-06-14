from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any

import torch
from pytorch_lightning import LightningModule
import numpy as np

from internal.metrics.metric import Metric, MetricImpl, CommonImageMetricImpl
from internal.dataparsers.dataparser import BatchT
from ..renderers.deformabel_xray_renderer import XrayRendererOuputs


@dataclass
class RotateXrayMetrics(Metric):
    w_gray_loss: float = 1.0
    w_ssim_loss: float = 1.0

    rgb_diff_loss: Literal["l1", "l2"] = "l1"

    lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex"
    """
    the vanilla 3DGS uses 'vgg', but 'alex' is faster
    """

    # 目前如果将 fused_ssim 设为 False 会导致 loss 无法正常下降， 可能是因为 pytorch 实现的ssim处理半精度时有问题
    fused_ssim: bool = True

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return RotateXrayMetricsImpl(self)


class RotateXrayMetricsImpl(CommonImageMetricImpl):
    config:  RotateXrayMetrics

    def _get_basic_metrics(
        self, 
        pl_module: LightningModule, 
        gaussian_model, 
        batch: BatchT, 
        outputs: XrayRendererOuputs
    ):
        _, image_info, _ = batch   # load depth_map as extra_data in internal/dataparsers/rotated_xray_dataparser.py
        _, gt_image, _ = image_info
        gt_image = self._ensure_gray_nchw(gt_image)
        pred_gray = self._ensure_gray_nchw(outputs.gray_image)
        
        gray_loss = self.rgb_diff_loss_fn(pred_gray, gt_image)
        
        ssim_metric = self.ssim(pred_gray, gt_image)
        ssim_loss = 1.0 - ssim_metric

        loss = (
            # image loss
            self.config.w_gray_loss * gray_loss +
            self.config.w_ssim_loss * ssim_loss
        )


        assert not torch.isnan(loss), "Loss is NaN!"
        
        metrics = {
            "loss": loss,
            "gray_loss": gray_loss,
            "ssim_loss": ssim_loss,
        }


        prog_bar = {
            "loss": True,
            "gray_loss": True,
            "ssim_loss": True,
        }
        return metrics, prog_bar
    

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch: BatchT, outputs: XrayRendererOuputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:    # type: ignore
        return self._get_basic_metrics(
            pl_module=pl_module,    # type: ignore
            gaussian_model=gaussian_model,
            batch=batch,
            outputs=outputs
        )
    
    def get_validate_metrics(self, pl_module, gaussian_model, batch: BatchT, outputs: XrayRendererOuputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, prog_bar = self._get_basic_metrics(
            pl_module,
            gaussian_model,
            batch,
            outputs
        )

        _, image_info, extra_data = batch
        _, gt_image, _ = image_info
        gt_image = self._ensure_gray_nchw(gt_image)

        self.add_image_validation_metrics(metrics, prog_bar, outputs.gray_image, gt_image)

        weight = extra_data.get("soft_mask_weight") if extra_data is not None else None
        if weight is not None:
            self.add_weighted_validation_metrics(metrics, prog_bar, outputs.gray_image, gt_image, weight)

        metrics.update(self.metric3d)
        
        return metrics, prog_bar
    
    def on_validation_epoch_start(self, pl_module):
        self.metric3d = self._compute_3d_metrics(pl_module=pl_module)
        return super().on_validation_epoch_start(pl_module)
