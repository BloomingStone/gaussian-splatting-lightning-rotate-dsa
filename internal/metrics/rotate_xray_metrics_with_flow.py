from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any

import torch
from lightning import LightningModule

from internal.metrics.metric import Metric, MetricImpl, CommonImageMetricImpl
from internal.renderers.deformabel_xray_renderer import  XrayRendererOuputs
from internal.dataparsers.dataparser import BatchT


@dataclass
class RotateXrayMetrics(Metric):
    w_gray_loss: float = 1.0
    w_ssim_loss: float = 1.0
    
    w_d_density: float = 0.0
    w_xyz_var: float = 0.0

    rgb_diff_loss: Literal["l1", "l2"] = "l1"

    lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex"
    """
    the vanilla 3DGS uses 'vgg', but 'alex' is faster
    """

    fused_ssim: bool = False

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
        
        # regularization losses on deforms variance
        d_xyz_var = outputs.deforms_var["d_xyz"]
        d_density_std = outputs.deforms_var["d_density"].sqrt()
        d_density_mean = outputs.deforms_mean["d_density"].abs()
        d_xyz_var_avg = d_xyz_var.mean()
        
        # use mean + 2*std as the uncertainty-aware density, which is roughly the upper bound of density with 95% confidence if 
        # we assume the deforms follow a normal distribution. 
        # This is to avoid under-estimate the density for points with high uncertainty.
        d_density_upper= d_density_mean + 2 * d_density_std
        d_density_upper_avg = d_density_upper.mean()

        loss = (
            # image loss
            self.config.w_gray_loss * gray_loss +
            self.config.w_ssim_loss * ssim_loss + 
            
            # regularization loss
            self.config.w_d_density * d_density_upper_avg + 
            self.config.w_xyz_var * d_xyz_var_avg
        )


        assert not torch.isnan(loss), "Loss is NaN!"
        
        metrics = {
            "loss": loss,
            "gray_loss": gray_loss,
            "ssim_loss": ssim_loss,
            "d_density_upper": d_density_upper_avg,
            "d_density_mean": d_density_mean.mean(),
            "d_density_std": d_density_std.mean(),
            "xyz_var": d_xyz_var_avg
        }


        prog_bar = {
            "loss": True,
            "gray_loss": True,
            "ssim_loss": True,
            "d_density_upper": True,
            "d_density_mean": False,
            "d_density_std": False,
            "xyz_var": True,
        }
        return metrics, prog_bar
    

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        return self._get_basic_metrics(
            pl_module=pl_module,    # type: ignore
            gaussian_model=gaussian_model,
            batch=batch,
            outputs=outputs     # type: ignore
        )
    
    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
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