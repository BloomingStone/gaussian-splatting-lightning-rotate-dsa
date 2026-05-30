from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any, cast

import torch
from pytorch_lightning import LightningModule

from internal.metrics.metric import Metric, MetricImpl, CommonImageMetricImpl
from internal.renderers.xray_4d_renderer import RenderRes
from internal.models.xray_4d_gaussian import Xray4DGaussianModel
from internal.dataparsers.dataparser import BatchT


@dataclass
class Xray4DMetrics(Metric):
    w_gray_loss: float = 1.0
    w_ssim_loss: float = 1.0
    
    w_density_var: float = 0.0
    w_xyz_var: float = 0.0

    rgb_diff_loss: Literal["l1", "l2"] = "l1"

    lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex"
    """
    the vanilla 3DGS uses 'vgg', but 'alex' is faster
    """

    fused_ssim: bool = False

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return Xray4DMetricsImpl(self)


class Xray4DMetricsImpl(CommonImageMetricImpl):
    config:  Xray4DMetrics

    def _get_basic_metrics(
        self, 
        pl_module: LightningModule, 
        gaussian_model, 
        batch: BatchT, 
        outputs: RenderRes
    ):
        _, image_info, _ = batch   # load depth_map as extra_data in internal/dataparsers/rotated_xray_dataparser.py
        _, gt_image, _ = image_info
        gt_image = self._ensure_gray_nchw(gt_image)
        pred_gray = self._ensure_gray_nchw(outputs.gray_image)
        
        gray_loss = self.rgb_diff_loss_fn(pred_gray, gt_image)
        
        ssim_metric = self.ssim(pred_gray, gt_image)
        ssim_loss = 1.0 - ssim_metric

        d_xyz_var = outputs.deforms_var["d_xyz"][outputs.density_mask].mean()
        
        gaussian_model = cast(Xray4DGaussianModel, gaussian_model)
        density_var_mean = gaussian_model.get_density_res_energy()[outputs.density_mask].mean()

        loss = (
            # image loss
            self.config.w_gray_loss * gray_loss +
            self.config.w_ssim_loss * ssim_loss + 
            
            # regularization loss
            self.config.w_density_var * density_var_mean +
            self.config.w_xyz_var * d_xyz_var
        )


        assert not torch.isnan(loss), "Loss is NaN!"
        
        metrics = {
            "loss": loss,
            "gray_loss": gray_loss,
            "ssim_loss": ssim_loss,
            "density_var": density_var_mean,
            "xyz_var": d_xyz_var
        }


        prog_bar = {
            "loss": True,
            "gray_loss": True,
            "ssim_loss": True,
            "density_var": True,
            "xyz_var": True,
        }
        return metrics, prog_bar
    

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        return self._get_basic_metrics(
            pl_module=pl_module,
            gaussian_model=gaussian_model,
            batch=batch,
            outputs=outputs
        )
    
    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs: RenderRes) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, prog_bar = self._get_basic_metrics(
            pl_module,
            gaussian_model,
            batch,
            outputs
        )

        _, image_info, _ = batch   # load depth_map as extra_data in internal/dataparsers/rotated_xray_dataparser.py
        _, gt_image, _ = image_info
        gt_image = self._ensure_gray_nchw(gt_image)

        self.add_image_validation_metrics(metrics, prog_bar, outputs.gray_image, gt_image)

        return metrics, prog_bar