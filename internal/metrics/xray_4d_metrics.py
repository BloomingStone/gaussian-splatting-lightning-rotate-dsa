from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any, cast

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_lightning import LightningModule

from internal.utils.ssim import ssim
from internal.metrics.metric import Metric, MetricImpl
from internal.renderers.xray_4d_renderer import RenderRes
from internal.models.xray_4d_gaussian import Xray4DGaussianModel


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


class Xray4DMetricsImpl(MetricImpl):
    config:  Xray4DMetrics
    
    def __init__(self, config: Xray4DMetrics, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.no_state_dict_models = {}

    @staticmethod
    def _create_fused_ssim_adapter():
        from fused_ssim import fused_ssim
        def adapter(pred, gt):
            return fused_ssim(pred.unsqueeze(0), gt.unsqueeze(0))
        return adapter


    def setup(self, stage: str, pl_module):
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.no_state_dict_models["lpips"] = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type=self.config.lpips_net_type)

        
        self.rgb_diff_loss_fn = self._l1_loss
        if self.config.rgb_diff_loss == "l2":
            print("Use L2 loss")
            self.rgb_diff_loss_fn = self._l2_loss

        self.ssim = ssim
        if self.config.fused_ssim:
            print("Fused SSIM enabled")
            self.ssim = self._create_fused_ssim_adapter()
    

    def _get_basic_metrics(
        self, 
        pl_module: LightningModule, 
        gaussian_model, 
        batch, 
        outputs: RenderRes
    ):
        image_info: Tuple[str, torch.Tensor, torch.Tensor]
        _, image_info, _ = batch   # load depth_map as extra_data in internal/dataparsers/rotated_xray_dataparser.py
        _, gt_image, _ = image_info
        gt_image = gt_image[0:1]
        
        gray_loss = self.rgb_diff_loss_fn(outputs.gray_image, gt_image)
        
        ssim_metric = self.ssim(outputs.gray_image, gt_image)
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

        image_info: Tuple[str, torch.Tensor, torch.Tensor]
        _, image_info, _ = batch   # load depth_map as extra_data in internal/dataparsers/rotated_xray_dataparser.py
        _, gt_image, masked_pixels = image_info
        gt_image = gt_image[0:1]    # [1, H, W] get gray image

        metrics["psnr"] = self.psnr(outputs.gray_image, gt_image)
        prog_bar["psnr"] = True
        
        gray2rgb = outputs.gray_image.clamp(0., 1.)[None].repeat(1, 3, 1, 1)    # [1, 3, H, W]
        gray2rgb_gt = gt_image.clamp(0., 1.)[None].repeat(1, 3, 1, 1)
        metrics["lpips"] = self.no_state_dict_models["lpips"](gray2rgb, gray2rgb_gt)
        prog_bar["lpips"] = True

        return metrics, prog_bar
    
    def on_parameter_move(self, *args, **kwargs):
        if "lpips" in self.no_state_dict_models:
            self.no_state_dict_models["lpips"] = self.no_state_dict_models["lpips"].to(*args, **kwargs)
        
    @staticmethod
    def _l1_loss(predict: torch.Tensor, gt: torch.Tensor):
        return torch.abs(predict - gt).mean()

    @staticmethod
    def _l2_loss(predict: torch.Tensor, gt: torch.Tensor):
        return torch.mean((predict - gt) ** 2)