from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any

import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from monai.losses.dice import DiceCELoss
import lightning

from internal.utils.ssim import ssim
from .metric import Metric, MetricImpl
from ..configs.instantiate_config import InstantiatableConfig
from ..renderers.deformabel_xray_renderer import RenderRes
from ..cameras.cameras import Camera

@dataclass
class RotateXrayMetrics(Metric):
    w_gray_loss_whole: float = 1.0
    w_ssim_loss_whole: float = 1.0
    w_dice_loss_start: float = 1.0
    w_dice_loss_end: float = 0.1

    rgb_diff_loss: Literal["l1", "l2"] = "l1"

    lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex"
    """
    the vanilla 3DGS uses 'vgg', but 'alex' is faster
    """

    fused_ssim: bool = False
    
    max_step_of_dice_loss: int = 2000

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return RotateXrayMetricsImpl(self)


class RotateXrayMetricsImpl(MetricImpl):
    config:  RotateXrayMetrics
    def __init__(self, config: RotateXrayMetrics, *args, **kwargs) -> None:
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
        
        self.dice_loss_fn = DiceCELoss()
    
    def _get_basic_metrics(
        self, 
        pl_module: lightning.LightningModule, 
        gaussian_model, 
        batch, 
        outputs: RenderRes
    ):
        image_info: Tuple[str, torch.Tensor, torch.Tensor]
        _, image_info, _ = batch   # load depth_map as extra_data in internal/dataparsers/rotated_xray_dataparser.py
        _, gt_image, masked_pixels = image_info
        gt_image = gt_image[0:1]
        masked_pixels = masked_pixels[0:1]
        
        gray_loss_whole = self.rgb_diff_loss_fn(outputs.gray_image_whole, gt_image)
        
        ssim_metric_whole = self.ssim(outputs.gray_image_whole, gt_image)
        ssim_loss_whole = 1.0 - ssim_metric_whole
        
        soft_coronary_mask = torch.sigmoid(((1-outputs.gray_image_coronary) - 0.01) * 10)
        dice_loss = self.dice_loss_fn(soft_coronary_mask[None], masked_pixels[None])
        
        step_now = pl_module.global_step
        step_max = self.config.max_step_of_dice_loss
        w0, w1 = self.config.w_dice_loss_start, self.config.w_dice_loss_end
        if step_now >= step_max:
            w_dice_loss = w1
        else:
            w_dice_loss = w0 + (w1 - w0) * (step_now / step_max)
        
        loss = (
            self.config.w_gray_loss_whole    * gray_loss_whole +
            self.config.w_ssim_loss_whole    * ssim_loss_whole +
            w_dice_loss                      * dice_loss
        )
        
        return {
            "loss": loss,
            "gray_loss_whole": gray_loss_whole,
            "ssim_loss_whole": ssim_loss_whole,
            "dice_loss": dice_loss
        }, {
            "loss": True,
            "gray_loss_whole": True,
            "ssim_loss_whole": True,
            "dice_loss": True
        }
    
    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        return self._get_basic_metrics(
            pl_module=pl_module,
            gaussian_model=gaussian_model,
            batch=batch,
            outputs=outputs,
        )
    
    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs: RenderRes) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, prog_bar = self._get_basic_metrics(pl_module, gaussian_model, batch, outputs)

        image_info: Tuple[str, torch.Tensor, torch.Tensor]
        _, image_info, _ = batch   # load depth_map as extra_data in internal/dataparsers/rotated_xray_dataparser.py
        _, gt_image, masked_pixels = image_info
        gt_image = gt_image[0:1]

        metrics["psnr_whole"] = self.psnr(outputs.gray_image_whole, gt_image)
        prog_bar["psnr_whole"] = True
        
        gray2rgb_whole = outputs.gray_image_whole.clamp(0., 1.)[None].repeat(1, 3, 1, 1)    # [1, 3, H, W]
        gray2rgb_gt_whole = gt_image.clamp(0., 1.)[None].repeat(1, 3, 1, 1)
        metrics["lpips_whole"] = self.no_state_dict_models["lpips"](gray2rgb_whole, gray2rgb_gt_whole)
        prog_bar["lpips_whole"] = True
        
        if outputs.alpha is not None:
            masked_pixels = masked_pixels[0:1].to(torch.uint8)
            soft_coronary_mask = torch.sigmoid((outputs.alpha - 0.01) * 5)
            dice_loss = self.dice_loss_fn(soft_coronary_mask[None], masked_pixels[None])
            metrics["dice_loss"] = dice_loss
            prog_bar["dice_loss"] = True

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