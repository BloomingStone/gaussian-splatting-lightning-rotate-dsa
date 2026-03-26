from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any

import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from monai.losses.dice import DiceCELoss

from ..utils.ssim import ssim
from .metric import Metric, MetricImpl
from ..renderers.deformabel_xray_renderer import RenderRes
from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel
from ..gaussian_splatting import GaussianSplatting

@dataclass
class RotateXrayMetrics(Metric):
    w_gray_loss_whole: float = 1.0
    w_ssim_loss_whole: float = 1.0
    
    w_phase_aware_loss: float = 0.0

    rgb_diff_loss: Literal["l1", "l2"] = "l1"

    lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex"
    """
    the vanilla 3DGS uses 'vgg', but 'alex' is faster
    """

    fused_ssim: bool = False

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return RotateXrayMetricsImpl(self)

def corr_loss(A: torch.Tensor, B: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    A = A - A.mean()
    B = B - B.mean()
    cov = (A * B).mean()
    stdA = A.std() + eps
    stdB = B.std() + eps
    return 1.0 - cov / (stdA * stdB)


def entropy(p: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    return - (p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)).mean()


def motion(d_xyz: torch.Tensor, d_scale: torch.Tensor, d_rotation: torch.Tensor) -> torch.Tensor:
    d_rotation_norm = torch.nn.functional.normalize(d_rotation, dim=-1)
    d_rotation_norm = d_rotation_norm.clamp(-1 + 1e-6, 1 - 1e-6)
    d_angle = 2 * torch.acos(d_rotation_norm[:, 0]).unsqueeze(-1)
    motion = torch.cat((d_xyz, d_scale, d_angle), dim=-1)
    return motion

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
        pl_module: GaussianSplatting, 
        gaussian_model: XrayCoronaryGaussianModel, 
        batch, 
        outputs: RenderRes
    ):
        image_info: Tuple[str, torch.Tensor, torch.Tensor]
        _, image_info, _ = batch   # load depth_map as extra_data in internal/dataparsers/rotated_xray_dataparser.py
        _, gt_image, _ = image_info
        gt_image = gt_image[0:1]
        
        gray_loss_whole = self.rgb_diff_loss_fn(outputs.gray_image, gt_image)
        
        ssim_metric_whole = self.ssim(outputs.gray_image, gt_image)
        ssim_loss_whole = 1.0 - ssim_metric_whole

        # phase 0 or 1 -> 1, phase 0.5 -> 0
        # phase 0: no motion, phase 0.5: most motion, phase 1: no motion
        d_xyz = outputs.d_means3D.norm(dim=-1).mean()   # (N, 3) -> (N,) -> scalar
        l_phase_aware_loss = (2*(outputs.time - 0.5)) ** 4 * d_xyz  
        l_phase_aware_loss = l_phase_aware_loss.mean()

        loss = (
            # image loss
            self.config.w_gray_loss_whole * gray_loss_whole +
            self.config.w_ssim_loss_whole * ssim_loss_whole +
            self.config.w_phase_aware_loss * l_phase_aware_loss
        )
        
        assert not torch.isnan(loss), "Loss is NaN!"
        
        metrics = {
            "loss": loss,
            "gray_loss_whole": gray_loss_whole,
            "ssim_loss_whole": ssim_loss_whole,
            "phase_aware_loss": l_phase_aware_loss
        }
        
        prog_bar = {
            "loss": True,
            
            "gray_loss_whole": True,
            "ssim_loss_whole": True,
            "phase_aware_loss": True,
        }
        
        if outputs.time < 0.1:
            metrics["small_phase_d_xyz_mean"] = d_xyz
            prog_bar["small_phase_d_xyz_mean"] = False
        
        return metrics, prog_bar
    
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

        metrics["psnr_whole"] = self.psnr(outputs.gray_image, gt_image)
        prog_bar["psnr_whole"] = True
        
        gray2rgb_whole = outputs.gray_image.clamp(0., 1.)[None].repeat(1, 3, 1, 1)    # [1, 3, H, W]
        gray2rgb_gt_whole = gt_image.clamp(0., 1.)[None].repeat(1, 3, 1, 1)
        metrics["lpips_whole"] = self.no_state_dict_models["lpips"](gray2rgb_whole, gray2rgb_gt_whole)
        prog_bar["lpips_whole"] = True

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