from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any

import torch
import torch.nn.functional as F
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

    # Weight for the original train loss branch (gray + ssim + phase aware).
    w_base_train_loss: float = 1.0
    enable_frangi_loss: bool = False
    w_frangi_loss: float = 0.0
    frangi_sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0)
    frangi_beta: float = 0.5
    frangi_gamma: float = 15.0
    frangi_black_ridges: bool = False
    frangi_eps: float = 1e-6

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
        self._build_frangi_kernels(dtype=torch.float32)

    def _build_frangi_kernels(self, dtype: torch.dtype = torch.float32):
        self.frangi_kernels = []
        for sigma in self.config.frangi_sigmas:
            if sigma <= 0:
                continue
            radius = max(1, int(round(3.0 * sigma)))
            ksize = 2 * radius + 1
            coords = torch.arange(-radius, radius + 1, dtype=dtype)
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")

            sigma2 = sigma * sigma
            gaussian = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma2)) / (2.0 * torch.pi * sigma2)

            dxx = ((xx * xx - sigma2) / (sigma2 * sigma2)) * gaussian
            dyy = ((yy * yy - sigma2) / (sigma2 * sigma2)) * gaussian
            dxy = ((xx * yy) / (sigma2 * sigma2 * sigma2)) * gaussian

            # Scale normalization for Hessian at current sigma.
            dxx = (sigma2 * dxx).unsqueeze(0).unsqueeze(0)
            dyy = (sigma2 * dyy).unsqueeze(0).unsqueeze(0)
            dxy = (sigma2 * dxy).unsqueeze(0).unsqueeze(0)

            self.frangi_kernels.append((dxx, dyy, dxy))

    @staticmethod
    def _to_nchw(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 2:
            return image.unsqueeze(0).unsqueeze(0)
        if image.dim() == 3:
            return image.unsqueeze(0)
        if image.dim() == 4:
            return image
        raise ValueError(f"Unsupported image dim for Frangi: {image.dim()}")

    @staticmethod
    def _from_nchw(filtered: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if ref.dim() == 2:
            return filtered[0, 0]
        if ref.dim() == 3:
            return filtered[0]
        return filtered

    @staticmethod
    def _conv2d_depthwise(image_nchw: torch.Tensor, kernel_1x1khkw: torch.Tensor) -> torch.Tensor:
        channels = image_nchw.shape[1]
        k = kernel_1x1khkw.shape[-1]
        weight = kernel_1x1khkw.to(device=image_nchw.device, dtype=image_nchw.dtype).expand(channels, 1, k, k)
        return F.conv2d(image_nchw, weight, padding=k // 2, groups=channels)

    def _frangi_filter(self, image: torch.Tensor) -> torch.Tensor:
        image_nchw = self._to_nchw(image)
        if len(self.frangi_kernels) == 0:
            return image

        eps = self.config.frangi_eps
        beta2 = max(self.config.frangi_beta * self.config.frangi_beta, eps)
        gamma2 = max(self.config.frangi_gamma * self.config.frangi_gamma, eps)

        vesselness_max = torch.zeros_like(image_nchw)
        for dxx_kernel, dyy_kernel, dxy_kernel in self.frangi_kernels:
            dxx = self._conv2d_depthwise(image_nchw, dxx_kernel)
            dyy = self._conv2d_depthwise(image_nchw, dyy_kernel)
            dxy = self._conv2d_depthwise(image_nchw, dxy_kernel)

            trace = dxx + dyy
            det_term = torch.sqrt((dxx - dyy) * (dxx - dyy) + 4.0 * dxy * dxy + eps)
            lambda1 = 0.5 * (trace + det_term)
            lambda2 = 0.5 * (trace - det_term)

            swap_mask = lambda1.abs() > lambda2.abs()
            lambda1_sorted = torch.where(swap_mask, lambda2, lambda1)
            lambda2_sorted = torch.where(swap_mask, lambda1, lambda2)

            rb = lambda1_sorted.abs() / (lambda2_sorted.abs() + eps)
            s2 = lambda1_sorted * lambda1_sorted + lambda2_sorted * lambda2_sorted
            vesselness = torch.exp(-(rb * rb) / (2.0 * beta2)) * (1.0 - torch.exp(-s2 / (2.0 * gamma2)))

            if self.config.frangi_black_ridges:
                vesselness = torch.where(lambda2_sorted < 0, torch.zeros_like(vesselness), vesselness)
            else:
                vesselness = torch.where(lambda2_sorted > 0, torch.zeros_like(vesselness), vesselness)

            vesselness_max = torch.maximum(vesselness_max, vesselness)

        return self._from_nchw(vesselness_max, image)
    
    def _get_basic_metrics(
        self, 
        pl_module: GaussianSplatting, 
        gaussian_model: XrayCoronaryGaussianModel, 
        batch, 
        outputs: RenderRes,
        include_frangi_in_loss: bool = False,
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

        base_loss = (
            # image loss
            self.config.w_gray_loss_whole * gray_loss_whole +
            self.config.w_ssim_loss_whole * ssim_loss_whole +
            self.config.w_phase_aware_loss * l_phase_aware_loss
        )

        loss = self.config.w_base_train_loss * base_loss
        frangi_loss = torch.zeros_like(loss)
        frangi_loss_weighted = torch.zeros_like(loss)
        if include_frangi_in_loss and self.config.enable_frangi_loss and self.config.w_frangi_loss > 0:
            pred_frangi = self._frangi_filter(outputs.gray_image)
            gt_frangi = self._frangi_filter(gt_image)
            frangi_loss = self.rgb_diff_loss_fn(pred_frangi, gt_frangi)
            frangi_loss_weighted = self.config.w_frangi_loss * frangi_loss
            loss = loss + frangi_loss_weighted
        
        assert not torch.isnan(loss), "Loss is NaN!"
        
        metrics = {
            "loss": loss,
            "base_train_loss": base_loss,
            "gray_loss_whole": gray_loss_whole,
            "ssim_loss_whole": ssim_loss_whole,
            "phase_aware_loss": l_phase_aware_loss,
        }

        if include_frangi_in_loss:
            metrics["frangi_loss"] = frangi_loss
            metrics["frangi_loss_weighted"] = frangi_loss_weighted
        
        prog_bar = {
            "loss": True,
            "base_train_loss": True,
            "gray_loss_whole": True,
            "ssim_loss_whole": True,
            "phase_aware_loss": True,
        }

        if include_frangi_in_loss:
            prog_bar["frangi_loss"] = True
            prog_bar["frangi_loss_weighted"] = True
        
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
            include_frangi_in_loss=True,
        )
    
    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs: RenderRes) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, prog_bar = self._get_basic_metrics(
            pl_module,
            gaussian_model,
            batch,
            outputs,
            include_frangi_in_loss=False,
        )

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