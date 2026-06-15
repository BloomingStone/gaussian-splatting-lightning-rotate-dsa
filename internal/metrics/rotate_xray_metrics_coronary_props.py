from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .ssim import ssim
from .metric import Metric, MetricImpl
from ..renderers.deformabel_xray_renderer_coronary_props import ImgT, MetaT, XrayRendererOuputs
from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel
from ..gaussian_splatting import GaussianSplatting
from ..dataset import BatchT

@dataclass
class RotateXrayMetrics(Metric):
    w_gray_loss: float = 1.0
    w_ssim_loss: float = 1.0
    
    w_motion_var_loss: float = 0.
    w_motion_mean_loss: float = 0.
    
    structure_loss_start_step: int = 1000
    
    w_coronary_props: float = 0.
    w_coronary_props_entropy: float = 0.
    
    props_entropy_k = 1.0
    w_motion_corr_loss: float = 0.
    w_density_corr_loss: float = 0.

    rgb_diff_loss: Literal["l1", "l2"] = "l1"

    lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex"
    """
    the vanilla 3DGS uses 'vgg', but 'alex' is faster
    """

    fused_ssim: bool = True

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
        
    
    def _get_basic_metrics(
        self, 
        pl_module: GaussianSplatting, 
        gaussian_model: XrayCoronaryGaussianModel, 
        batch: BatchT, 
        outputs: XrayRendererOuputs
    ):
        gt_image = batch.image_info.gt_image[0:1]
        
        gray_loss = self.rgb_diff_loss_fn(outputs.gray_image, gt_image)
        
        ssim_metric = self.ssim(outputs.gray_image, gt_image)
        ssim_loss = 1.0 - ssim_metric
        
        p = outputs.coronary_props.clamp(0., 1.)
        p_mean = p.mean()
        p_entropy = entropy(p ** self.config.props_entropy_k, eps=1e-6)
        
        xyz_var_norm = torch.norm(outputs.d_motion_var[:, :3], dim=-1)
        xyz_mean_norm = torch.norm(outputs.d_motion_mean[:, :3], dim=-1)
        
        # motion_var_mean = xyz_var_norm[p.squeeze() > 0.5].mean()
        # L_motion = torch.exp( - xyz_mean_norm[p.squeeze() > 0.5]).mean().clamp(0., 1.)   # we want to maximize the motion
        
        motion_var_mean = xyz_var_norm.mean()
        L_motion = xyz_mean_norm.mean()

        if pl_module.global_step < self.config.structure_loss_start_step:
            w_p_entropy = 0.0
            w_p_mean = 0.0
            loss_motion_corr = torch.tensor(0.0, device=gt_image.device)
            loss_density_corr = torch.tensor(0.0, device=gt_image.device)
        else:
            w_p_entropy = self.config.w_coronary_props_entropy
            w_p_mean = self.config.w_coronary_props
            loss_motion_corr = corr_loss(p, xyz_var_norm+xyz_mean_norm)
            s = torch.topk(gaussian_model.get_scales(), k=2, dim=-1)[0][:, -1]
            pho = gaussian_model.get_density().squeeze() / s
            pho = pho.clamp(1e-5, 1e5)
            loss_density_corr = corr_loss(p, pho)
            
        # if torch.isnan(loss_motion_corr) or loss_motion_corr > 0.3:
        #     loss_motion_corr = torch.tensor(0.0, device=loss_motion_corr.device)
        
        # if torch.isnan(p_entropy):
        #     p_entropy = torch.tensor(0.0, device=p_entropy.device)
            
        # if torch.isnan(L_motion):
        #     L_motion = torch.tensor(0.0, device=L_motion.device)
            
        # if torch.isnan(motion_var_mean):
        #     motion_var_mean = torch.tensor(0.0, device=motion_var_mean.device)
        
        loss = (
            # image loss
            self.config.w_gray_loss * gray_loss +
            self.config.w_ssim_loss * ssim_loss +
            
            # motion  loss
            self.config.w_motion_var_loss * motion_var_mean +
            self.config.w_motion_mean_loss * L_motion +
            
            # structural loss
            w_p_mean                * p_mean +
            w_p_entropy             * p_entropy + 
            self.config.w_motion_corr_loss * loss_motion_corr + 
            self.config.w_density_corr_loss * loss_density_corr
        )
        
        # assert not torch.isnan(loss), "Loss is NaN!"
        
        return {
            "loss": loss,
            
            "gray_loss": gray_loss,
            "ssim_loss": ssim_loss,
            
            "motion_var_mean": motion_var_mean,
            "motion_mean_mean": L_motion,
            
            "coronary_props_mean": p_mean,
            "coronary_props_entropy": p_entropy,
            "loss_corr": loss_motion_corr
        }, {
            "loss": True,
            
            "gray_loss": True,
            "ssim_loss": True,
            
            "motion_var_mean": False,
            "motion_mean_mean": True,
            
            "coronary_props_mean": True,
            "coronary_props_entropy": True,
            "loss_corr": True,
        }
    
    def get_train_metrics(
        self, 
        pl_module: GaussianSplatting, 
        gaussian_model: XrayCoronaryGaussianModel, 
        step: int, 
        batch: BatchT, 
        outputs: XrayRendererOuputs
    ) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        return self._get_basic_metrics(
            pl_module=pl_module,
            gaussian_model=gaussian_model,
            batch=batch,
            outputs=outputs,
        )
    
    def get_validate_metrics(self, pl_module: GaussianSplatting, gaussian_model: XrayCoronaryGaussianModel, batch: BatchT, outputs: XrayRendererOuputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, prog_bar = self._get_basic_metrics(pl_module, gaussian_model, batch, outputs)
        gt_image = batch.image_info.gt_image[0:1]

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