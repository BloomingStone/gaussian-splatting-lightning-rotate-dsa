from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any

import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from monai.losses.dice import DiceCELoss

from internal.utils.ssim import ssim
from .metric import Metric, MetricImpl
from ..configs.instantiate_config import InstantiatableConfig
from ..renderers.deformabel_xray_renderer import RenderRes
from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel
from internal.gaussian_splatting import GaussianSplatting

@dataclass
class RotateXrayMetrics(Metric):
    w_gray_loss_whole: float = 1.0
    w_ssim_loss_whole: float = 1.0
    
    w_motion_var_loss: float = 0.
    w_motion_mean_loss: float = 0.005
    
    structure_loss_start_step: int = 2000
    w_coronary_props: float = 0.005
    w_coronary_props_entropy: float = 0.005
    props_entropy_k = 1.0
    w_motion_corr_loss: float = 1.
    w_density_corr_loss: float = 1.
    
    iter_aware_window_length = 1000

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
        
        p = outputs.coronary_props.clamp(0., 1.)
        p_mean = p.mean()
        p_entropy = entropy(p ** self.config.props_entropy_k, eps=1e-6)
        
        xyz_var_norm = torch.norm(outputs.d_motion_var[:, :3], dim=-1)
        xyz_mean_norm = torch.norm(outputs.d_motion_mean[:, :3], dim=-1)
        
        motion_var_mean = xyz_var_norm[p.squeeze() > 0.5].mean()
        L_motion = torch.exp( - xyz_mean_norm[p.squeeze() > 0.5]).mean().clamp(0., 1.)   # we want to maximize the motion

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
            
        if torch.isnan(loss_motion_corr) or loss_motion_corr > 0.3:
            loss_motion_corr = torch.tensor(0.0, device=loss_motion_corr.device)
        
        if torch.isnan(p_entropy):
            p_entropy = torch.tensor(0.0, device=p_entropy.device)
            
        if torch.isnan(L_motion):
            L_motion = torch.tensor(0.0, device=L_motion.device)
            
        if torch.isnan(motion_var_mean):
            motion_var_mean = torch.tensor(0.0, device=motion_var_mean.device)
        
        loss = (
            # image loss
            self.config.w_gray_loss_whole * gray_loss_whole +
            self.config.w_ssim_loss_whole * ssim_loss_whole +
            
            # motion  loss
            self.config.w_motion_var_loss * motion_var_mean +
            self.config.w_motion_mean_loss * L_motion +
            
            # structural loss
            w_p_mean                * p_mean +
            w_p_entropy             * p_entropy + 
            self.config.w_motion_corr_loss * loss_motion_corr + 
            self.config.w_density_corr_loss * loss_density_corr
        )
        
        window = min(pl_module.global_step / self.config.iter_aware_window_length, 1.0)
        if outputs.time.detach() > window:
            loss = loss * 0.01
        
        assert not torch.isnan(loss), "Loss is NaN!"
        
        return {
            "loss": loss,
            
            "gray_loss_whole": gray_loss_whole,
            "ssim_loss_whole": ssim_loss_whole,
            
            "motion_var_mean": motion_var_mean,
            "motion_mean_mean": L_motion,
            
            "coronary_props_mean": p_mean,
            "coronary_props_entropy": p_entropy,
            "loss_corr": loss_motion_corr
        }, {
            "loss": True,
            
            "gray_loss_whole": True,
            "ssim_loss_whole": True,
            
            "motion_var_mean": False,
            "motion_mean_mean": True,
            
            "coronary_props_mean": True,
            "coronary_props_entropy": True,
            "loss_corr": True,
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