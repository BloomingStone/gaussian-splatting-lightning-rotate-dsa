from typing import Any, Optional, Tuple
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor
from lightning import LightningModule
from gsplat import rasterization
from jaxtyping import Float32

from .renderer import Renderer, RendererOutputInfo, RendererOutputTypes
from ..schedulers import ExponentialDecayScheduler
from ..cameras import Camera
from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel
from ..cameras import Camera
from ..models.coronary_deform_model import DeformModel, DeformModelConfig
from ..models.coronary_segmentation_model import SegModel, SegModelConfig
from ..utils.network_factory import NetworkFactory
from ..utils.general_utils import get_linear_noise_func
from ..utils.gaussian_utils import GaussianTransformUtils


@dataclass
class DeformNetworkConfig:
    """
    Args:
        tcnn: whether use tiny-cuda-nn as network implementation
    """

    tcnn: bool = False
    n_layers: int = 8
    n_neurons: int = 256


@dataclass
class SegNetworkConfig:
    """
    Args:
        tcnn: whether use tiny-cuda-nn as network implementation
    """

    tcnn: bool = False
    n_layers: int = 8
    n_neurons: int = 256


@dataclass
class XYZEncodingConfig:
    n_frequencies: int = 10


@dataclass
class TimeEncodingConfig:
    n_frequencies: int = 6
    n_layers: int = 2
    n_neurons: int = 256
    n_output_dim: int = 30


@dataclass
class DeformableRendererOptimizationConfig:
    lr: float = 0.0008
    max_steps: int = 40_000
    lr_final_factor: float = 0.002
    eps: float = 1e-15
    warm_up: int = 20_000
    enable_ast: bool = True


@dataclass
class RenderRes:
    gray_image: Float32[Tensor, "1 h w"]
    depth: Float32[Tensor, "1 h w"]
    coronary_probs: Float32[Tensor, "1 h w"]
    viewspace_points: Float32[Tensor, "n 2"]
    visibility_filter: Float32[Tensor, "n"]
    radii: Float32[Tensor, "n"]
    has_coronary_probs: bool
    
    d_motion_mean_total: Tensor
    d_motion_var_total: Tensor
    
    def reverse_gray_scale(self):
        self.gray_image.mul_(-1).add_(1)    # 1 - gray_image
    
    def __getitem__(self, item):
        return getattr(self, item)
    
    def __contains__(self, item):
        return hasattr(self, item)
    
    def __len__(self):
        return len(self.__dict__)
    
    def __iter__(self):
        return iter(self.__dict__)
    
    def items(self):
        output_keys = [
            "viewspace_points",
            "visibility_filter",
            "radii",
        ]
        for key in output_keys:
            yield key, getattr(self, key)
    
    def get(self, key: str, defualt_value: Tensor|None) -> Tensor|None:
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return defualt_value

class CoronaryDeformableXrayRenderer(Renderer):
    def __init__(
            self,
            deform_network: DeformNetworkConfig,
            segmentation_network: SegNetworkConfig,
            xyz_encoding: XYZEncodingConfig,
            time_encoding: TimeEncodingConfig,
            optimization: DeformableRendererOptimizationConfig,
            reverse_gray_scale: bool = False    # DSA image usually has reverse gray scale
    ) -> None:
        super().__init__()
        self.deform_network_config = deform_network
        self.segmentation_config = segmentation_network
        self.xyz_encoding_config = xyz_encoding
        self.time_encoding_config = time_encoding
        self.optimization_config = optimization
        self.reverse_gray_scale = reverse_gray_scale
    
    def forward(
            self,
            viewpoint_camera: Camera,
            pc: XrayCoronaryGaussianModel,
            **kwargs,
    ) -> RenderRes:
        t = viewpoint_camera.time.unsqueeze(0)
        N = pc.get_xyz.shape[0]
        time_input = t.unsqueeze(0).expand(N, -1)
        d_xyz, d_scaling, d_rotation = self.deform_model(pc.get_xyz.detach(), time_input)
        
        res = self._render(
            d_xyz,
            d_scaling,
            d_rotation,
            time_input,
            viewpoint_camera=viewpoint_camera,
            pc=pc
        )
        if self.reverse_gray_scale is True:
            res.reverse_gray_scale()
        return res

    def training_forward(
            self,
            step: int,
            module: LightningModule,
            viewpoint_camera: Camera,
            pc: XrayCoronaryGaussianModel,
            **kwargs,
    ) -> RenderRes:
        t = viewpoint_camera.time.unsqueeze(0)
        N = pc.get_xyz.shape[0]
        time_input = t.unsqueeze(0).expand(N, -1)
        ast_noise = 0
        if self.optimization_config.enable_ast is True:
            time_interval = 1 / ((step % self.train_set_length) + 1)
            ast_noise = torch.randn(1, 1, device=pc.get_xyz.device).expand(N, -1) * time_interval * self.smooth_term(step)
        d_xyz, d_scaling, d_rotation = self.deform_model(pc.get_xyz.detach(), time_input + ast_noise)
        torch.cuda.empty_cache()  # avoid CUDA OOM
        
        res = self._render(
            d_xyz,
            d_scaling,
            d_rotation,
            time_input,
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            step=step
        )

        if self.reverse_gray_scale is True:
            res.reverse_gray_scale()
        return res
    
    def _render(
            self,
            d_xyz,
            d_scaling,
            d_rotation,
            time_input,
            viewpoint_camera: Camera,
            pc: XrayCoronaryGaussianModel,
            step: int|None = None
    ) -> RenderRes:
        means3D: Tensor = pc.get_xyz + d_xyz
        normalized_qvec = torch.nn.functional.normalize(d_rotation)
        rotations: Tensor = GaussianTransformUtils.quat_multiply(pc.get_rotation, normalized_qvec)
        d_motion_mean_total, d_motion_var_total = pc.update_motions(d_xyz, d_scaling, d_rotation)

        if torch.isnan(d_motion_mean_total):
            pass
        
        scales: Tensor = pc.get_scaling + d_scaling
        opacity = pc.get_opacity
        features = pc.get_features
        
        viewmats = viewpoint_camera.world_to_camera.transpose(-1, -2)[None]     # C=1, 4, 4
        Ks = viewpoint_camera.get_K()[None, :3, :3]     # C=1, 3, 3
        width = int(viewpoint_camera.width.item())
        height = int(viewpoint_camera.height.item())
        
        render_colors, _, meta_whole = rasterization(
            means       =   means3D,
            quats       =   rotations,
            scales      =   scales,
            opacities   =   opacity.squeeze(),
            colors      =   features,
            render_mode =   "RGB",
            viewmats=viewmats, # C=1, 4, 4
            Ks=Ks,  # C=1, 3, 3
            width=width,
            height=height,
            rasterize_mode="antialiased",    # Mip-Splatting: Alias-free 3D Gaussian Splatting
            absgrad = True,      # AbsGS: Recovering Fine Details for 3D Gaussian Splatting,
            packed=False    # packed=True meets bug with backgrounds. ref: https://github.com/nerfstudio-project/gsplat/issues/826
        )
        meta_whole["means2d"].requires_grad_(True)
        meta_whole["means2d"].retain_grad()
        
        gray_image=render_colors[..., 0]     # 1, H, W

        viewspace_points = meta_whole["means2d"]
        radii = meta_whole["radii"][0].amax(dim=-1)
        visibility_filter = radii > 0
        
        if step is None or step > self.optimization_config.warm_up:
            d_xyz, _, _ = self.deform_model(pc.get_xyz.detach(), torch.zeros_like(time_input))     # segment_as time/phase=0
            means3D_new: Tensor = pc.get_xyz + d_xyz
            
            seg_probs = self.seg_model(
                means3D_new.detach(),
                pc.get_gray().detach(),
                pc.get_motion_mean().detach(),
                pc.get_motion_var().detach()
            ).to(means3D.dtype)
            
            render_seg_colors, _, _ = rasterization(
                means       =   means3D.detach(),
                quats       =   rotations.detach(),
                scales      =   scales.detach(),
                opacities   =   seg_probs.squeeze(),    #(N,)
                colors      =   seg_probs,              #(N,1)
                render_mode =   "RGB+ED",
                viewmats=viewmats, # C=1, 4, 4
                Ks=Ks,  # C=1, 3, 3
                width=width,
                height=height,
                rasterize_mode="antialiased",    # Mip-Splatting: Alias-free 3D Gaussian Splatting
                absgrad = True,      # AbsGS: Recovering Fine Details for 3D Gaussian Splatting,
                packed=False    # packed=True meets bug with backgrounds. ref: https://github.com/nerfstudio-project/gsplat/issues/826
            )
            
            seg_probs_2d = render_seg_colors[..., 0]
            depth = render_seg_colors[..., 1]
            has_coronary_probs = True
        else:
            seg_probs_2d = torch.zeros_like(gray_image).to(gray_image)
            depth = torch.zeros_like(gray_image).to(gray_image)
            has_coronary_probs = False

        return RenderRes(
            gray_image=gray_image,
            depth=depth,
            coronary_probs=seg_probs_2d,
            viewspace_points=viewspace_points,
            visibility_filter=visibility_filter,
            radii=radii,
            has_coronary_probs=has_coronary_probs,
            d_motion_mean_total=d_motion_mean_total,
            d_motion_var_total=d_motion_var_total
        )
    
    def setup(self, stage: str, lightning_module, *args: Any, **kwargs: Any) -> Any:
        if stage == "fit":
            self.train_set_length = len(lightning_module.trainer.datamodule.dataparser_outputs.train_set)
        network_factory = NetworkFactory(tcnn=self.deform_network_config.tcnn)

        self.deform_model = DeformModel(
            network_factory, 
            DeformModelConfig(
                D=self.deform_network_config.n_layers,
                W=self.deform_network_config.n_neurons,
                multires=self.xyz_encoding_config.n_frequencies,
                t_D=self.time_encoding_config.n_layers
            )
        )
        
        self.seg_model = SegModel(
            network_factory,
            SegModelConfig(
                D=self.segmentation_config.n_layers,
                W=self.segmentation_config.n_neurons,
                multires=self.xyz_encoding_config.n_frequencies,
            )
        )
        
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    
    def training_setup(self, module) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.deform_model.parameters()),
                    "name": "deform",
                },
                {
                    "params": list(self.seg_model.parameters()),
                    "name": "seg",
                }
                ],
            lr=self.optimization_config.lr,
            eps=self.optimization_config.eps,
        )
        
        k = self.optimization_config.lr_final_factor
        iter_max = self.optimization_config.max_steps
        iter_warmup = self.optimization_config.warm_up
        lr_lmabda_deform = lambda iter: k ** min(iter / iter_max, 1)
        lr_lmabda_seg = lambda iter: k ** min(max(iter-iter_warmup, 0) / iter_max, 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=[lr_lmabda_deform, lr_lmabda_seg],
        )

        return optimizer, scheduler

    def get_available_outputs(self) -> dict:
        cmap = {"colormap": "gray"}
        return {
            "gray_image": RendererOutputInfo("gray_image", RendererOutputTypes.GRAY, other_kwargs=cmap),
            "coronary_probs": RendererOutputInfo("coronary_probs", RendererOutputTypes.GRAY, other_kwargs=cmap),
            "depth": RendererOutputInfo("depth", RendererOutputTypes.GRAY, other_kwargs=cmap),
        }