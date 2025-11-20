from typing import Any, Optional, Tuple
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor
from lightning import LightningModule
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from jaxtyping import Float32

from .renderer import Renderer, RendererOutputInfo, RendererOutputTypes
from ..cameras import Camera
from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel, XrayGassianState
from .vanilla_renderer import VanillaRenderer
from ..cameras import Camera
from ..models.deform_model import DeformModel
from ..utils.network_factory import NetworkFactory
from ..utils.general_utils import get_linear_noise_func
from ..utils.rigid_utils import from_homogenous, to_homogenous
from ..utils.rotation import qvec2rot
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
    is_6dof: bool = False
    rotate_xyz: bool = False
    chunk: int = -1  # avoid CUDA oom,


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
    warm_up: int = 3_000
    enable_ast: bool = True


@dataclass
class RenderRes:
    gray_image_coronary: Float32[Tensor, "1 h w"]
    gray_image_whole: Float32[Tensor, "1 h w"]
    depth: Float32[Tensor, "1 h w"]
    alpha: Float32[Tensor, "1 h w"]
    viewspace_points: dict[XrayGassianState, Float32[Tensor, "n 2"]]
    visibility_filter: dict[XrayGassianState, Float32[Tensor, "n"]]
    radii: dict[XrayGassianState, Float32[Tensor, "n"]]
    
    def reverse_gray_scale(self, ) -> "RenderRes":
        return RenderRes(
            gray_image_coronary=1.0 - self.gray_image_coronary,
            gray_image_whole=1.0 - self.gray_image_whole,
            depth=self.depth,
            alpha=self.alpha,
            viewspace_points=self.viewspace_points,
            visibility_filter=self.visibility_filter,
            radii=self.radii,
        )
    
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
            xyz_encoding: XYZEncodingConfig,
            time_encoding: TimeEncodingConfig,
            optimization: DeformableRendererOptimizationConfig,
            reverse_gray_scale: bool = False    # DSA image usually has reverse gray scale
    ) -> None:
        super().__init__()
        self.deform_network_config = deform_network
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
        pc.state = XrayGassianState.CORONARY
        N = pc.get_xyz.shape[0]
        time_input = viewpoint_camera.time.unsqueeze(0).expand(N, -1)
        d_xyz, d_rotation, d_scaling = self.deform_model(pc.get_xyz.detach(), time_input)
        
        res = self._render(
            d_xyz,
            d_rotation,
            d_scaling,
            viewpoint_camera=viewpoint_camera,
            pc=pc
        )
        if self.reverse_gray_scale is True:
            res = res.reverse_gray_scale()
        return res

    def training_forward(
            self,
            step: int,
            module: LightningModule,
            viewpoint_camera: Camera,
            pc: XrayCoronaryGaussianModel,
            **kwargs,
    ) -> RenderRes:
        pc.state = XrayGassianState.CORONARY
        d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        if step <= self.optimization_config.warm_up:
            pass    # TODO
        N = pc.get_xyz.shape[0]
        time_input = viewpoint_camera.time.unsqueeze(0).expand(N, -1)
        ast_noise = 0
        if self.optimization_config.enable_ast is True:
            time_interval = 1 / ((step % self.train_set_length) + 1)
            ast_noise = torch.randn(1, 1, device=pc.get_xyz.device).expand(N, -1) * time_interval * self.smooth_term(step)
        d_xyz, d_rotation, d_scaling = self.deform_model(pc.get_xyz.detach(), time_input + ast_noise)
        torch.cuda.empty_cache()  # avoid CUDA OOM
        
        res = self._render(
            d_xyz,
            d_rotation,
            d_scaling,
            viewpoint_camera=viewpoint_camera,
            pc=pc
        )
        
        if self.reverse_gray_scale is True:
            res = res.reverse_gray_scale()
        return res
    
    def _render(
            self,
            d_xyz,
            d_rotation,
            d_scaling,
            viewpoint_camera: Camera,
            pc: XrayCoronaryGaussianModel
    ) -> RenderRes:
        pc.state = XrayGassianState.CORONARY
        if self.deform_network_config.rotate_xyz is True:
            if torch.is_tensor(d_xyz) is True:
                normalized_qvec = torch.nn.functional.normalize(d_rotation)
                # rotate gaussians
                rotations = GaussianTransformUtils.quat_multiply(pc.get_rotation, normalized_qvec)
                # transform xyz
                so3 = qvec2rot(normalized_qvec)
                means3D = torch.matmul(pc.get_xyz.unsqueeze(1), torch.transpose(so3, 1, 2)).squeeze(1) + d_xyz
            else:
                # in warm up
                means3D = pc.get_xyz
                rotations = pc.get_rotation
        else:
            # original processing
            if self.deform_network_config.is_6dof is True:
                if torch.is_tensor(d_xyz) is False:
                    means3D = pc.get_xyz
                else:
                    means3D = from_homogenous(torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
            else:
                means3D = pc.get_xyz + d_xyz
            rotations = pc.get_rotation + d_rotation

        opacity = pc.get_opacity
        scales = pc.get_scaling + d_scaling
        features = pc.get_features
        
        viewmats = viewpoint_camera.world_to_camera.transpose(-1, -2)[None]     # C=1, 4, 4
        Ks = viewpoint_camera.get_K()[None, :3, :3]     # C=1, 3, 3
        width = int(viewpoint_camera.width.item())
        height = int(viewpoint_camera.height.item())
        
        def combine(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            return torch.cat((x1, x2))
        
        pc.state = XrayGassianState.BACKGROUND
        render_colors_whole, render_alphas_whole, meta_whole = rasterization(
            means       =   combine(means3D, pc.get_xyz),
            quats       =   combine(rotations, pc.get_rotation),
            scales      =   combine(scales, pc.get_scaling),
            opacities   =   combine(opacity, pc.get_opacity).squeeze(),
            colors      =   combine(features, pc.get_features),
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
        pc.state = XrayGassianState.WHOLE
        
        gray_image_whole=render_colors_whole[..., 0]     # 1, H, W

        radii = meta_whole["radii"][0].max(dim=-1).values
        visibility_filter = radii > 0
        
        n_cor = pc.gaussians.n_coronary_gs
        assert n_cor is not None
        
        viewspace_points = {
            XrayGassianState.CORONARY: meta_whole["means2d"],   # make slice ([:n_cor]) here will cause grad loss 
            XrayGassianState.BACKGROUND: meta_whole["means2d"]
        }
        radii = {
            XrayGassianState.CORONARY: meta_whole["radii"][0, :n_cor].max(dim=-1).values, 
            XrayGassianState.BACKGROUND: meta_whole["radii"][0, n_cor:].max(dim=-1).values
        }
        
        visibility_filter = {
            XrayGassianState.CORONARY: radii[XrayGassianState.CORONARY] > 0, 
            XrayGassianState.BACKGROUND: radii[XrayGassianState.BACKGROUND] > 0
        }
        
        
        pc.state = XrayGassianState.CORONARY
        # ref: https://docs.gsplat.studio/main/apis/rasterization.html#gsplat.rasterization
        render_colors_coronary, render_alphas_coronary, meta_coronary =rasterization(
            means=      means3D,            # N, 3
            quats=      rotations,          # N, 4
            scales=     scales,             # N, 3
            opacities=  opacity.squeeze(),  # N,
            colors=     features,           # N, D=1
            render_mode="RGB+ED",
            viewmats=viewmats, # C=1, 4, 4
            Ks=Ks,  # C=1, 3, 3
            width=width,
            height=height,
            rasterize_mode="antialiased",    # Mip-Splatting: Alias-free 3D Gaussian Splatting
            absgrad = True,      # AbsGS: Recovering Fine Details for 3D Gaussian Splatting,
            packed=False    # packed=True meets bug with backgrounds. ref: https://github.com/nerfstudio-project/gsplat/issues/826
        )
        gray_image_coronary=render_colors_coronary[..., 0]     # 1, H, W
        depth_coronary=render_colors_coronary[..., 1]
        alpha_coronary=render_alphas_coronary[..., 0]
        pc.state = XrayGassianState.WHOLE
        
        assert pc.state == XrayGassianState.WHOLE
        return RenderRes(
            gray_image_coronary=gray_image_coronary,
            gray_image_whole=gray_image_whole,
            depth=depth_coronary,
            alpha=alpha_coronary,
            viewspace_points=viewspace_points,
            visibility_filter=visibility_filter,
            radii=radii
        )
        
    
    def setup(self, stage: str, lightning_module, *args: Any, **kwargs: Any) -> Any:
        if stage == "fit":
            self.train_set_length = len(lightning_module.trainer.datamodule.dataparser_outputs.train_set)

        network_factory = NetworkFactory(tcnn=self.deform_network_config.tcnn)

        self.deform_model = DeformModel(
            network_factory=network_factory,
            D=self.deform_network_config.n_layers,
            W=self.deform_network_config.n_neurons,
            multires=self.xyz_encoding_config.n_frequencies,
            t_D=self.time_encoding_config.n_layers,
            t_W=self.time_encoding_config.n_neurons,
            t_multires=self.time_encoding_config.n_frequencies,
            is_6dof=self.deform_network_config.is_6dof,
            chunk=self.deform_network_config.chunk,
        )
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    
    def training_setup(self, module) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.Adam(
            [{
                "params": list(self.deform_model.parameters()),
                "name": "deform",
            }],
            lr=self.optimization_config.lr,
            eps=self.optimization_config.eps,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: self.optimization_config.lr_final_factor ** min(iter / self.optimization_config.max_steps, 1),
        )

        return optimizer, scheduler

    def get_available_outputs(self) -> dict:
        cmap = {"colormap": "gray"}
        return {
            "gray_image_coronary": RendererOutputInfo("gray_image_coronary", RendererOutputTypes.GRAY, other_kwargs=cmap),
            "gray_image_whole": RendererOutputInfo("gray_image_whole", RendererOutputTypes.GRAY, other_kwargs=cmap),
            "depth": RendererOutputInfo("depth", RendererOutputTypes.GRAY, other_kwargs=cmap),
            "alpha": RendererOutputInfo("alpha", RendererOutputTypes.GRAY, other_kwargs=cmap),
        }