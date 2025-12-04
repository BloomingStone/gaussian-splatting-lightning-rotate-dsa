from typing import Any, Optional, Tuple
from dataclasses import dataclass
from functools import partial
import math

import torch
from torch import Tensor
from lightning import LightningModule
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from jaxtyping import Float32

from .renderer import Renderer, RendererOutputInfo, RendererOutputTypes
from ..cameras import Camera
from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel, XrayGassianState
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
    chunk: int = -1  # avoid CUDA oom


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
    gray_image_backgound: Float32[Tensor, "1 h w"]
    viewspace_points: dict[XrayGassianState, Float32[Tensor, "n 2"]]
    visibility_filter: dict[XrayGassianState, Float32[Tensor, "n"]]
    radii: dict[XrayGassianState, Float32[Tensor, "n"]]
    
    def exp_neg_img(self) -> "RenderRes":
        return RenderRes(
            gray_image_coronary= 1 - 0.001 * self.gray_image_coronary,
            gray_image_whole= 1 - 0.001 * self.gray_image_whole,
            gray_image_backgound= 1 - 0.001 * self.gray_image_backgound,
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
            exp_neg_img: bool = False    # DSA image usually has reverse gray scale
    ) -> None:
        super().__init__()
        self.deform_network_config = deform_network
        self.xyz_encoding_config = xyz_encoding
        self.time_encoding_config = time_encoding
        self.optimization_config = optimization
        self.exp_neg_img = exp_neg_img
    
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
        return res.exp_neg_img() if self.exp_neg_img else res

    def training_forward(
            self,
            step: int,
            module: LightningModule,
            viewpoint_camera: Camera,
            pc: XrayCoronaryGaussianModel,
            **kwargs,
    ) -> RenderRes:
        pc.state = XrayGassianState.CORONARY
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
        return res.exp_neg_img() if self.exp_neg_img else res
    
    def static_forward(
            self,
            step: int,
            module: LightningModule,
            viewpoint_camera: Camera,
            pc: XrayCoronaryGaussianModel,
            **kwargs,
    ):
        rasterizer = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=math.tan(viewpoint_camera.fov_x * 0.5),
            tanfovy=math.tan(viewpoint_camera.fov_y * 0.5),
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            mode=1, #cone beam
            debug=False,
        ))
        
        def render_by_state(state: XrayGassianState) -> Tuple[Float32[Tensor, "1 h w"], Float32[Tensor, "n"], Float32[Tensor, "n 3"]]:
            pc.state = state
            means_2d = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=pc.get_xyz.device) + 0
            rendered_image, radii = rasterizer(
                means3D     =   pc.get_xyz,
                means2D     =   means_2d,
                opacities   =   pc.get_opacity,
                scales      =   pc.get_scaling,
                rotations   =   pc.get_rotation,
            )
            pc.state = XrayGassianState.WHOLE
            return rendered_image, radii, means_2d
        
        rendered_image_coronary, radii_coronary, means_2d_coronary = render_by_state(XrayGassianState.CORONARY)
        rendered_image_background, radii_background, means_2d_background = render_by_state(XrayGassianState.BACKGROUND)
        
        
        assert pc.state == XrayGassianState.WHOLE
        res = RenderRes(
            gray_image_coronary     =   rendered_image_coronary,
            gray_image_whole        =   rendered_image_coronary + rendered_image_background,
            gray_image_backgound    =   rendered_image_background,
            viewspace_points        =   {
                                            XrayGassianState.CORONARY: means_2d_coronary,
                                            XrayGassianState.BACKGROUND: means_2d_background
                                        },
            visibility_filter       =   {
                                            XrayGassianState.CORONARY: radii_coronary > 0, 
                                            XrayGassianState.BACKGROUND: radii_background > 0
                                        },
            radii                   =   {
                                            XrayGassianState.CORONARY: radii_coronary, 
                                            XrayGassianState.BACKGROUND: radii_background
                                        }
        )
        return res.exp_neg_img() if self.exp_neg_img else res
        
    
    def _render(
            self,
            d_xyz,
            d_rotation,
            d_scaling,
            viewpoint_camera: Camera,
            pc: XrayCoronaryGaussianModel
    ) -> RenderRes:
        pc.state = XrayGassianState.CORONARY
        scales = pc.get_scaling + d_scaling
        
        if self.deform_network_config.rotate_xyz is True:
            if torch.is_tensor(d_xyz) is True:
                normalized_qvec = torch.nn.functional.normalize(d_rotation)
                # rotate gaussians
                rotations = GaussianTransformUtils.quat_multiply(pc.get_rotation, normalized_qvec)
                # transform xyz
                means3D = pc.get_xyz + d_xyz
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
        
        rotations = torch.nn.functional.normalize(rotations, dim=-1)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=math.tan(viewpoint_camera.fov_x * 0.5),
            tanfovy=math.tan(viewpoint_camera.fov_y * 0.5),
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            mode=1, #cone beam
            debug=False,
        )
        
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        means_2d_coronary = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=means3D.device) + 0
        rendered_image_coronary, radii_coronary = rasterizer(
            means3D=means3D,
            means2D=means_2d_coronary,
            opacities=pc.get_opacity,
            scales=scales,
            rotations=rotations,
        )
        
        pc.state = XrayGassianState.BACKGROUND
        means_2d_background = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=pc.get_xyz.device) + 0
        rendered_image_background, radii_background = rasterizer(
            means3D=pc.get_xyz,
            means2D=means_2d_background,
            opacities=pc.get_opacity,
            scales=pc.get_scaling,
            rotations=pc.get_rotation,
        )

        pc.state = XrayGassianState.WHOLE
        
        viewspace_points = {
            XrayGassianState.CORONARY: means_2d_coronary,   # make slice ([:n_cor]) here will cause grad loss 
            XrayGassianState.BACKGROUND: means_2d_background
        }
        radii = {
            XrayGassianState.CORONARY: radii_coronary, 
            XrayGassianState.BACKGROUND: radii_background
        }
        
        visibility_filter = {
            XrayGassianState.CORONARY: radii_coronary > 0, 
            XrayGassianState.BACKGROUND: radii_background > 0
        }
        
        rendered_image_whole = rendered_image_coronary + rendered_image_background
        
        
        assert pc.state == XrayGassianState.WHOLE
        return RenderRes(
            gray_image_coronary=rendered_image_coronary,
            gray_image_whole=rendered_image_whole,
            gray_image_backgound=rendered_image_background,
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
        }