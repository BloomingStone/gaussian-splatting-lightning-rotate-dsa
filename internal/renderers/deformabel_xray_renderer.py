from typing import Any, Optional, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor
from lightning import LightningModule
from gsplat import rasterization
from jaxtyping import Float32

from internal.models.gaussian import GaussianModel

from .renderer import Renderer, RendererOutputInfo, RendererOutputTypes
from ..schedulers import ExponentialDecayScheduler
from ..cameras import Camera
from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel
from ..cameras import Camera
from ..models.coronary_deform_model import DeformModel, DeformModelConfig
from ..utils.network_factory import NetworkFactory
from ..utils.general_utils import get_linear_noise_func
from ..utils.gaussian_utils import GaussianTransformUtils


@dataclass
class DeformableRendererOptimizationConfig:
    lr: float = 1e-3
    max_steps: int = 40_000
    lr_final_factor: float = 0.002
    eps: float = 1e-15
    warm_up: int = 300
    enable_ast: bool = True


@dataclass
class RenderRes:
    gray_image: Float32[Tensor, "1 h w"]
    gray_coronary: Float32[Tensor, "1 h w"] | None
    viewspace_points: Float32[Tensor, "n 2"]
    visibility_filter: Float32[Tensor, "n"]
    radii: Float32[Tensor, "n"]
    
    d_motion_mean: Tensor | None
    d_motion_var: Tensor | None
    
    means3D: Tensor
    rotation: Tensor
    scales: Tensor
    
    moving_mask: Tensor | None
    
    def reverse_gray_scale(self):
        self.gray_image.mul_(-1).add_(1)    # 1 - gray_image
        if self.gray_coronary is not None:
            self.gray_coronary.mul_(-1).add_(1)  # 1 - gray_coronary
    
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
            optimization: DeformableRendererOptimizationConfig,
            deform_network: DeformModelConfig = DeformModelConfig(),
            reverse_gray_scale: bool = False    # DSA image usually has reverse gray scale
    ) -> None:
        super().__init__()
        self.deform_network_config = deform_network
        self.optimization_config = optimization
        self.reverse_gray_scale = reverse_gray_scale
        
        self.step_record: int = 0
    
    def forward(
        self,
        viewpoint_camera: Camera,
        pc: XrayCoronaryGaussianModel,
        **kwargs,
    ) -> RenderRes:
        means3D = pc.get_xyz.detach()
        gray = pc.get_gray().detach()
        rotation = pc.get_rotation.detach()
        scales: Tensor = pc.get_scaling.detach()
        opacity = pc.get_opacity.detach()
        time = viewpoint_camera.time.unsqueeze(0).expand(means3D.shape[0], -1)
        
        d_xyz, d_scaling, d_rotation, moving_probs = self.deform_model(means3D, time)
        torch.cuda.empty_cache()  # avoid CUDA OOM
        moving_mask = (moving_probs > 0.5).squeeze()
        
        means3D = means3D + d_xyz
        normalized_qvec = torch.nn.functional.normalize(d_rotation)
        rotation = GaussianTransformUtils.quat_multiply(rotation, normalized_qvec)
        scales = scales + d_scaling
        
        d_motion_mean = pc.get_motion_mean().detach()
        d_motion_var = pc.get_motion_var().detach()
        
        gray_image, meta_whole = self._render(viewpoint_camera, means3D, rotation, scales, opacity, gray)

        viewspace_points = meta_whole["means2d"]
        radii = meta_whole["radii"][0].amax(dim=-1)
        visibility_filter = radii > 0

        if torch.any(moving_mask):
            gray_coronary, _ = self._render(
                viewpoint_camera, means3D[moving_mask], rotation[moving_mask], 
                scales[moving_mask], opacity[moving_mask], gray[moving_mask]
            )
        else:
            gray_coronary = torch.zeros_like(gray_image).to(gray_image)
        
        res = RenderRes(
            gray_image, gray_coronary,      # rendered
            viewspace_points, visibility_filter, radii, # grad meta
            d_motion_mean, d_motion_var,    # moving mean & var
            means3D, rotation, scales,      # properties after deform
            moving_mask                     # moving mask from segmentation model
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
        **kwargs
    )-> RenderRes:
        self.step_record = step
        # clone properties
        means3D = pc.get_xyz
        gray = pc.get_gray()
        rotation = pc.get_rotation
        scales: Tensor = pc.get_scaling
        opacity = pc.get_opacity

        # time & ast_noise
        N = means3D.shape[0]
        time = viewpoint_camera.time.unsqueeze(0).expand(N, -1)
        if (self.optimization_config.enable_ast is True\
                and not torch.allclose(time, torch.zeros_like(time))
                and step > self.optimization_config.warm_up
        ):
            time_interval = 1 / ((step % self.train_set_length) + 1)
            ast_noise = torch.randn(1, 1, device=means3D.device).expand(N, -1) * time_interval * self.smooth_term(step)
        
            # update means3D, rotation, scales
            d_xyz, d_scaling, d_rotation, moving_probs = self.deform_model(
                means3D.detach(), 
                time + ast_noise
            )
            torch.cuda.empty_cache()  # avoid CUDA OOM
            
            means3D = means3D + d_xyz
            normalized_qvec = torch.nn.functional.normalize(d_rotation)
            rotation = GaussianTransformUtils.quat_multiply(rotation, normalized_qvec)
            scales = scales + d_scaling
            
            d_motion_mean, d_motion_var = pc.update_motions(d_xyz, d_scaling, d_rotation)
        
        else:
            _, _, _, moving_probs = self.deform_model(
                means3D.detach(), 
                time
            )
            d_motion_mean = pc.get_motion_mean().detach()
            d_motion_var = pc.get_motion_var().detach()

        gray_image, meta_whole = self._render(viewpoint_camera, means3D, rotation, scales, opacity, gray, is_training=True)

        viewspace_points = meta_whole["means2d"]
        radii = meta_whole["radii"][0].amax(dim=-1)
        visibility_filter = radii > 0

        gray_coronary = None
        
        res = RenderRes(
            gray_image, gray_coronary,      # rendered
            viewspace_points, visibility_filter, radii, # grad meta
            d_motion_mean, d_motion_var,    # moving mean & var
            means3D, rotation, scales,      # properties after deform
            moving_mask = (moving_probs > 0.5).squeeze()  # moving mask from segmentation model
        )

        if self.reverse_gray_scale is True:
            res.reverse_gray_scale()
        return res
    
    def _render(
        self, 
        viewpoint_camera,
        means3D,
        rotation,
        scales,
        opacity,
        gray,
        is_training: bool = False
    ):
        viewmats = viewpoint_camera.world_to_camera.transpose(-1, -2)[None]     # C=1, 4, 4
        Ks = viewpoint_camera.get_K()[None, :3, :3]     # C=1, 3, 3
        width = int(viewpoint_camera.width.item())
        height = int(viewpoint_camera.height.item())
        
        render_colors, _, meta = rasterization(
            means       =   means3D,
            quats       =   rotation,
            scales      =   scales,
            opacities   =   opacity.squeeze(),
            colors      =   gray,
            render_mode =   "RGB",
            viewmats=viewmats, # C=1, 4, 4
            Ks=Ks,  # C=1, 3, 3
            width=width,
            height=height,
            rasterize_mode="antialiased",    # Mip-Splatting: Alias-free 3D Gaussian Splatting
            absgrad = True,      # AbsGS: Recovering Fine Details for 3D Gaussian Splatting,
            packed=False    # packed=True meets bug with backgrounds. ref: https://github.com/nerfstudio-project/gsplat/issues/826
        )
        if is_training:
            meta["means2d"].requires_grad_(True)
            meta["means2d"].retain_grad()
        
        gray_image=render_colors[..., 0]     # 1, H, W
        
        return gray_image, meta
    
    
    def setup(self, stage: str, lightning_module, *args: Any, **kwargs: Any) -> Any:
        if stage == "fit":
            self.train_set_length = len(lightning_module.trainer.datamodule.dataparser_outputs.train_set)
        
        self.deform_model = DeformModel(self.deform_network_config)
        
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
        
        k = self.optimization_config.lr_final_factor
        iter_max = self.optimization_config.max_steps
        warm_up = self.optimization_config.warm_up
        lr_lmabda_deform = lambda iter: k ** min((iter-warm_up) / (iter_max-warm_up), 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lmabda_deform,
        )

        return optimizer, scheduler

    def get_available_outputs(self) -> dict:
        cmap = {"colormap": "gray"}
        return {
            "gray_image": RendererOutputInfo("gray_image", RendererOutputTypes.GRAY, other_kwargs=cmap),
            "gray_coronary": RendererOutputInfo("gray_coronary", RendererOutputTypes.GRAY, other_kwargs=cmap)
        }
