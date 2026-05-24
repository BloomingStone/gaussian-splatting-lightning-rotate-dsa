from typing import Any, Optional, Tuple
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from lightning import LightningModule
from jaxtyping import Float32
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from .renderer import Renderer, RendererOutputs
from ..cameras import Camera
from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel
from ..cameras import Camera
from ..deform_models.coronary_props_deform import HashGridDefromModel, HashGridDeformConfig
from ..utils.general_utils import get_linear_noise_func
from ..visualizers import Visualizer, FloatColormapVisualizer, ColorMapName


@dataclass
class DeformableRendererOptimizationConfig:
    lr: float = 1e-3
    max_steps: int = 40_000
    lr_final_factor: float = 0.002
    eps: float = 1e-8
    warm_up: int = 0
    enable_ast: bool = True

@dataclass
class XrayRendererOuputs(RendererOutputs):
    class ImageTypes(StrEnum):
        GRAY = "gray"
        CORONARY = "coronary"
    
    class MetaTypes(StrEnum):
        VIEWSPACE_POINTS = "viewspace_points"
        VISIBILITY_FILTER = "visibility_filter"
        RADII = "radii"
        
        D_MOTION_MEAN = "d_motion_mean"
        D_MOTION_VAR = "d_motion_var"
        D_MEANS3D = "d_means3D"
        D_ROTATION = "d_rotation"
        D_SCALES = "d_scales"
        CORONARY_PROPS = "coronary_props"
        TIME = "time"
    
    images: dict[ImageTypes, tuple[Tensor, Visualizer|None]]
    meta: dict[MetaTypes, Tensor]
    
    is_warm_up: bool

    @property
    def gray_image(self) -> Tensor:
        return self.images[self.ImageTypes.GRAY][0]

    @property
    def coronary_image(self) -> Tensor:
        return self.images[self.ImageTypes.CORONARY][0]

    @property
    def viewspace_points(self) -> Tensor:
        return self.meta[self.MetaTypes.VIEWSPACE_POINTS]

    @property
    def visibility_filter(self) -> Tensor:
        return self.meta[self.MetaTypes.VISIBILITY_FILTER]

    @property
    def radii(self) -> Tensor:
        return self.meta[self.MetaTypes.RADII]

    @property
    def d_motion_mean(self) -> Tensor:
        return self.meta[self.MetaTypes.D_MOTION_MEAN]

    @property
    def d_motion_var(self) -> Tensor:
        return self.meta[self.MetaTypes.D_MOTION_VAR]

    @property
    def d_means3D(self) -> Tensor:
        return self.meta[self.MetaTypes.D_MEANS3D]

    @property
    def d_rotation(self) -> Tensor:
        return self.meta[self.MetaTypes.D_ROTATION]

    @property
    def d_scales(self) -> Tensor:
        return self.meta[self.MetaTypes.D_SCALES]

    @property
    def coronary_props(self) -> Tensor:
        return self.meta[self.MetaTypes.CORONARY_PROPS]

    @property
    def time(self) -> Tensor:
        return self.meta[self.MetaTypes.TIME]

ImgT = XrayRendererOuputs.ImageTypes
MetaT = XrayRendererOuputs.MetaTypes


class CoronaryDeformableXrayRenderer(Renderer):
    deform_model: HashGridDefromModel
    
    def __init__(
            self,
            optimization_config: DeformableRendererOptimizationConfig,
            deform_model_config: HashGridDeformConfig
    ) -> None:
        super().__init__()
        self.deform_model_config = deform_model_config
        self.optimization_config = optimization_config
        self._grad_hook_registered = False

    @staticmethod
    def _ensure_finite(name: str, tensor: Tensor, step: int | None = None) -> None:
        if torch.isfinite(tensor).all():
            return

        finite_mask = torch.isfinite(tensor)
        finite_vals = tensor[finite_mask]
        if finite_vals.numel() > 0:
            min_v = float(finite_vals.min().item())
            max_v = float(finite_vals.max().item())
        else:
            min_v = float("nan")
            max_v = float("nan")

        step_msg = f" at step={step}" if step is not None else ""
        raise RuntimeError(
            f"Non-finite tensor detected for '{name}'{step_msg}. "
            f"shape={tuple(tensor.shape)}, finite_min={min_v}, finite_max={max_v}"
        )

    def forward(
        self,
        viewpoint_camera: Camera,
        pc: XrayCoronaryGaussianModel,
        **kwargs,
    ) -> XrayRendererOuputs:
        means3D = pc.get_means().detach()
        density = pc.get_density().detach()
        rotation = pc.get_rotations().detach()
        scales = pc.get_scales().detach()
        
        time = viewpoint_camera.time.unsqueeze(0).expand(means3D.shape[0], -1)
        

        d_xyz, d_scaling, d_rotation, coronary_props = self.deform_model(means3D.detach(), time)
        self._ensure_finite("d_xyz", d_xyz)
        self._ensure_finite("d_scaling", d_scaling)
        self._ensure_finite("d_rotation", d_rotation)
        means3D, rotation, scales = self.deform_model.deform(
            means3D, rotation, scales, d_xyz, d_rotation, d_scaling
        )
        self._ensure_finite("means3D", means3D)
        self._ensure_finite("rotation", rotation)
        self._ensure_finite("scales", scales)
        
        d_motion_mean = pc.get_motion_mean().detach()
        d_motion_var = pc.get_motion_var().detach()
        
        gray_image, meta_whole = self._render(viewpoint_camera, means3D, rotation, scales, density)

        viewspace_points = meta_whole["viewspace_points"]
        radii = meta_whole["radii"]
        visibility_filter = radii > 0
        
        mask = (density > torch.quantile(density, 0.90)).squeeze()
        if not torch.any(mask):
            mask = torch.ones_like(mask, dtype=torch.bool)
        
        gray_coronary, _ = self._render(
            viewpoint_camera, means3D[mask], rotation[mask], 
            scales[mask], density[mask]
        )
        
        return XrayRendererOuputs(
            images={
                ImgT.GRAY: (gray_image, FloatColormapVisualizer(ColorMapName.GRAY)),
                ImgT.CORONARY: (gray_coronary, FloatColormapVisualizer(ColorMapName.GRAY)),
            },
            meta={
                MetaT.VIEWSPACE_POINTS: viewspace_points,
                MetaT.VISIBILITY_FILTER: visibility_filter,
                MetaT.RADII: radii,
                MetaT.D_MOTION_MEAN: d_motion_mean,
                MetaT.D_MOTION_VAR: d_motion_var,
                MetaT.D_MEANS3D: d_xyz,
                MetaT.D_ROTATION: d_rotation,
                MetaT.D_SCALES: d_scaling,
                MetaT.CORONARY_PROPS: coronary_props.squeeze(),
                MetaT.TIME: viewpoint_camera.time,
            },
            is_warm_up=viewpoint_camera.time.item() <= self.optimization_config.warm_up
        )
        

    def training_forward(
        self, 
        step: int, 
        module: LightningModule, 
        viewpoint_camera: Camera, 
        pc: XrayCoronaryGaussianModel,
        **kwargs
    )-> XrayRendererOuputs:
        # clone properties
        means3D = pc.get_means()
        density = pc.get_density()
        rotation = pc.get_rotations()
        scales = pc.get_scales()

        N = means3D.shape[0]
        time = viewpoint_camera.time.unsqueeze(0).expand(N, -1)
        
        if self.optimization_config.enable_ast:     # add AST noise
            time_interval = 1 / ((step % self.train_set_length) + 1)
            ast_noise = torch.randn(1, 1, device=means3D.device).expand(N, -1) * time_interval * self.smooth_term(step)
            time = time + ast_noise

        # update means3D, rotation, scales
        d_xyz, d_scaling, d_rotation, coronary_props = self.deform_model(means3D.detach(), time.detach())
        assert torch.isnan(coronary_props).sum() == 0, "coronary_props has NaN!"
        
        
        self._ensure_finite("d_xyz", d_xyz, step=step)
        self._ensure_finite("d_scaling", d_scaling, step=step)
        self._ensure_finite("d_rotation", d_rotation, step=step)
        
        means3D, rotation, scales = self.deform_model.deform(
            means3D, rotation, scales, d_xyz, d_rotation, d_scaling
        )
        self._ensure_finite("means3D", means3D, step=step)
        self._ensure_finite("rotation", rotation, step=step)
        self._ensure_finite("scales", scales, step=step)
        
        d_motion_mean, d_motion_var = pc.update_motions(d_xyz, d_scaling, d_rotation)   #EMA of motion

        gray_image, meta_whole = self._render(viewpoint_camera, means3D, rotation, scales, density)

        viewspace_points = meta_whole["viewspace_points"]
        radii = meta_whole["radii"]
        visibility_filter = radii > 0

        
        return XrayRendererOuputs(
            images={
                ImgT.GRAY: (gray_image, FloatColormapVisualizer(ColorMapName.GRAY)),
            },
            meta={
                MetaT.VIEWSPACE_POINTS: viewspace_points,
                MetaT.VISIBILITY_FILTER: visibility_filter,
                MetaT.RADII: radii,
                MetaT.D_MOTION_MEAN: d_motion_mean,
                MetaT.D_MOTION_VAR: d_motion_var,
                MetaT.D_MEANS3D: d_xyz,
                MetaT.D_ROTATION: d_rotation,
                MetaT.D_SCALES: d_scaling,
                MetaT.CORONARY_PROPS: coronary_props.squeeze(),
                MetaT.TIME: viewpoint_camera.time,
            },
            is_warm_up=viewpoint_camera.time.item() <= self.optimization_config.warm_up
        )
    
    def _render(
        self, 
        viewpoint_camera: Camera,
        means3D: Tensor,
        rotation: Tensor,
        scales: Tensor,
        density: Tensor
    ) -> Tuple[Tensor, dict[str, Tensor]]:

        rasterizer = GaussianRasterizer(GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height.item()),
            image_width=int(viewpoint_camera.width.item()),
            tanfovx=torch.tan(viewpoint_camera.fov_x * 0.5).item(),
            tanfovy=torch.tan(viewpoint_camera.fov_y * 0.5).item(),
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            mode=1, #cone beam
            debug=False,
        ))
        
        means_2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=means3D.device) + 0
        
        rendered_image: Tensor; radii: Tensor
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means_2D,
            opacities=density,
            scales=scales,
            rotations=rotation,
        )
        
        # DSA uses original light intensity rather than log(I0/I), therefore we need to convert back
        rendered_image = torch.exp( - torch.clamp(rendered_image, min=1e-3, max=14.0))
        
        meta = {
            "viewspace_points": means_2D,
            "radii": radii,
        }
        
        return rendered_image, meta
    
    
    def setup(self, stage: str, lightning_module, *args: Any, **kwargs: Any) -> Any:
        if stage == "fit":
            self.train_set_length = len(lightning_module.trainer.datamodule.dataparser_outputs.train_set)
        
        self.deform_model = self.deform_model_config.instantiate()
        
        total_steps = lightning_module.trainer.max_steps
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=total_steps*0.8)
    
    def training_setup(self, module) -> tuple[list[Optimizer]|None, list[LRScheduler]|None]:
        optimizer = torch.optim.adam.Adam(
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

        return [optimizer], [scheduler]