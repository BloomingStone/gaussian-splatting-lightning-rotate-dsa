from typing import Any, Optional, Tuple, cast
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.adam import Adam
from lightning import LightningModule
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel

from .renderer import Renderer, RendererOutputs
from ..cameras import Camera
from ..deform_models import DeformModel, DefromModelConfig, Deforms, GSParam
from ..utils.general_utils import get_linear_noise_func
from ..visualizers import FloatColormapVisualizer, ColorMapName, Visualizer


@dataclass
class XrayRendererOuputs(RendererOutputs):
    class ImageTypes(StrEnum):
        GRAY = "gray"
        CORONARY = "coronary"
    
    class MetaTypes(StrEnum):
        VIEWSPACE_POINTS = "viewspace_points"
        VISIBILITY_FILTER = "visibility_filter"
        RADII = "radii"
        
        DEFORMS_MEAN = "deforms_mean"
        DEFORMS_VAR = "deforms_var"
        
        DEFORMS = "deforms"
        
        TIME = "time"
        PHASE = "phase"
        
        MASK = "mask"
    
    images: dict[ImageTypes, tuple[Tensor, Visualizer|None]]    # type: ignore
    meta: dict[MetaTypes, Any]   # type: ignore
    
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
    def deforms_mean(self) -> dict[str, Tensor]:
        return self.meta[self.MetaTypes.DEFORMS_MEAN]

    @property
    def deforms_var(self) -> dict[str, Tensor]:
        return self.meta[self.MetaTypes.DEFORMS_VAR]

    @property
    def deforms(self) -> Deforms:
        return self.meta[self.MetaTypes.DEFORMS]

    @property
    def time(self) -> Tensor:
        return self.meta[self.MetaTypes.TIME]
    
    @property
    def phase(self) -> Tensor:
        return self.meta[self.MetaTypes.PHASE]
    
    @property
    def mask(self) -> Optional[Tensor]:
        return self.meta.get(self.MetaTypes.MASK, None)

ImgT = XrayRendererOuputs.ImageTypes
MetaT = XrayRendererOuputs.MetaTypes



@dataclass
class DeformableRendererOptimizationConfig:
    lr: float = 1e-3
    max_steps: int = 40_000
    lr_final_factor: float = 0.002
    eps: float = 1e-8
    warm_up: int = -1
    enable_ast: bool = True


class CoronaryDeformableXrayRenderer(Renderer):
    deform_model: DeformModel
    
    def __init__(
            self,
            optimization_config: DeformableRendererOptimizationConfig,
            deform_model_config: DefromModelConfig
    ) -> None:
        super().__init__()
        self.deform_model_config = deform_model_config
        self.optimization_config = optimization_config
        self._grad_hook_registered = False

    def forward(
        self,
        viewpoint_camera: Camera,
        pc,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        render_types: list|None = None,
        **kwargs,
    ) -> XrayRendererOuputs:
        pc = cast(XrayCoronaryGaussianModel, pc)
        gs = GSParam(
            xyz=pc.get_means().detach(),
            rotation=pc.get_rotations().detach(),
            scaling=pc.get_scales().detach(),
            density=pc.get_density().detach(),
        )
        
        time = viewpoint_camera.time.unsqueeze(0).expand(gs.xyz.shape[0], -1)
        phase = viewpoint_camera.phase.unsqueeze(0).expand(gs.xyz.shape[0], -1)
        deforms: Deforms = self.deform_model(gs.xyz.detach(), t=time.detach(), phase=phase.detach())
        gs = self.deform_model.deform(gs, deforms)
        gray_image, meta_whole = self._render(viewpoint_camera, gs)

        viewspace_points = meta_whole["viewspace_points"]
        radii = meta_whole["radii"]
        visibility_filter = radii > 0
        
        mask = (gs.density > torch.quantile(gs.density, 0.90)).squeeze()
        if not torch.any(mask):
            mask = torch.ones_like(mask, dtype=torch.bool)
        
        gray_coronary, _ = self._render(
            viewpoint_camera,
            GSParam(
                xyz=gs.xyz[mask],
                rotation=gs.rotation[mask],
                scaling=gs.scaling[mask],
                density=gs.density[mask],
            )
        )
        
        deforms_mean = pc.deforms_recorder.get_deforms_mean()
        deforms_var = pc.deforms_recorder.get_deforms_var()
        
        return XrayRendererOuputs(
            images={
                ImgT.GRAY: (gray_image, FloatColormapVisualizer(ColorMapName.GRAY)),
                ImgT.CORONARY: (gray_coronary, FloatColormapVisualizer(ColorMapName.GRAY))
            },
            meta={
                MetaT.VIEWSPACE_POINTS: viewspace_points,
                MetaT.VISIBILITY_FILTER: visibility_filter,
                MetaT.RADII: radii,
                MetaT.DEFORMS_MEAN: deforms_mean,
                MetaT.DEFORMS_VAR: deforms_var,
                MetaT.DEFORMS: deforms,
                MetaT.TIME: viewpoint_camera.time
            },
            is_warm_up=False
        )


    def training_forward(
        self, 
        step: int, 
        module: LightningModule, 
        viewpoint_camera: Camera, 
        pc,
        bg_color: torch.Tensor,
        render_types: list|None = None,
        **kwargs
    )-> XrayRendererOuputs:
        pc = cast(XrayCoronaryGaussianModel, pc)
        # clone properties
        gs = GSParam(
            xyz=pc.get_means(),
            rotation=pc.get_rotations(),
            scaling=pc.get_scales(),
            density=pc.get_density(),
        )

        N = gs.xyz.shape[0]
        time = viewpoint_camera.time.unsqueeze(0).expand(N, -1)
        phase = viewpoint_camera.phase.unsqueeze(0).expand(N, -1)
        
        if self.optimization_config.enable_ast:     # add AST noise
            time_interval = 1 / ((step % self.train_set_length) + 1)
            ast_noise = torch.randn(1, 1, device=gs.xyz.device).expand(N, -1) * time_interval * self.smooth_term(step)
            time = time + ast_noise
            phase = phase + ast_noise
        
        deforms = self.deform_model(gs.xyz.detach(), t=time.detach(), phase=phase.detach())
        
        is_warm_up = step < self.optimization_config.warm_up
        
        if is_warm_up:
            deforms_mean = pc.deforms_recorder.get_deforms_mean()
            deforms_var = pc.deforms_recorder.get_deforms_var()
        else:
            gs = self.deform_model.deform(gs, deforms)
            deforms_mean, deforms_var = pc.deforms_recorder.update(deforms)

        gray_image, meta_whole = self._render(viewpoint_camera, gs)

        if is_warm_up:
            # dummy operation to make sure deforms receive gradients during warm-up
            # otherwise, there might be an error (AssertionError: No inf checks were recorded for this optimizer.) when 
            # the optimizer tries to step with zero gradients
            gray_image = gray_image + 0.0 * (
                deforms.d_xyz.sum() + deforms.d_rotation.sum() + deforms.d_scaling.sum()
            )

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
                MetaT.DEFORMS_MEAN: deforms_mean,
                MetaT.DEFORMS_VAR: deforms_var,
                MetaT.DEFORMS: deforms,
                MetaT.TIME: viewpoint_camera.time,
                MetaT.PHASE: viewpoint_camera.phase,
            },
            is_warm_up=is_warm_up
        )
    
    def _render(
        self, 
        viewpoint_camera: Camera,
        gs: GSParam,
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
        
        means_2D = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=gs.xyz.device) + 0
        
        rendered_image: Tensor; radii: Tensor
        rendered_image, radii = rasterizer(
            means3D=gs.xyz,
            means2D=means_2D,
            opacities=gs.density,
            scales=gs.scaling,
            rotations=gs.rotation,
        )
        
        # DSA uses original light intensity rather than log(I0/I), therefore we need to convert back
        rendered_image = torch.exp( - torch.clamp(rendered_image, min=1e-3, max=14.0))
        # rendered_image = 1 - rendered_image
        
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
        optimizer = Adam(
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
