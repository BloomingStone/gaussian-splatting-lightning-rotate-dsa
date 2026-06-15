from typing import Any, Optional, Tuple, cast
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor
from lightning import LightningModule
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel
from .renderer import Renderer, RendererOutputs
from ..deform_models import GSParam
from ..cameras import Camera
from ..visualizers import FloatColormapVisualizer, ColorMapName, Visualizer


@dataclass
class RendererOptimizationConfig:
    lr: float = 1e-3
    lr_final_factor: float = 0.002
    eps: float = 1e-8


@dataclass
class XrayRendererOuputs(RendererOutputs):
    class ImageTypes(StrEnum):
        GRAY = "gray"
    
    class MetaTypes(StrEnum):
        VIEWSPACE_POINTS = "viewspace_points"
        VISIBILITY_FILTER = "visibility_filter"
        RADII = "radii"
    
    images: dict[ImageTypes, tuple[Tensor, Visualizer|None]]    # type: ignore
    meta: dict[MetaTypes, Any]   # type: ignore

    @property
    def gray_image(self) -> Tensor:
        return self.images[self.ImageTypes.GRAY][0]

    @property
    def viewspace_points(self) -> Tensor:
        return self.meta[self.MetaTypes.VIEWSPACE_POINTS]

    @property
    def visibility_filter(self) -> Tensor:
        return self.meta[self.MetaTypes.VISIBILITY_FILTER]

    @property
    def radii(self) -> Tensor:
        return self.meta[self.MetaTypes.RADII]

ImgT = XrayRendererOuputs.ImageTypes
MetaT = XrayRendererOuputs.MetaTypes


class XrayRenderer(Renderer):

    def __init__(
            self,
            optimization_config: RendererOptimizationConfig,
    ) -> None:
        super().__init__()
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

        gray_image, meta_whole = self._render(viewpoint_camera, gs)

        viewspace_points = meta_whole["viewspace_points"]
        radii = meta_whole["radii"]
        visibility_filter = radii > 0
        
        res = XrayRendererOuputs(
            images={
                ImgT.GRAY: (gray_image, FloatColormapVisualizer(ColorMapName.GRAY)),
            },
            meta={
                MetaT.VIEWSPACE_POINTS: viewspace_points,
                MetaT.VISIBILITY_FILTER: visibility_filter,
                MetaT.RADII: radii
            },
        )

        return res
        

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

        gray_image, meta_whole = self._render(viewpoint_camera, gs)

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
                MetaT.RADII: radii
            },
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
        
        total_steps = lightning_module.trainer.max_steps