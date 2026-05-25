from typing import Any, Optional, Tuple, cast
from dataclasses import dataclass

import torch
from torch import Tensor
from lightning import LightningModule
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel

from .renderer import Renderer, RendererOutputInfo, RendererOutputTypes
from ..cameras import Camera
from ..deform_models import DeformModel, DefromModelConfig, Deforms, GSParam
from ..utils.general_utils import get_linear_noise_func


@dataclass
class DeformableRendererOptimizationConfig:
    lr: float = 1e-3
    max_steps: int = 40_000
    lr_final_factor: float = 0.002
    eps: float = 1e-8
    warm_up: int = -1
    enable_ast: bool = True
    log_gradients: bool = True
    grad_log_interval: int = 100
    density_ramp_steps: int = 2000


@dataclass
class RenderRes:
    gray_image: Tensor
    gray_coronary: Tensor | None
    viewspace_points: Tensor
    visibility_filter: Tensor
    radii: Tensor
    
    deforms_mean: dict[str, Tensor]
    deforms_var: dict[str, Tensor]
    
    deforms: Deforms
    
    time: Tensor
    density_mask: Tensor|None = None
    
    in_warm_up: bool = False

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
    ) -> RenderRes:
        pc = cast(XrayCoronaryGaussianModel, pc)
        gs = GSParam(
            xyz=pc.get_means().detach(),
            rotation=pc.get_rotations().detach(),
            scaling=pc.get_scales().detach(),
            density=pc.get_density().detach(),
        )
        
        time = viewpoint_camera.time.unsqueeze(0).expand(gs.xyz.shape[0], -1)
        deforms: Deforms = self.deform_model(gs.xyz.detach(), time.detach())
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
        
        res = RenderRes(
            gray_image, gray_coronary,                  # rendered
            viewspace_points, visibility_filter, radii, # grad meta
            deforms_mean, deforms_var, deforms,         # deforms and mean & var
            viewpoint_camera.time, mask,                 # other info     
            in_warm_up=False
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
    )-> RenderRes:
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
        
        if self.optimization_config.enable_ast:     # add AST noise
            time_interval = 1 / ((step % self.train_set_length) + 1)
            ast_noise = torch.randn(1, 1, device=gs.xyz.device).expand(N, -1) * time_interval * self.smooth_term(step)
            time = time + ast_noise

        if module.global_step == 144:
            pass
        
        deforms = self.deform_model(gs.xyz.detach(), time.detach())
        
        # apply density amplitude ramp for models that output d_density (with_flow)
        ramp_steps = getattr(self.optimization_config, "density_ramp_steps", 2000)
        warm_up = self.optimization_config.warm_up
        if ramp_steps > 0:
            alpha = float(max(0.0, min(1.0, (step - warm_up) / ramp_steps)))
        else:
            alpha = 1.0
        if hasattr(deforms, "d_density"):
            try:
                deforms.d_density = deforms.d_density * alpha
            except Exception:
                pass

        if step > self.optimization_config.warm_up:
            # update means3D, rotation, scales
            gs = self.deform_model.deform(gs, deforms)
            deforms_mean, deforms_var = pc.deforms_recorder.update(deforms)
        else:
            deforms_mean = pc.deforms_recorder.get_deforms_mean()
            deforms_var = pc.deforms_recorder.get_deforms_var()

        gray_image, meta_whole = self._render(viewpoint_camera, gs)

        viewspace_points = meta_whole["viewspace_points"]
        radii = meta_whole["radii"]
        visibility_filter = radii > 0

        gray_coronary = None
        
        mask = (gs.density > torch.quantile(gs.density, 0.90)).squeeze()
        if not torch.any(mask):
            mask = torch.ones_like(mask, dtype=torch.bool)
        
        return RenderRes(
            gray_image, gray_coronary,                  # rendered
            viewspace_points, visibility_filter, radii, # grad meta
            deforms_mean, deforms_var, deforms,         # deforms and mean & var
            viewpoint_camera.time, mask,                 # other info     
            in_warm_up=False
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