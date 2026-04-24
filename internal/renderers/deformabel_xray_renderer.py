from typing import Any, Optional, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor
from lightning import LightningModule
from jaxtyping import Float32
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from .renderer import Renderer, RendererOutputInfo, RendererOutputTypes
from ..cameras import Camera
from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel
from ..cameras import Camera
from ..deform_models import DeformModel, DefromModelConfig
from ..utils.network_factory import NetworkFactory
from ..utils.general_utils import get_linear_noise_func
from ..utils.gaussian_utils import GaussianTransformUtils


@dataclass
class DeformableRendererOptimizationConfig:
    lr: float = 1e-3
    max_steps: int = 40_000
    lr_final_factor: float = 0.002
    eps: float = 1e-8
    warm_up: int = 0
    enable_ast: bool = True
    log_gradients: bool = True
    grad_log_interval: int = 100


@dataclass
class RenderRes:
    gray_image: Float32[Tensor, "1 h w"]
    gray_coronary: Float32[Tensor, "1 h w"] | None
    viewspace_points: Float32[Tensor, "n 2"]
    visibility_filter: Float32[Tensor, "n"]
    radii: Float32[Tensor, "n"]
    
    d_motion_mean: Tensor
    d_motion_var: Tensor
    
    d_means3D: Tensor
    d_rotation: Tensor
    d_scales: Tensor
    
    time: Tensor
    
    in_warm_up: bool

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
    ) -> RenderRes:
        means3D = pc.get_means().detach()
        density = pc.get_density().detach()
        rotation = pc.get_rotations().detach()
        scales = pc.get_scales().detach()
        
        time = viewpoint_camera.time.unsqueeze(0).expand(means3D.shape[0], -1)
        
        d_xyz, d_scaling, d_rotation = self.deform_model(means3D.detach(), time.detach())
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
        
        res = RenderRes(
            gray_image, gray_coronary,      # rendered
            viewspace_points, visibility_filter, radii, # grad meta
            d_motion_mean, d_motion_var,    # moving mean & var
            d_xyz, d_rotation, d_scaling,      # deform properties
            viewpoint_camera.time,
            in_warm_up=False
        )
        return res
        

    def training_forward(
        self, 
        step: int, 
        module: LightningModule, 
        viewpoint_camera: Camera, 
        pc: XrayCoronaryGaussianModel,
        **kwargs
    )-> RenderRes:
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
        d_xyz, d_scaling, d_rotation= self.deform_model(means3D.detach(), time.detach())
        self._ensure_finite("d_xyz", d_xyz, step=step)
        self._ensure_finite("d_scaling", d_scaling, step=step)
        self._ensure_finite("d_rotation", d_rotation, step=step)
        
        means3D, rotation, scales = self.deform_model.deform(
            means3D, rotation, scales, d_xyz, d_rotation, d_scaling
        )
        self._ensure_finite("means3D", means3D, step=step)
        self._ensure_finite("rotation", rotation, step=step)
        self._ensure_finite("scales", scales, step=step)
        d_motion_mean, d_motion_var = pc.update_motions(d_xyz, d_scaling, d_rotation)

        gray_image, meta_whole = self._render(viewpoint_camera, means3D, rotation, scales, density)

        viewspace_points = meta_whole["viewspace_points"]
        radii = meta_whole["radii"]
        visibility_filter = radii > 0

        gray_coronary = None
        
        return RenderRes(
            gray_image, gray_coronary,      # rendered
            viewspace_points, visibility_filter, radii, # grad meta
            d_motion_mean, d_motion_var,    # moving mean & var
            d_xyz, d_rotation, d_scaling,      # deforms
            viewpoint_camera.time,
            in_warm_up=step <= self.optimization_config.warm_up
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