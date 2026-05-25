from typing import cast, override

import torch

from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel

from ..cameras import Camera
from ..deform_models import GSParam
from ..deform_models.deform_with_flow import DeformsWithFlow, DeformWithFlowConfig

from .deformabel_xray_renderer import DeformableRendererOptimizationConfig, RenderRes, CoronaryDeformableXrayRenderer

class DeformableXrayRendererWithFlow(CoronaryDeformableXrayRenderer):
    @override
    def __init__(
        self,
        optimization_config: DeformableRendererOptimizationConfig,
        deform_model_config: DeformWithFlowConfig
    ):
        super().__init__(optimization_config, deform_model_config)
        self.deform_model_config = deform_model_config
    
    @override
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
        deforms: DeformsWithFlow = self.deform_model(gs.xyz.detach(), time.detach())
        gs = self.deform_model.deform(gs, deforms)
        gray_image, meta_whole = self._render(viewpoint_camera, gs)

        viewspace_points = meta_whole["viewspace_points"]
        radii = meta_whole["radii"]
        visibility_filter = radii > 0
        
        mask = (gs.density > torch.quantile(gs.density, 0.90)).squeeze()
        if not torch.any(mask):
            mask = torch.ones_like(mask, dtype=torch.bool)
        
        deforms_mean = pc.deforms_recorder.get_deforms_mean()
        deforms_var = pc.deforms_recorder.get_deforms_var()
        
        assert "d_density" in deforms_mean and "d_density" in deforms_var, "DeformsMARecoder does not have d_density component."
        # use mean + 2*std as the uncertainty-aware density, which is roughly the upper bound of density with 95% confidence if 
        # we assume the deforms follow a normal distribution. 
        # This is to avoid under-estimate the density for points with high uncertainty.
        d_density_upper= deforms_mean["d_density"].abs() + 2 * deforms_var["d_density"].sqrt()   
        gray_coronary, _ = self._render(
            viewpoint_camera,
            GSParam(
                xyz=gs.xyz[mask],
                rotation=gs.rotation[mask],
                scaling=gs.scaling[mask],
                density=d_density_upper[mask],
            )
        )
        
        
        res = RenderRes(
            gray_image, gray_coronary,                  # rendered
            viewspace_points, visibility_filter, radii, # grad meta
            deforms_mean, deforms_var, deforms,         # deforms and mean & var
            viewpoint_camera.time, mask,                 # other info     
            in_warm_up=False
        )
        return res