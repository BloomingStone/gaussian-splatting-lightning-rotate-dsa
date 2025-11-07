import torch

from .deformable_renderer import DeformableRenderer, DeformNetworkConfig, XYZEncodingConfig, TimeEncodingConfig, DeformableRendererOptimizationConfig
from ..cameras import Camera
from ..models.gaussian import GaussianModel


class DeformableXrayRenderer(DeformableRenderer):
    def __init__(
        self,
        deform_network: DeformNetworkConfig,
        xyz_encoding: XYZEncodingConfig,
        time_encoding: TimeEncodingConfig,
        optimization: DeformableRendererOptimizationConfig, 
    ):
        super().__init__(deform_network, xyz_encoding, time_encoding, optimization)
    
    def _render(
        self,
        d_xyz,
        d_rotation,
        d_scaling,
        viewpoint_camera: Camera,
        pc: GaussianModel,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
    ):
        res = super()._render(
            d_xyz,
            d_rotation,
            d_scaling,
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
        )
        res["render"] = res["render"].mean(dim=0, keepdim=True).repeat(3, 1, 1)
        return res