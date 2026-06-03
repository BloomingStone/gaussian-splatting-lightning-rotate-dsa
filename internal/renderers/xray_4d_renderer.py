from typing import cast

import torch
from lightning import LightningModule


from .deformabel_xray_renderer import CoronaryDeformableXrayRenderer, XrayRendererOuputs
from ..cameras import Camera
from ..deform_models import Deforms, GSParam
from ..models.xray_4d_gaussian import Xray4DGaussianModel

class Xray4DRender(CoronaryDeformableXrayRenderer):
    def forward(
        self,
        viewpoint_camera: Camera,
        pc,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        render_types: list|None = None,
        **kwargs,
    ) -> XrayRendererOuputs:
        pc = cast(Xray4DGaussianModel, pc)
        xyz=pc.get_means().detach()
        N = xyz.shape[0]
        time = viewpoint_camera.time.unsqueeze(0).expand(N, -1)
        density = pc.get_density(float(viewpoint_camera.time.item())).detach()
        gs = GSParam(
            xyz=xyz,
            rotation=pc.get_rotations().detach(),
            scaling=pc.get_scales().detach(),
            density=density,
        )
        
        deforms: Deforms = self.deform_model(gs.xyz.detach(), time.detach())
        gs = self.deform_model.deform(gs, deforms)
        gray_image, meta_whole = self._render(viewpoint_camera, gs)

        viewspace_points = meta_whole["viewspace_points"]
        radii = meta_whole["radii"]
        visibility_filter = radii > 0
        
        mask = (gs.density > torch.quantile(gs.density, 0.90)).squeeze()
        if not torch.any(mask):
            mask = torch.ones_like(mask, dtype=torch.bool)

        density_std = pc.get_density_std(do_activate=True)
        
        gray_coronary, _ = self._render(
            viewpoint_camera,
            GSParam(
                xyz=gs.xyz[mask],
                rotation=gs.rotation[mask],
                scaling=gs.scaling[mask],
                density=density_std[mask],
            )
        )
        
        deforms_mean = pc.deforms_recorder.get_deforms_mean()
        deforms_var = pc.deforms_recorder.get_deforms_var()
        
        res = XrayRendererOuputs(
            images={
                XrayRendererOuputs.ImageTypes.GRAY: (gray_image, None),
                XrayRendererOuputs.ImageTypes.CORONARY: (gray_coronary, None),
            },
            meta={
                XrayRendererOuputs.MetaTypes.VIEWSPACE_POINTS: viewspace_points,
                XrayRendererOuputs.MetaTypes.VISIBILITY_FILTER: visibility_filter,
                XrayRendererOuputs.MetaTypes.RADII: radii,
                XrayRendererOuputs.MetaTypes.DEFORMS_MEAN: deforms_mean,
                XrayRendererOuputs.MetaTypes.DEFORMS_VAR: deforms_var,
                XrayRendererOuputs.MetaTypes.DEFORMS: deforms,
                XrayRendererOuputs.MetaTypes.TIME: viewpoint_camera.time,
                XrayRendererOuputs.MetaTypes.MASK: mask,
            },
            is_warm_up=False
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
        pc = cast(Xray4DGaussianModel, pc)

        xyz=pc.get_means()
        N = xyz.shape[0]
        time = viewpoint_camera.time.unsqueeze(0).expand(N, -1)
        density = pc.get_density(time)
        
        gs = GSParam(
            xyz=xyz,
            rotation=pc.get_rotations(),
            scaling=pc.get_scales(),
            density=density,
        )
        
        if self.optimization_config.enable_ast:     # add AST noise
            time_interval = 1 / ((step % self.train_set_length) + 1)
            ast_noise = torch.randn(1, 1, device=gs.xyz.device).expand(N, -1) * time_interval * self.smooth_term(step)
            time = time + ast_noise

        if module.global_step == 144:
            pass
        
        deforms = self.deform_model(gs.xyz.detach(), time.detach())
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
        
        mask = (gs.density > torch.quantile(gs.density, 0.90)).squeeze()
        if not torch.any(mask):
            mask = torch.ones_like(mask, dtype=torch.bool)
        
        return XrayRendererOuputs(
            images={
                XrayRendererOuputs.ImageTypes.GRAY: (gray_image, None),
            },
            meta={
                XrayRendererOuputs.MetaTypes.VIEWSPACE_POINTS: viewspace_points,
                XrayRendererOuputs.MetaTypes.VISIBILITY_FILTER: visibility_filter,
                XrayRendererOuputs.MetaTypes.RADII: radii,
                XrayRendererOuputs.MetaTypes.DEFORMS_MEAN: deforms_mean,
                XrayRendererOuputs.MetaTypes.DEFORMS_VAR: deforms_var,
                XrayRendererOuputs.MetaTypes.DEFORMS: deforms,
                XrayRendererOuputs.MetaTypes.TIME: viewpoint_camera.time,
                XrayRendererOuputs.MetaTypes.MASK: mask,
            },
            is_warm_up=False
        )