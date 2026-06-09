from dataclasses import dataclass
from typing import Literal

import torch

from .saver import Saver, ThreadedSaverModule
from .x_ray_saver import (
    VtpSavePayload,
    NiftiSavePayload,
    SaveOutputsPayload,
    _save_outputs,
)
from ..renderers.deformabel_xray_renderer_with_flow import DeformableXrayRendererWithFlow
from ..deform_models import Deforms, GSParam
from ..models.xray_coronary_gaussian_with_flow import XrayCoronaryGaussianModelWithFlow
from ..gaussian_splatting import GaussianSplatting
from ..dataparsers.xray_dataparser import XRayMeta


@dataclass
class XRaySaver_Flow(Saver):
    save_ckpt: bool = True
    save_vtp: bool = True
    save_nii: bool = True
    save_phase: float = 0.0
    save_time_or_type: float | Literal["mean", "std", "var", "mean+2std"] = "mean+2std"
    
    def instantiate(self, *args, **kwargs) -> "XRaySaverModule_Flow":
        return XRaySaverModule_Flow(self)


class XRaySaverModule_Flow(ThreadedSaverModule):
    def __init__(self, config: XRaySaver_Flow):
        super().__init__(thread_name_prefix="xray-save")
        self.config = config


    def save(self, pl_module: GaussianSplatting):
        if pl_module.trainer.global_rank != 0:
            return

        if self.config.save_ckpt:
            self._save_ckpt(pl_module)
        
        pc = pl_module.gaussian_model
        assert isinstance(pc, XrayCoronaryGaussianModelWithFlow)
        
        renderer = pl_module.renderer
        assert isinstance(renderer, DeformableXrayRendererWithFlow)
        deform_model = renderer.deform_model
        
        means3D = pc.get_means().detach()
        density = pc.get_density().detach()
        rotation = pc.get_rotations().detach()
        scales = pc.get_scales().detach()
        
        if isinstance(self.config.save_time_or_type, float):
            t = float(self.config.save_time_or_type)
        else:
            t = 0
        
        
        deforms: Deforms = deform_model(
            xyz   = means3D.detach(), 
            t     = torch.full((means3D.shape[0], 1), t, device=means3D.device),   # input time is t
            phase = torch.full((means3D.shape[0], 1), self.config.save_phase, device=means3D.device),   # input phase is save_phase
        )
        
        source_deformed = deform_model.deform(
            GSParam(xyz=means3D, rotation=rotation, scaling=scales, density=density), deforms
        )
        means3D, rotation, scales = source_deformed.xyz, source_deformed.rotation, source_deformed.scaling
        
        d_density_var = pc.deforms_recorder.get_deforms_var()["d_density"]
        match self.config.save_time_or_type:
            case "mean":
                pass   # keep original density, which is roughly the mean density at different time steps
            case "var":
                density = d_density_var
            case "std": 
                density = d_density_var.sqrt()
            case "mean+2std":
                density = density + 2 * d_density_var.sqrt()
            case float():
                density = source_deformed.density   # use the deformed density at time t as density
            case _:
                raise ValueError(f"Invalid save_time_or_type: {self.config.save_time_or_type}")
        
        gs_param = GSParam(xyz=means3D, rotation=rotation, scaling=scales, density=density)
        vtp_payload = VtpSavePayload.build_from_gsparam(
            pl_module, 
            gs_param
        ) if self.config.save_vtp else None
        
        datamodule = pl_module.get_datamodule()
        meta = datamodule.dataparser.meta
        assert isinstance(meta, XRayMeta)
        volume_shape = tuple(meta.volume_size)
        coronary_affine = meta.centering_affine

        nifti_payload = NiftiSavePayload.build_from_gsparam(
            pl_module, 
            gs_param,
            volume_shape,
            coronary_affine,
        ) if self.config.save_nii else None


        payload = SaveOutputsPayload(vtp=vtp_payload, nifti=nifti_payload)
        self._submit_save_task(_save_outputs, payload)
        