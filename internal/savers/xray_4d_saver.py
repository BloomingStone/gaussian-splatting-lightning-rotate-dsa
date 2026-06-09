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
from ..renderers.xray_4d_renderer import Xray4DRender
from ..deform_models import Deforms, GSParam
from ..dataparsers.xray_dataparser import XRayMeta
from ..gaussian_splatting import GaussianSplatting


@dataclass
class XRaySaver(Saver):
    save_ckpt: bool = True
    save_vtp: bool = True
    save_nii: bool = True
    save_phase: float = 0.0
    save_time_or_type: float | Literal["mean", "std", "var", "mean+2std"] = "mean+2std"

    def instantiate(self, *args, **kwargs) -> "XRaySaverModule":
        return XRaySaverModule(self)


class XRaySaverModule(ThreadedSaverModule):
    def __init__(self, config: XRaySaver):
        super().__init__(thread_name_prefix="xray-save")
        self.config = config

    def save(self, pl_module: GaussianSplatting):
        if pl_module.trainer.global_rank != 0:
            return

        if self.config.save_ckpt:
            self._save_ckpt(pl_module)

        pc = pl_module.gaussian_model
        assert isinstance(pl_module.renderer, Xray4DRender)
        deform_model = pl_module.renderer.deform_model

        means3D = pc.get_means().detach()
        density = pc.get_density_mean().detach()
        rotation = pc.get_rotations().detach()
        scales = pc.get_scales().detach()

        deforms: Deforms = deform_model(
            means3D.detach(),
            torch.full((means3D.shape[0], 1), self.config.save_phase, device=means3D.device),
        )

        source_deformed = deform_model.deform(
            GSParam(xyz=means3D, rotation=rotation, scaling=scales, density=density), deforms
        )
        means3D, rotation, scales = source_deformed.xyz, source_deformed.rotation, source_deformed.scaling

        vtp_payload = None
        if self.config.save_vtp:
            gs_param = GSParam(xyz=means3D, rotation=rotation, scaling=scales, density=density)
            vtp_payload = VtpSavePayload.build_from_gsparam(pl_module, gs_param)
        
        datamodule = pl_module.get_datamodule()
        meta = datamodule.dataparser.meta
        assert isinstance(meta, XRayMeta)
        volume_shape = tuple(meta.volume_size)
        coronary_affine = meta.centering_affine

        nifti_payload = None
        if self.config.save_nii:
            nifti_payload = NiftiSavePayload.build_from_gsparam(
                pl_module, 
                GSParam(xyz=means3D, rotation=rotation, scaling=scales, density=density), 
                volume_shape, coronary_affine
            )

        payload = SaveOutputsPayload(vtp=vtp_payload, nifti=nifti_payload)
        self._submit_save_task(_save_outputs, payload)
        