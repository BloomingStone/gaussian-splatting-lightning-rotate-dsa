from dataclasses import dataclass
from typing import Literal

import torch

from .saver import Saver, ThreadedSaverModule
from .x_ray_saver import (
    VtpSavePayload,
    SaveOutputsPayload,
    _save_outputs,
    build_nii_payloads,
)
from ..renderers.xray_4d_renderer import Xray4DRender
from ..deform_models import Deforms, GSParam
from ..dataparsers.xray_dataparser import XRayMeta
from ..gaussian_splatting import GaussianSplatting
from ..models.xray_4d_gaussian import Xray4DGaussianModel


@dataclass
class XRaySaver(Saver):
    save_ckpt: bool = False
    save_vtp: bool = False
    save_volume: bool = False
    save_label_threshold: float | None = None
    save_phase: float = 0.0
    save_time_or_type: float | Literal["mean", "std", "mean+2std"] = "mean+2std"
    default_save_time: float = 0.5

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
        assert isinstance(pc, Xray4DGaussianModel)
        deform_model = pl_module.renderer.deform_model

        means3D = pc.get_means().detach()
        match self.config.save_time_or_type:
            case "mean":
                density = pc.get_density_mean().detach()
            case "std":
                density = pc.get_density_std().detach()
            case "mean+2std":
                density = pc.get_density_mean().detach() + 2 * pc.get_density_std().detach()
            case float() as t:
                density = pc.get_density(t).detach()
            case _:
                raise ValueError(f"Unsupported save_time_or_type: {self.config.save_time_or_type}")
        
        rotation = pc.get_rotations().detach()
        scales = pc.get_scales().detach()

        if isinstance(self.config.save_time_or_type, float):
            save_time = self.config.save_time_or_type
        else:
            save_time = self.config.default_save_time
        
        deforms: Deforms = deform_model(
            xyz = means3D.detach(), 
            t = torch.full((means3D.shape[0], 1), save_time, device=means3D.device),
            phase = torch.full((means3D.shape[0], 1), self.config.save_phase, device=means3D.device),
        )

        source_deformed = deform_model.deform(
            GSParam(xyz=means3D, rotation=rotation, scaling=scales, density=density), deforms
        )
        
        vtp_payload = VtpSavePayload.build_from_gsparam(
            pl_module, 
            source_deformed
        ) if self.config.save_vtp else None
        
        datamodule = pl_module.get_datamodule()
        meta = datamodule.dataparser.meta
        assert isinstance(meta, XRayMeta)
        volume_shape = tuple(meta.volume_size)
        coronary_affine = meta.centering_affine

        payload = SaveOutputsPayload(
            vtp=vtp_payload,
            nifti=build_nii_payloads(
                pl_module, source_deformed, volume_shape, coronary_affine,
                self.config.save_volume, self.config.save_label_threshold,
            ),
        )
        self._submit_save_task(_save_outputs, payload)

        