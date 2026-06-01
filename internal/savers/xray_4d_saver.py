from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Literal

import torch
import numpy as np

from .x_ray_saver import XRaySaver as BaseXRaySaver, XRaySaverModule as BaseXRaySaverModule, deform_field_to_volume, gaussians_to_volume_by_Rasterizer
from ..renderers.xray_4d_renderer import Xray4DRender
from ..deform_models import Deforms, GSParam


@dataclass
class XRaySaver(BaseXRaySaver):
    save_time_or_type: float | Literal["mean", "std", "var", "mean+2std"] = "mean+2std"

    def instantiate(self, *args, **kwargs) -> "XRaySaverModule":
        return XRaySaverModule(self)


class XRaySaverModule(BaseXRaySaverModule):
    def __init__(self, config: XRaySaver):
        super().__init__(config)
        self.config = config
        self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xray-save")
        self._pending_save: Future | None = None

    def save(self, pl_module):
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
            vtp_payload = self._make_vtp_save_payload(pl_module, gs_param)

        nifti_payload = None
        if self.config.save_nii:
            nifti_payload = self._make_nii_save_payload(pl_module, GSParam(xyz=means3D, rotation=rotation, scaling=scales, density=density))

        if self._pending_save is not None and not self._pending_save.done():
            self._pending_save.result()

        payload = self.SaveOutputsPayload(vtp=vtp_payload, nifti=nifti_payload)
        self._pending_save = self._save_executor.submit(self._save_outputs, payload)
        