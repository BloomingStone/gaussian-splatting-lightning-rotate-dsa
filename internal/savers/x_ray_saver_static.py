from dataclasses import dataclass
import numpy as np

from .saver import Saver, ThreadedSaverModule
from .x_ray_saver import (
    VtpSavePayload,
    NiftiSavePayload,
    SaveOutputsPayload,
    _save_outputs,
)
from ..deform_models import GSParam
from ..dataparsers.threeDGR_parser import ThreeDGRCarMeta
from ..gaussian_splatting import GaussianSplatting


@dataclass
class XRaySaver(Saver):
    save_ckpt: bool = True
    save_vtp: bool = True
    save_nii: bool = True
    
    threshold: float|None = None

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

        means3D = pc.get_means().detach()
        density = pc.get_density().detach()
        rotation = pc.get_rotations().detach()
        scales = pc.get_scales().detach()
        gs_param = GSParam(xyz=means3D, rotation=rotation, scaling=scales, density=density)

        vtp_payload = None
        if self.config.save_vtp:
            vtp_payload = VtpSavePayload.build_from_gsparam(pl_module, gs_param)
        
        datamodule = pl_module.get_datamodule()
        meta = datamodule.dataparser.meta
        assert isinstance(meta, ThreeDGRCarMeta)
        volume_shape = tuple(meta.projs_meta.param.nVoxels)
        coronary_affine = meta.projs_meta.param.affine

        nifti_payload = None
        if self.config.save_nii:
            nifti_payload = NiftiSavePayload.build_from_gsparam(
                pl_module, 
                gs_param, 
                volume_shape, coronary_affine
            )
            if self.config.threshold is not None:
                nifti_payload.volume = (nifti_payload.volume > self.config.threshold).astype(np.uint8)  # binarize the volume for saving

        payload = SaveOutputsPayload(vtp=vtp_payload, nifti=nifti_payload)
        self._submit_save_task(_save_outputs, payload)
        