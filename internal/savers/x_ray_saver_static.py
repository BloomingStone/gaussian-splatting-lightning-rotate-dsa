from dataclasses import dataclass

from .saver import Saver, ThreadedSaverModule
from .x_ray_saver import (
    VtpSavePayload,
    NiftiSavePayload,
    SaveOutputsPayload,
    _save_outputs,
)
from ..deform_models import GSParam
from ..gaussian_splatting import GaussianSplatting
from ..dataparsers.xray_dataparser import XRayMeta


@dataclass
class XRaySaver_Static(Saver):
    save_ckpt: bool = True
    save_vtp: bool = True
    save_nii: bool = True
    
    def instantiate(self, *args, **kwargs) -> "XRaySaverModule_Static":
        return XRaySaverModule_Static(self)


class XRaySaverModule_Static(ThreadedSaverModule):
    def __init__(self, config: XRaySaver_Static):
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
        