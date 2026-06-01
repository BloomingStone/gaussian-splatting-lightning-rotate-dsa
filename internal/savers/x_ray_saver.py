from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from typing import cast

import numpy as np
import torch
import pyvista as pv
from nibabel import loadsave as nib_io
from nibabel.nifti1 import Nifti1Image
from xray_gaussian_rasterization_voxelization import (
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

from .saver import Saver, SaverModule
from ..gaussian_splatting import GaussianSplatting
from ..renderers.deformabel_xray_renderer import CoronaryDeformableXrayRenderer
from ..deform_models import DeformModel, Deforms, GSParam
from ..dataparsers.xray_dataparser import XRayMeta


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    q: (N, 4)  (w, x, y, z)
    return: (N, 3, 3)
    """
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(dim=1)

    B = q.shape[0]

    R = torch.zeros((B, 3, 3), device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R

@torch.no_grad()
def deform_field_to_volume(
    deform_model: DeformModel,
    shape: tuple[int, int, int],
    affine: np.ndarray,
    zoomed_shape: tuple[int, int, int],
    batch_size: int = 256*256,
    save_phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(deform_model.parameters()).device
    orgin = affine[:3, 3]
    A = affine[:3, :3]
    shape_in_world = A @ np.array(shape)
    
    # new_A @ zoomed_shape = A @ shape  => new_A = A @ shape / zoomed_shape
    new_A = A @ np.diag(np.array(shape) / np.array(zoomed_shape))
    new_affine = np.eye(4)
    new_affine[:3, :3] = new_A
    new_affine[:3, 3] = orgin
    
    x = torch.linspace(orgin[0], shape_in_world[0]+orgin[0], zoomed_shape[0])
    y = torch.linspace(orgin[1], shape_in_world[1]+orgin[1], zoomed_shape[1])
    z = torch.linspace(orgin[2], shape_in_world[2]+orgin[2], zoomed_shape[2])
    xyz = torch.stack(torch.meshgrid(x, y, z), dim=-1)
    xyz = xyz.reshape(-1, 3)
    
    xyz = xyz.to(device)
    dxyz_chunks: list[torch.Tensor] = []

    for start in range(0, xyz.shape[0], batch_size):
        end = min(start + batch_size, xyz.shape[0])
        xyz_batch = xyz[start:end]
        t_batch = torch.full((xyz_batch.shape[0], 1), save_phase, device=device)  # input phase is save_phase
        deforms: Deforms = deform_model(xyz_batch, t_batch)
        dxyz_chunks.append(deforms.d_xyz.cpu().float())

    dxyz_volume = torch.cat(dxyz_chunks, dim=0).reshape(zoomed_shape + (3,))
    return dxyz_volume.numpy(), new_affine


@torch.no_grad()
def gaussians_to_volume_by_Rasterizer(
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotation: torch.Tensor,
    density: torch.Tensor,
    shape: tuple[int, int, int],
    affine: np.ndarray
) -> np.ndarray:
    A = affine[:3, :3]
    spacing = np.linalg.norm(A, axis=0)    # make sure the spacing is positive
    sVoxel = spacing * np.array(shape)  # sVoxel = size of whole volume (length of XYZ, not the voxel size)
    center = affine[:3, 3] + 0.5 * A @ shape   # center of the volume in world coordinates
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=1,
        nVoxel_x=int(shape[0]),
        nVoxel_y=int(shape[1]),
        nVoxel_z=int(shape[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=False,
    )
    voxelizer = GaussianVoxelizer(voxel_settings)
    vol_pred, radii = voxelizer(
        means3D,
        density,
        scales,
        rotation,
    )
    
    vol_pred = vol_pred.cpu().squeeze().numpy()
    
    D = A / spacing

    # check if D is close to a pure permutation and flip of axes
    perm = np.argmax(np.abs(D), axis=0)   # shape (3,)

    if len(np.unique(perm)) != 3:
        raise ValueError("Affine includes oblique rotation/shear; transpose+flip is insufficient")

    aligned_score = np.abs(D[perm, np.arange(3)])
    if not np.allclose(aligned_score, 1.0, atol=1e-3):
        raise ValueError("Affine includes arbitrary rotation; need interpolation-based resampling")

    # permute the volume to align with the affine
    vol = np.transpose(vol_pred, axes=tuple(perm))

    # flip axes if the affine includes a flip
    signs = np.sign(D[perm, np.arange(3)])
    flip_axes = tuple(np.where(signs < 0)[0].tolist())
    if len(flip_axes) > 0:
        vol = np.flip(vol, axis=flip_axes)
    
    return vol

@dataclass
class XRaySaver(Saver):
    save_ckpt: bool = True
    save_vtp: bool = True
    save_nii: bool = True
    save_phase: float = 0.0
    def instantiate(self, *args, **kwargs) -> "XRaySaverModule":
        return XRaySaverModule(self)



class XRaySaverModule(SaverModule):
    @dataclass
    class VtpSavePayload:
        path: Path
        means: np.ndarray
        scales: np.ndarray
        rotations: np.ndarray
        density: np.ndarray


    @dataclass
    class NiftiSavePayload:
        volume_path: Path
        dxyz_volume_path: Path
        volume: np.ndarray
        dxyz_volume: np.ndarray
        volume_affine: np.ndarray
        dxyz_affine: np.ndarray


    @dataclass
    class SaveOutputsPayload:
        vtp: XRaySaverModule.VtpSavePayload | None = None
        nifti: XRaySaverModule.NiftiSavePayload | None = None
    
    def __init__(self, config: XRaySaver):
        super().__init__()
        self.config = config
        self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xray-save")
        self._pending_save: Future | None = None

    @staticmethod
    def _save_outputs(
        payload: SaveOutputsPayload,
    ) -> None:
        if payload.vtp is not None:
            pd = pv.PolyData(payload.vtp.means)
            pd.point_data["density"] = payload.vtp.density
            pd.point_data["scales"] = payload.vtp.scales
            pd.point_data["rotations"] = payload.vtp.rotations
            pd.field_data["model_type"] = np.array(["XrayCoronaryGaussian"])
            pd.save(payload.vtp.path)

        if payload.nifti is not None:
            nib_io.save(
                Nifti1Image(payload.nifti.volume, payload.nifti.volume_affine),
                str(payload.nifti.volume_path),
            )
            nib_io.save(
                Nifti1Image(payload.nifti.dxyz_volume, payload.nifti.dxyz_affine),
                str(payload.nifti.dxyz_volume_path),
            )

    def __del__(self):
        self._save_executor.shutdown(wait=False)
    
    def save(self, pl_module: GaussianSplatting):
        if pl_module.trainer.global_rank != 0:
            return

        if self.config.save_ckpt:
            self._save_ckpt(pl_module)
        
        pc = pl_module.gaussian_model
        
        renderer = cast(CoronaryDeformableXrayRenderer, pl_module.renderer)
        deform_model = renderer.deform_model
        
        means3D = pc.get_means().detach()
        density = pc.get_density().detach()
        rotation = pc.get_rotations().detach()
        scales = pc.get_scales().detach()
        
        # there is no contrast flow in simple generated rotating xray dataset, so we can just input the save_phase as the time 
        # input to get the deformation at that phase
        deforms: Deforms = deform_model(
            xyz = means3D.detach(), 
            t = torch.full((means3D.shape[0], 1), self.config.save_phase, device=means3D.device),   # input time is save_phase
            phase = torch.full((means3D.shape[0], 1), self.config.save_phase, device=means3D.device),   # input phase is save_phase
        )
        
        source_deformed = deform_model.deform(
            GSParam(xyz=means3D, rotation=rotation, scaling=scales, density=density), deforms
        )
        
        vtp_payload = self._make_vtp_save_payload(
            pl_module, 
            source_deformed
        ) if self.config.save_vtp else None

        nifti_payload = self._make_nii_save_payload(
            pl_module, 
            source_deformed
        ) if self.config.save_nii else None

        # save in threads
        if self._pending_save is not None and not self._pending_save.done():
            self._pending_save.result()

        payload = self.SaveOutputsPayload(vtp=vtp_payload, nifti=nifti_payload)
        self._pending_save = self._save_executor.submit(self._save_outputs, payload)
    
    
    def _save_ckpt(self, pl_module: GaussianSplatting):
        epoch = pl_module.trainer.current_epoch
        step = pl_module.trainer.global_step
        output_root = Path(pl_module.hparams["output_path"])
        ckpt_dir = output_root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"epoch={epoch}-step={step}.ckpt"
        
        pl_module.trainer.save_checkpoint(ckpt_path)
    
    
    def _make_nii_save_payload(
        self, 
        pl_module: GaussianSplatting,
        gs_param: GSParam,
    ) -> NiftiSavePayload:
        datamodule = pl_module.get_datamodule()
        meta = datamodule.dataparser.meta
        assert isinstance(meta, XRayMeta)
        volume_shape = tuple(meta.volume_size)
        coronary_affine = meta.centering_affine
        volume = gaussians_to_volume_by_Rasterizer(
            gs_param.xyz, gs_param.scaling, gs_param.rotation, gs_param.density, volume_shape, coronary_affine
        )
        dxyz_volume, zoomed_affine = deform_field_to_volume(
            pl_module.renderer.deform_model, volume_shape, coronary_affine, zoomed_shape=(128, 128, 128)
        )
        coronary_affine_save = np.array(coronary_affine, copy=True)
        torch.cuda.empty_cache()  # avoid CUDA OOM
        
        output_root = Path(pl_module.hparams["output_path"])
        epoch = pl_module.trainer.current_epoch
        step = pl_module.trainer.global_step

        volume_dir = output_root / "volumes"
        volume_dir.mkdir(parents=True, exist_ok=True)
        volume_nii_path = volume_dir / f"volume__epoch={epoch}-step={step}.nii.gz"
        dxyz_volume_nii_path = volume_dir / f"dxyz_volume__epoch={epoch}-step={step}.nii.gz"

        return self.NiftiSavePayload(
            volume_path=volume_nii_path,
            dxyz_volume_path=dxyz_volume_nii_path,
            volume=volume,
            dxyz_volume=dxyz_volume,
            volume_affine=coronary_affine_save,
            dxyz_affine=zoomed_affine,
        )
    
    
    def _make_vtp_save_payload(
        self,
        pl_module: GaussianSplatting,
        gs_param: GSParam,
    ) -> VtpSavePayload:
        
        output_root = Path(pl_module.hparams["output_path"])
        step = pl_module.trainer.global_step

        vtp_dir = output_root / "point_cloud"
        vtp_dir.mkdir(parents=True, exist_ok=True)
        vtp_path = vtp_dir / f"iteration_{step}.vtp"

        return self.VtpSavePayload(
            path=vtp_path,
            means=gs_param.xyz.detach().cpu().numpy().astype(np.float32),
            scales=gs_param.scaling.detach().cpu().numpy().astype(np.float32),
            rotations=gs_param.rotation.detach().cpu().numpy().astype(np.float32),
            density=gs_param.density.detach().cpu().numpy().astype(np.float32),
        )

        