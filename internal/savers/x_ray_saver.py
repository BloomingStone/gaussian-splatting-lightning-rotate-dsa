from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch
from gsplat.exporter import export_splats
import nibabel as nib
from nibabel import loadsave as nib_io
from nibabel.nifti1 import Nifti1Image
from xray_gaussian_rasterization_voxelization import (
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

from . import Saver, SaverModule
from ..gaussian_splatting import GaussianSplatting
from ..mp_strategy import MPStrategy
from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel
from ..renderers.deformabel_xray_renderer import CoronaryDeformableXrayRenderer
from ..models.coronary_deform_model import DeformModel


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
    batch_size: int = 512*512*40,
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
        t_batch = torch.zeros((xyz_batch.shape[0], 1), device=device)  # input phase is 0.
        d_xyz, _, _ = deform_model(xyz_batch, t_batch)
        dxyz_chunks.append(d_xyz.cpu().float())

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
    save_ply: bool = True
    save_nii: bool = True
    def instantiate(self, *args, **kwargs) -> "XRaySaverModule":
        return XRaySaverModule(self)


@dataclass
class PlySavePayload:
    path: Path
    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    sh0: torch.Tensor


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
    ply: PlySavePayload | None = None
    nifti: NiftiSavePayload | None = None

class XRaySaverModule(SaverModule):
    def __init__(self, config: XRaySaver):
        super().__init__()
        self.config = config
        self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xray-save")
        self._pending_save: Future | None = None

    @staticmethod
    def _save_outputs(
        payload: SaveOutputsPayload,
    ) -> None:
        if payload.ply is not None:
            export_splats(
                means=payload.ply.means,
                scales=payload.ply.scales,
                quats=payload.ply.quats,
                opacities=payload.ply.opacities.squeeze(),
                sh0=payload.ply.sh0,
                shN=torch.zeros(
                    payload.ply.opacities.shape[0],
                    0,
                    3,
                    dtype=payload.ply.sh0.dtype,
                    device=payload.ply.sh0.device,
                ),
                save_to=str(payload.ply.path),
            )

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
        is_mp_strategy = isinstance(pl_module.trainer.strategy, MPStrategy)
        if pl_module.trainer.global_rank != 0 and not is_mp_strategy:
            return

        epoch = pl_module.trainer.current_epoch
        step = pl_module.trainer.global_step
        output_root = Path(pl_module.hparams["output_path"])
        
        ckpt_dir = output_root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        
        ckpt_suffix = f"-rank={pl_module.global_rank}" if is_mp_strategy else ""
        if self.config.save_ckpt:
            ckpt_path = ckpt_dir / f"epoch={epoch}-step={step}{ckpt_suffix}.ckpt"
            
            pl_module.trainer.save_checkpoint(ckpt_path)
        
        assert isinstance(pl_module.gaussian_model, XrayCoronaryGaussianModel)
        pc = pl_module.gaussian_model
        
        assert isinstance(pl_module.renderer, CoronaryDeformableXrayRenderer)
        deform_model = pl_module.renderer.deform_model
        
        means3D = pc.get_means().detach()
        density = pc.get_density().detach()
        rotation = pc.get_rotations().detach()
        scales = pc.get_scales().detach()
        
        d_xyz, d_scaling, d_rotation = deform_model(
            means3D.detach(), 
            torch.zeros(means3D.shape[0], 1).to(means3D.device),   # input phase is 0
        )
        
        means3D, rotation, scales = DeformModel.deform(
            means3D, rotation, scales, d_xyz, d_rotation, d_scaling
        )
        
        ply_payload: PlySavePayload | None = None

        if self.config.save_ply:
            ply_path = ckpt_dir / f"epoch={epoch}-step={step}{ckpt_suffix}.ply"
            gray = torch.exp(-density)
            sh0 = gray[..., None].repeat(1, 1, 3)
            ply_payload = PlySavePayload(
                path=ply_path,
                means=means3D.detach().cpu(),
                scales=pc.scale_inverse_activation(scales).detach().cpu(),
                quats=pc.scale_inverse_activation(rotation).detach().cpu(),
                opacities=gray.detach().cpu(),
                sh0=sh0.detach().cpu(),
            )

        nifti_payload: NiftiSavePayload | None = None

        if self.config.save_nii:
            volume_shape = pl_module.trainer.datamodule.dataparser.volume_shape     # type: ignore
            coronary_affine = pl_module.trainer.datamodule.dataparser.coronary_affine   # type: ignore
            volume = gaussians_to_volume_by_Rasterizer(
                means3D, scales, rotation, density, volume_shape, coronary_affine
            )
            dxyz_volume, zoomed_affine = deform_field_to_volume(deform_model, volume_shape, coronary_affine, zoomed_shape=(128, 128, 128))
            coronary_affine_save = np.array(coronary_affine, copy=True)
            torch.cuda.empty_cache()  # avoid CUDA OOM
            
            volume_nii_path = ckpt_dir / f"volume__epoch={epoch}-step={step}{ckpt_suffix}.nii.gz"
            dxyz_volume_nii_path = ckpt_dir / f"dxyz_volume__epoch={epoch}-step={step}{ckpt_suffix}.nii.gz"
            nifti_payload = NiftiSavePayload(
                volume_path=volume_nii_path,
                dxyz_volume_path=dxyz_volume_nii_path,
                volume=volume,
                dxyz_volume=dxyz_volume,
                volume_affine=coronary_affine_save,
                dxyz_affine=zoomed_affine,
            )

        if self._pending_save is not None and not self._pending_save.done():
            self._pending_save.result()

        payload = SaveOutputsPayload(ply=ply_payload, nifti=nifti_payload)
        self._pending_save = self._save_executor.submit(self._save_outputs, payload)
        