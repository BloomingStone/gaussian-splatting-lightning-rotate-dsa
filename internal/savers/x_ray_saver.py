from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from gsplat.exporter import export_splats
import nibabel as nib

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


def gaussians_to_volume_by_batch(
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotation: torch.Tensor,
    density: torch.Tensor,
    shape: tuple[int, int, int],
    gaussian_batch: int = 8,
    small_filter_threshold: None|float = 0.002      # mu = 0.0002 is mu of water
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Convert the density Gaussian point cloud to 3D voxels. 
    First, the affine transformation matrix of volume will be calculated based on 
    the spatial distribution of the Goth points. The value of each grid point is 
    the accumulation of the densities of all Gauss at this point
    
    Args: 
        means3D: (N, 3)
        scales: (N, 3)
        rotation: (N, 4)
        density: (N, 1)
        shape: (3,)
        gaussian_batch: int = 512, the number of Gaussians to be processed in a batch. 
            It can be adjusted according to the GPU memory.
        small_filter_threshold: None|float = 0.002, density threshold to filter small density gaussian.
    Returns: 
        volume: (shape[0], shape[1], shape[2]) 
        affine: (4, 4) 
        size: (3, 3): size of the bounding box 
    """
    device = means3D.device
    D, H, W = shape
    
    with torch.no_grad():
        # ---------- felter small density gaussian ----------
        if small_filter_threshold is not None:
            mask = (density > small_filter_threshold).squeeze()
            means3D = means3D[mask]
            scales = scales[mask]
            rotation = rotation[mask]
            density = density[mask]

        # ---------- bounding box ----------
        extent = 3.0 * scales.max(dim=1)[0].unsqueeze(1)
        mins = (means3D - extent).min(dim=0)[0]
        maxs = (means3D + extent).max(dim=0)[0]

        size = maxs - mins
        spacing = size / torch.tensor([D, H, W], device=device)

        # ---------- affine ----------
        affine = torch.eye(4, device=device)
        affine[0,0] = spacing[0]
        affine[1,1] = spacing[1]
        affine[2,2] = spacing[2]
        affine[:3,3] = mins

        # ---------- build world grid ----------
        xs = torch.arange(D, device=device)
        ys = torch.arange(H, device=device)
        zs = torch.arange(W, device=device)

        grid = torch.stack(
            torch.meshgrid(xs, ys, zs, indexing="ij"),
            dim=-1
        ).reshape(-1,3).float()

        world = (grid * spacing + mins).half()   # (V,3)
        V = world.shape[0]

        volume = torch.zeros(V, device=device)

        # ---------- precompute Σ^-1 ----------
        R = quaternion_to_matrix(rotation)
        Sigma_inv = torch.inverse(
            (R @ torch.diag_embed(scales**2) @ R.transpose(1,2)).float()
        ).half()  # (N,3,3)

        N = means3D.shape[0]

        # ---------- sum gaussian by batch ----------
        # \pho(x) = \sum_i \pho_i exp(-0.5 (x - \mu_i)^T \Sigma_i^{-1} (x - \mu_i))
        for start in range(0, N, gaussian_batch):
            end = min(start + gaussian_batch, N)

            mu = means3D[start:end]        # (B,3)
            inv = Sigma_inv[start:end]     # (B,3,3)  \Sigma_i^{-1}
            rho = density[start:end]       # (B,1)

            # (V,1,3) - (1,B,3)
            diff = (world.unsqueeze(1) - mu.unsqueeze(0))   # (V,B,3)  (x - \mu_i)

            # quadratic form
            exponent = -0.5 * torch.einsum(
                "vbi,bij,vbj->vb",
                diff,
                inv,
                diff
            )

            val = rho.squeeze(1).float() * torch.exp(exponent)  # (V,B)

            volume += val.sum(dim=1) # (V,)

        volume = volume.reshape(D, H, W)

        return volume.cpu().numpy(), affine.cpu().numpy(), size.cpu().numpy()


def gaussians_to_volume(
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotation: torch.Tensor,
    density: torch.Tensor,
    shape: tuple[int, int, int],
    small_filter_threshold: None|float = 0.002
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 
    Convert the density Gaussian point cloud to 3D voxels. 
    First, the affine transformation matrix of volume will be calculated based on 
    the spatial distribution of the Goth points. The value of each grid point is 
    the accumulation of the densities of all Gauss at this point
    
    Args: 
        means3D: (N, 3)
        scales: (N, 3)
        rotation: (N, 4)
        density: (N, 1)
        shape: (3,) 
            It can be adjusted according to the GPU memory.
        small_filter_threshold: None|float = 0.002, density threshold to filter small density gaussian.
    Returns: 
        volume: (shape[0], shape[1], shape[2]) 
        affine: (4, 4) 
        size: (3, 3): size of the bounding box 
    """
    device = means3D.device
    D, H, W = shape
    
    # ---------- felter small density gaussian ----------
    with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
        if small_filter_threshold is not None:
            mask = (density > small_filter_threshold).squeeze()
            means3D = means3D[mask]
            scales = scales[mask]
            rotation = rotation[mask]
            density = density[mask]
        
        means3D = means3D.detach().float()
        scales = scales.detach().float()
        rotation = rotation.detach().float()
        density = density.detach().float()

        # ---------- bounding box ----------
        extent = 3.0 * scales.max(dim=1)[0].unsqueeze(1)
        mins = (means3D - extent).min(dim=0).values
        maxs = (means3D + extent).max(dim=0).values
        
        bounding_min = torch.quantile(mins, 0.01)
        bounding_max = torch.quantile(maxs, 0.99)

        size = bounding_max - bounding_min
        spacing = size / torch.tensor([D, H, W], device=device)

        # ---------- affine & volume ----------
        affine = torch.eye(4, device=device)
        affine[0,0] = spacing[0]
        affine[1,1] = spacing[1]
        affine[2,2] = spacing[2]
        affine[:3,3] = mins

        volume = torch.zeros(shape, device=device)

        # ---------- precompute Σ^-1 ----------
        R = quaternion_to_matrix(rotation).float()
        Sigma_inv = torch.inverse(
            (R @ torch.diag_embed(scales**2).float() @ R.transpose(1,2)).float()
        )  # (N,3,3)

        # ---------- sum gaussian by batch ----------
        # \pho(x) = \sum_i \pho_i exp(-0.5 (x - \mu_i)^T \Sigma_i^{-1} (x - \mu_i))
        for i in range(means3D.shape[0]):

            mu = means3D[i]
            sigma = scales[i]
            rho = density[i, 0]
            inv = Sigma_inv[i]
            
            # 只在 3σ bounding box 内计算
            sigma_max = torch.abs(sigma).max()
            local_min = mu - 3 * sigma_max
            local_max = mu + 3 * sigma_max
            
            # 转为 voxel index
            vmin = ((local_min - mins) / spacing).long()
            vmax = ((local_max - mins) / spacing).long()

            lower = torch.zeros(3, device=device)
            upper = torch.tensor([D-1, H-1, W-1], device=device)

            vmin = torch.clamp(vmin, min=lower, max=upper).tolist()
            vmax = torch.clamp(vmax, min=lower, max=upper).tolist()

            xs = torch.arange(vmin[0], vmax[0]+1, device=device)
            ys = torch.arange(vmin[1], vmax[1]+1, device=device)
            zs = torch.arange(vmin[2], vmax[2]+1, device=device)

            grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
            grid = grid.reshape(-1, 3).float()

            # 转为 world 坐标
            world = grid * spacing + mins

            diff = world - mu
            exponent = -0.5 * torch.sum(diff @ inv * diff, dim=1)

            val = rho * torch.exp(exponent)

            volume[
                grid[:, 0].long(),
                grid[:, 1].long(),
                grid[:, 2].long()
            ] += val
            
            if not torch.isfinite(volume).all():
                raise FloatingPointError(f"val contains inf or nan")
            

        volume = volume.reshape(D, H, W)

        return volume.cpu().numpy(), affine.cpu().numpy(), size.cpu().numpy()


@dataclass
class XRaySaver(Saver):
    def instantiate(self, *args, **kwargs) -> "XRaySaverModule":
        return XRaySaverModule(self)

class XRaySaverModule(SaverModule):
    def __init__(self, config: XRaySaver):
        super().__init__()
        self.config = config
    
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
            torch.zeros(means3D.shape[0], 1).to(means3D.device)
        )
        
        means3D, rotation, scales = DeformModel.deform(
            means3D, rotation, scales, d_xyz, d_rotation, d_scaling
        )
        
        ply_path = ckpt_dir / f"epoch={epoch}-step={step}{ckpt_suffix}.ply"

        gray = torch.exp( - density)
        sh0 = gray[..., None].repeat(1, 1, 3)
        export_splats(
            means=means3D,
            scales=pc.scale_inverse_activation(scales),
            quats=pc.scale_inverse_activation(rotation),
            opacities=gray.squeeze(),
            sh0=sh0,
            shN=torch.zeros(gray.shape[0], 0, 3).to(sh0),
            save_to=str(ply_path)
        )
        
        volume, affine, size = gaussians_to_volume(
            means3D, scales, rotation, density, (256, 256, 256)
        )
        torch.cuda.empty_cache()  # avoid CUDA OOM
        
        nii_path = ckpt_dir / f"volume__epoch={epoch}-step={step}{ckpt_suffix}.nii.gz"
        nib.loadsave.save(nib.nifti1.Nifti1Image(volume, affine), str(nii_path))
        