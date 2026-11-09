"""Export the R2-Gaussian reconstruction density volume as NIfTI, using the
exact same pipeline as GS-contrast-flow's `gaussians_to_volume_by_Rasterizer`
(voxelize + affine permute/flip) so it is directly comparable with the label.

Run in the GS-contrast-flow environment (has nibabel + the voxelizer).

Usage:
    pixi run python scripts/export_r2g_volume_nii.py \
        --ckpt_pickle /media/F/sj/Code/r2_gaussian/output/lca_last140_r2g_deform/point_cloud/iteration_30000/point_cloud.pickle \
        --output /media/F/sj/Code/r2_gaussian/output/lca_vis/r2g_density.nii.gz
"""

import argparse
import json
import pickle

import numpy as np
import torch
import nibabel as nib
from nibabel.nifti1 import Nifti1Image

from internal.savers.x_ray_saver import gaussians_to_volume_by_Rasterizer

SCENE_SCALE = 2.0 / 202.0  # same as r2_gaussian readLCAInfo (2 / max(sVoxel_mm))


def softplus(x):
    return torch.log(1.0 + torch.exp(x))


def main(args):
    with open(args.ckpt_pickle, "rb") as f:
        data = pickle.load(f)

    xyz = torch.tensor(data["xyz"], dtype=torch.float32, device="cuda")
    density_raw = torch.tensor(data["density"], dtype=torch.float32, device="cuda")
    scale_raw = torch.tensor(data["scale"], dtype=torch.float32, device="cuda")
    rotation_raw = torch.tensor(data["rotation"], dtype=torch.float32, device="cuda")
    scale_bound = data.get("scale_bound", None)
    print(f"n points: {xyz.shape[0]}, scale_bound: {scale_bound}")

    # Replicate R2-Gaussian activations.
    density = softplus(density_raw)
    if scale_bound is not None:
        lo, hi = float(scale_bound[0]), float(scale_bound[1])
        scale = torch.sigmoid(scale_raw) * (hi - lo) + lo
    else:
        scale = torch.exp(scale_raw)
    rotation = torch.nn.functional.normalize(rotation_raw, dim=-1)

    # R2-Gaussian gaussians are in scene-scaled units; convert to mm to match
    # the centering affine used by gaussians_to_volume_by_Rasterizer.
    xyz_mm = xyz / SCENE_SCALE
    scale_mm = scale / SCENE_SCALE

    # Centering affine (same as the label's NIfTI affine).
    meta = json.load(open(args.meta_json))
    centering_affine = np.array(meta["lca_centering_affine"], dtype=np.float64)
    shape = tuple(int(x) for x in meta["volume_size"])

    print("Voxelizing (mm space, with affine permute/flip) ...")
    vol = gaussians_to_volume_by_Rasterizer(
        xyz_mm, scale_mm, rotation, density, shape, centering_affine, to_cpu=True
    )
    print(f"volume: shape {vol.shape}, min {vol.min():.5f}, max {vol.max():.5f}, mean {vol.mean():.5f}")

    nifti = Nifti1Image(vol.astype(np.float32), affine=centering_affine)
    nib.save(nifti, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_pickle", default="/media/F/sj/Code/r2_gaussian/output/lca_last140_r2g_deform/point_cloud/iteration_30000/point_cloud.pickle", type=str)
    parser.add_argument("--meta_json", default="/media/F/sj/Code/GS-contrast-flow/data/multi_layers/rbf_reader_flow_contrast_LCA/rotate_dsa.json", type=str)
    parser.add_argument("--output", default="/media/F/sj/Code/r2_gaussian/output/lca_vis/r2g_density.nii.gz", type=str)
    args = parser.parse_args()
    main(args)
