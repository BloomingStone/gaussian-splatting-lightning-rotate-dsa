"""Voxelise the deform-applied point cloud (from export_deformed_pcd.py) to NIfTI.

The input pickle already contains ACTIVATED values in mm, so this script passes
them straight to `gaussians_to_volume_by_Rasterizer` (no re-activation).

Run in the GS-contrast-flow environment (has nibabel + the voxelizer).

Usage:
    pixi run python scripts/export_deformed_volume_nii.py \
        --pcd_pkl /media/F/sj/Code/r2_gaussian/output/lca_vis/deformed_pcd_phase0.5.pkl \
        --output /media/F/sj/Code/r2_gaussian/output/lca_vis/r2g_density_deformed_phase0.5.nii.gz
"""
import argparse
import json
import pickle

import numpy as np
import torch
import nibabel as nib
from nibabel.nifti1 import Nifti1Image

from internal.savers.x_ray_saver import gaussians_to_volume_by_Rasterizer


def main(args):
    with open(args.pcd_pkl, "rb") as f:
        d = pickle.load(f)

    xyz = torch.tensor(d["xyz_mm"], dtype=torch.float32, device="cuda")
    scale = torch.tensor(d["scale_mm"], dtype=torch.float32, device="cuda")
    rot = torch.tensor(d["rotation"], dtype=torch.float32, device="cuda")
    dens = torch.tensor(d["density"], dtype=torch.float32, device="cuda")

    meta = json.load(open(args.meta_json))
    ca = np.array(meta["lca_centering_affine"], dtype=np.float64)
    shape = tuple(int(x) for x in meta["volume_size"])

    print(f"n points: {xyz.shape[0]}")
    print("Voxelizing (mm space, affine permute/flip) ...")
    vol = gaussians_to_volume_by_Rasterizer(
        xyz, scale, rot, dens, shape, ca, to_cpu=True
    )
    print(
        f"volume: shape {vol.shape}, min {vol.min():.5f}, "
        f"max {vol.max():.5f}, mean {vol.mean():.5f}"
    )

    nib.save(Nifti1Image(vol.astype(np.float32), affine=ca), args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pcd_pkl",
        default="/media/F/sj/Code/r2_gaussian/output/lca_vis/deformed_pcd_phase0.5.pkl",
    )
    p.add_argument(
        "--meta_json",
        default="/media/F/sj/Code/GS-contrast-flow/data/multi_layers/rbf_reader_flow_contrast_LCA/rotate_dsa.json",
    )
    p.add_argument(
        "--output",
        default="/media/F/sj/Code/r2_gaussian/output/lca_vis/r2g_density_deformed_phase0.5.nii.gz",
    )
    args = p.parse_args()
    main(args)
