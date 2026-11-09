"""Generate an ODL-based FDK initialization point cloud for r2_gaussian.

Replicates GS-contrast-flow's FdkCloudParser geometry (alpha + pi, centering
affine) but reconstructs the I-domain line integral (J -> I = -ln J), so the
density values match r2_gaussian's training domain.

Output format matches r2_gaussian's `init_*.npy`:
    [xyz_scaled(3), density(1)]   (xyz in scene-scaled units, density > 0)

Run in the GS-contrast-flow environment.
"""
import argparse
import json
import os.path as osp

import numpy as np
from PIL import Image

from internal.dataparsers.xray_dataparser.conebeam import (
    ConeBeamParams,
    ConeBeamProjector,
    PngOdlTransform,
)

SCENE_SCALE = 2.0 / 202.0


def main(args):
    meta = json.load(open(args.meta_json))
    ca = np.array(meta["lca_centering_affine"], dtype=np.float64)
    shape = tuple(int(x) for x in meta["volume_size"])
    c = meta["c_arm_geometry"]
    frames = meta["frames"]
    total = len(frames)
    sel = frames[total - args.last_n_frames:]

    dso = float(c["sod"])
    dde = float(c["sdd"]) - dso
    dh = float(c["dely"])
    dw = float(c["delx"])
    alphas = np.array(
        [float(f["alpha_degree"]) / 180.0 * np.pi + np.pi for f in sel],
        dtype=np.float64,
    )

    # Load I-domain projections (matches r2_gaussian readLCAInfo).
    projs = []
    for f in sel:
        fi = int(f["frame"])
        J = np.asarray(
            Image.open(osp.join(args.image_dir, f"{fi:03d}.png")).convert("L"),
            dtype=np.float64,
        ) / 255.0
        J = np.clip(J, 1e-3, 1.0)
        I = -np.log(J)
        projs.append((I * SCENE_SCALE * args.proj_scale).astype(np.float32))
    projs = np.stack(projs)  # (N, 512, 512)
    print(f"projections {projs.shape}, mean {projs.mean():.5f}")

    param = ConeBeamParams.init_from(
        shape=shape,
        affine=ca,
        alphas=alphas,
        proj_size=(512, 512),
        dh=dh,
        dw=dw,
        dde=dde,
        dso=dso,
    )
    projector = ConeBeamProjector(
        param=param,
        img_transform=PngOdlTransform(reverse_gray=False),
        align_ras=True,
    )

    print("ODL FDK reconstruction ...")
    volume = projector.backward_proj(projs, use_filter=True)  # NIfTI order
    print(
        f"volume {volume.shape}, min {volume.min():.5f}, max {volume.max():.5f}, "
        f"mean {volume.mean():.5f}"
    )

    # Sample points above a percentile threshold (positive density).
    thr = np.percentile(volume, args.density_percentile)
    mask = volume > thr
    indices = np.argwhere(mask)
    print(
        f"threshold {thr:.5f} ({args.density_percentile}%), "
        f"voxels above: {len(indices)}"
    )
    assert len(indices) >= args.n_points, "not enough voxels; lower density_percentile"

    rng = np.random.default_rng(args.seed)
    chosen = indices[rng.choice(len(indices), args.n_points, replace=True)]
    # ODL reconstructs the attenuation coefficient in mm^-1; r2_gaussian works
    # in scene-scaled units, so density must be divided by scene_scale.
    density = (
        volume[chosen[:, 0], chosen[:, 1], chosen[:, 2]] / SCENE_SCALE
    )

    # voxel -> world mm via centering affine, then -> scene-scaled units.
    coords_h = np.concatenate([chosen, np.ones((chosen.shape[0], 1))], axis=1)
    world_mm = (coords_h @ ca.T)[:, :3]
    xyz_scaled = world_mm * SCENE_SCALE

    out = np.concatenate([xyz_scaled, density[:, None]], axis=1)
    np.save(args.output, out)
    print(
        f"saved {args.output} ({out.shape[0]} points), "
        f"xyz range {xyz_scaled.min(0)} .. {xyz_scaled.max(0)}, "
        f"density mean {density.mean():.5f}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meta_json", default="/media/F/sj/Code/GS-contrast-flow/data/multi_layers/rbf_reader_flow_contrast_LCA/rotate_dsa.json")
    p.add_argument("--image_dir", default="/media/F/sj/Code/GS-contrast-flow/data/multi_layers/rbf_reader_flow_contrast_LCA/rotate_dsa")
    p.add_argument("--last_n_frames", default=140, type=int)
    p.add_argument("--proj_scale", default=9.0, type=float)
    p.add_argument("--n_points", default=100000, type=int)
    p.add_argument("--density_thresh", default=0.005, type=float)
    p.add_argument("--density_percentile", default=90.0, type=float)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--output", default="/media/F/sj/Code/GS-contrast-flow/data/multi_layers/rbf_reader_flow_contrast_LCA/init_odl_rbf_reader_flow_contrast_LCA.npy")
    args = p.parse_args()
    main(args)
