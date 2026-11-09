"""Generate a random point-cloud initialization for r2_gaussian, matching
GS-contrast-flow stage1's RandomCloudParser (uniform random inside the volume
AABB) and its initial density.

Output format matches r2_gaussian's `init_*.npy`:
    [xyz_scaled(3), density(1)]   (xyz in scene-scaled units, density in
    scene-scaled attenuation units, matching r2_gaussian's training domain).

Run in the GS-contrast-flow environment.
"""
import argparse
import json

import numpy as np

SCENE_SCALE = 2.0 / 202.0
MU_WATER = 0.002  # mm^-1 (stage1 XrayCoronaryGaussian density_activation)


def main(args):
    meta = json.load(open(args.meta_json))
    ca = np.array(meta["lca_centering_affine"], dtype=np.float64)
    shape = np.array(meta["volume_size"], dtype=np.float64)

    # AABB corners in world (mm) coordinates.
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ]) * shape
    corners_h = np.hstack([corners, np.ones((8, 1))])
    corners_world = (corners_h @ ca.T)[:, :3]

    rng = np.random.default_rng(args.seed)
    xyz_mm = rng.random((args.n_points, 3)) * (
        corners_world.max(axis=0) - corners_world.min(axis=0)
    ) + corners_world.min(axis=0)

    # stage1 initial density: MU_WATER * softplus(1) in mm^-1.
    density_mm = MU_WATER * np.log1p(np.e)  # softplus(1) = ln(1 + e)
    density_scaled = density_mm / SCENE_SCALE

    xyz_scaled = xyz_mm * SCENE_SCALE
    out = np.concatenate([xyz_scaled, np.full((args.n_points, 1), density_scaled)], axis=1)
    np.save(args.output, out)
    print(f"saved {args.output}: {out.shape[0]} points, "
          f"xyz range {xyz_scaled.min(0)} .. {xyz_scaled.max(0)}, "
          f"density {density_scaled:.5f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meta_json", default="/media/F/sj/Code/GS-contrast-flow/data/example_grids/rotate_dsa.json")
    p.add_argument("--n_points", default=100000, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--output", default="/media/F/sj/Code/GS-contrast-flow/data/example_grids/init_random_example_grids.npy")
    args = p.parse_args()
    main(args)
