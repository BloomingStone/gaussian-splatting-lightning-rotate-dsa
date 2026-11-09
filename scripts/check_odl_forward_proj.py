"""ODL forward-projection validation: label nii -> ODL forward projection
-> compare with the 2D label PNG (phase 0 = frame 0).

Run in the GS-contrast-flow environment (has ODL).
"""
import argparse
import json
import os

import numpy as np
import nibabel as nib
from PIL import Image

from internal.dataparsers.xray_dataparser.conebeam import (
    ConeBeamParams,
    ConeBeamProjector,
    PngOdlTransform,
)


def main(args):
    meta = json.load(open(args.meta_json))
    ca = np.array(meta["lca_centering_affine"], dtype=np.float64)
    shape = tuple(int(x) for x in meta["volume_size"])
    c_arm = meta["c_arm_geometry"]
    frames = meta["frames"]

    # Label nii (x,y,z) NIfTI order.
    lab = nib.load(args.label).get_fdata().astype(np.float32)

    dso = float(c_arm["sod"])
    dde = float(c_arm["sdd"]) - dso
    dh = float(c_arm["dely"])
    dw = float(c_arm["delx"])

    os.makedirs(args.out_dir, exist_ok=True)

    # ODL requires >= 2 angles; project all frames and pick the requested ones.
    alphas = np.array(
        [float(f["alpha_degree"]) / 180.0 * np.pi + np.pi for f in frames],
        dtype=np.float64,
    )
    proj_size = (512, 512)
    param = ConeBeamParams.init_from(
        shape=shape,
        affine=ca,
        alphas=alphas,
        proj_size=proj_size,
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
    projs = projector.forward_proj(lab)  # (N, 512, 512) in PNG domain
    print(f"forward projections shape {projs.shape}")

    for fid in args.frame_ids:
        proj = projs[fid]
        gt = np.asarray(Image.open(os.path.join(args.label_dir, f"{fid:03d}.png")).convert("L")).astype(np.float32) / 255.0

        # Dice at a few thresholds.
        pn = proj - proj.min()
        pn = pn / (pn.max() + 1e-8)
        print(f"frame {fid}: forward-proj range [{proj.min():.3f},{proj.max():.3f}]")
        for thr in (0.1, 0.2, 0.3, 0.5):
            pm = pn > thr
            gm = gt > 0.5
            inter = (pm & gm).sum()
            d = 2 * inter / (pm.sum() + gm.sum()) if (pm.sum() + gm.sum()) else 0.0
            print(f"  thr {thr}: dice={d:.4f}")

        # Overlay
        rgb = np.stack([pn, pn, pn], -1) * 255
        gt_rgb = np.stack([gt, gt, gt], -1) * 255
        Image.fromarray((rgb * 0.5 + gt_rgb * 0.5).astype(np.uint8)).save(
            os.path.join(args.out_dir, f"odl_fwd_overlay_{fid:03d}.png")
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meta_json", default="/media/F/sj/Code/GS-contrast-flow/data/multi_layers/rbf_reader_flow_contrast_LCA/rotate_dsa.json")
    p.add_argument("--label", default="/media/F/sj/Code/GS-contrast-flow/data/multi_layers/rbf_reader_flow_contrast_LCA/coronary_label.nii.gz")
    p.add_argument("--label_dir", default="/media/F/sj/Code/GS-contrast-flow/data/multi_layers/rbf_reader_flow_contrast_LCA/label")
    p.add_argument("--frame_ids", nargs="+", type=int, default=[0, 40, 90, 179])
    p.add_argument("--out_dir", default="/media/F/sj/Code/r2_gaussian/output/lca_vis/odl_fwd")
    args = p.parse_args()
    main(args)
