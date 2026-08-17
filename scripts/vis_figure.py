"""
Generate label isosurface visualization for Diseased_17 (LCA & RCA).
- Only label (segmented volume) isosurface comparison — no volume slices
- GT coronary AABB mask applied to all labels for clean visualization
"""

import os
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = "/media/data3/sj/Code/GS-dev-contrast-flow"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/vis")

BRANCHES = ["LCA", "RCA"]

METHODS = [
    ("Static GS", "StaticGS"),
    ("K-Planes", "KPlanes"),
    ("FDK", "FDK"),
    ("Deform-GS$_t$", "DeformGS_t"),
    ("Deform-GS$_\\phi$", "DeformGS_phi"),
    ("Deform-GS (t+$\\phi$)", "DeformGS_t_phi"),
    ("Ours (Flow-GS)", "FlowGS"),
]

DATA_DIRS = {
    "LCA": os.path.join(BASE_DIR, "data/gen_4d_output_all/flow/asoca-diseased__Diseased_17__LCA"),
    "RCA": os.path.join(BASE_DIR, "data/gen_4d_output_all/flow/asoca-diseased__Diseased_17__RCA"),
}

# Camera positions tuned via vis_camera.py for each branch
# [cam_position, focal_point, view_up]
CAM_POSITION = {
    "LCA": [
        (475.86768750317765, 250.29393304796548, 103.3320074385581),   # camera position
        (236.1754550609433, 178.44540274746743, -1.3641787468238986),   # focal point
        (-0.4482869625725806, 0.2413310815292358, 0.860696292704563),   # view up
    ],
    "RCA": [
        (-40.91710627519557, 140.2450152009345, -3.0019030553441235),   # camera position
        (182.5454142876626, 155.4984609701409, 6.2253833220258565),   # focal point
        (-0.03513773276097744, -0.08747089651827837, 0.9955471771838376),   # view up
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_gt_aabb(gt_volume: np.ndarray, pad: int = 10):
    """Compute axis-aligned bounding box from GT non-zero voxels."""
    coords = np.argwhere(gt_volume > 0.5)
    if len(coords) == 0:
        return None
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)
    return (
        max(0, x_min - pad), min(gt_volume.shape[0], x_max + pad),
        max(0, y_min - pad), min(gt_volume.shape[1], y_max + pad),
        max(0, z_min - pad), min(gt_volume.shape[2], z_max + pad),
    )


def apply_aabb_mask(volume: np.ndarray, aabb: tuple) -> np.ndarray:
    """Zero out everything outside the given AABB."""
    x_min, x_max, y_min, y_max, z_min, z_max = aabb
    masked = np.zeros_like(volume)
    masked[x_min:x_max, y_min:y_max, z_min:z_max] = \
        volume[x_min:x_max, y_min:y_max, z_min:z_max]
    return masked


def render_isosurface(volume: np.ndarray, color: str = "#c0392b",
                      spacing: tuple = (1.0, 1.0, 1.0),
                      origin: tuple = (0.0, 0.0, 0.0),
                      camera_position: list | None = None) -> np.ndarray | None:
    """Render a 3D isosurface using pyvista with data-centred camera."""
    try:
        pv.start_xvfb()
    except Exception:
        pass

    nx, ny, nz = volume.shape
    sx, sy, sz = spacing
    centre_world = np.array([
        nx / 2.0 * sx + origin[0],
        ny / 2.0 * sy + origin[1],
        nz / 2.0 * sz + origin[2],
    ])
    max_extent = max(nx * sx, ny * sy, nz * sz)

    # Build ImageData with padded volume for point_data
    grid = pv.ImageData()
    grid.dimensions = (nx + 1, ny + 1, nz + 1)
    grid.spacing = spacing
    grid.origin = origin
    padded = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.float32)
    padded[:nx, :ny, :nz] = volume
    grid.point_data["values"] = padded.ravel(order="F")

    try:
        mesh = grid.contour(isosurfaces=[0.5], scalars="values")
    except Exception:
        return None
    if mesh is None or mesh.n_points < 20:
        return None

    plotter = pv.Plotter(off_screen=True, window_size=[400, 400])
    plotter.set_background("white")
    plotter.add_mesh(mesh, color=color, smooth_shading=True, ambient=0.35)

    if camera_position is not None:
        plotter.camera_position = camera_position
    else:
        plotter.camera.position = centre_world + (max_extent, -max_extent, max_extent * 0.8)
        plotter.camera.focal_point = centre_world
        plotter.camera.up = (0, 0, 1)

    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_methods = len(METHODS)
    n_cols = n_methods + 1  # GT + 5 methods

    # Collect all renders: branch_render[branch]['gt'] and branch_render[branch][i]
    branch_render = {}

    for branch in BRANCHES:
        print(f"\n{'='*60}")
        print(f"Processing {branch}")
        print(f"{'='*60}")

        # Load GT label
        gt_path = os.path.join(DATA_DIRS[branch], "coronary_label.nii.gz")
        gt_nii = nib.load(gt_path)
        gt_vol = np.asanyarray(gt_nii.get_fdata(), dtype=np.float32)
        A = gt_nii.affine[:3, :3]
        spacing = tuple(abs(A[i, i]) for i in range(3))  # voxel size
        origin = tuple(gt_nii.affine[:3, 3])              # RAS origin
        print(f"  spacing={spacing}, origin=({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})")

        # Compute GT AABB
        aabb = get_gt_aabb(gt_vol, pad=10)
        print(f"  GT AABB x[{aabb[0]}:{aabb[1]}] y[{aabb[2]}:{aabb[3]}] z[{aabb[4]}:{aabb[5]}]")

        cam_pos = CAM_POSITION[branch]

        # Mask & render GT (with correct world-coordinate affine)
        gt_masked = apply_aabb_mask(gt_vol, aabb)
        gt_render = render_isosurface(gt_masked, color="#c0392b",
                                      spacing=spacing, origin=origin,
                                      camera_position=cam_pos)

        # Collect all method renders
        renders = []

        # Collect all method renders
        renders = []
        for disp_name, short_name in METHODS:
            version_dir = os.path.join(OUTPUT_DIR, "vis", f"{short_name}_{branch}")
            label_paths = sorted(glob.glob(
                os.path.join(version_dir, "volumes", "*label*thr*.nii.gz")
            ))
            if not label_paths:
                renders.append(None)
                print(f"  {disp_name}: NO LABEL FILE")
                continue
            label_vol = np.asanyarray(
                nib.load(label_paths[-1]).get_fdata(), dtype=np.float32
            )
            label_masked = apply_aabb_mask(label_vol, aabb)
            nnz = np.count_nonzero(label_masked)
            render_img = render_isosurface(label_masked, color="#c0392b",
                                           spacing=spacing, origin=origin,
                                           camera_position=cam_pos)
            renders.append(render_img)
            print(f"  {disp_name}: nnz={nnz}, render={'OK' if render_img is not None else 'FAILED'}")

        branch_render[branch] = {"gt": gt_render, "methods": renders}

    # ─── Save individual images ───
    out_dir = os.path.join(OUTPUT_DIR, "vis_figure")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Saving individual images to {out_dir}")
    print(f"{'='*60}")

    for branch in BRANCHES:
        # GT
        img = branch_render[branch]["gt"]
        if img is not None:
            path = os.path.join(out_dir, f"GT_{branch}.png")
            plt.imsave(path, img)
            print(f"  ✓ GT_{branch}.png")

        for i, (disp_name, _) in enumerate(METHODS):
            img = branch_render[branch]["methods"][i]
            if img is not None:
                safe_name = disp_name.replace("$", "").replace("\\", "").replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
                path = os.path.join(out_dir, f"{safe_name}_{branch}.png")
                plt.imsave(path, img)
                print(f"  ✓ {safe_name}_{branch}.png")

    print(f"\nDone! {len(os.listdir(out_dir))} images saved to {out_dir}")


if __name__ == "__main__":
    main()
