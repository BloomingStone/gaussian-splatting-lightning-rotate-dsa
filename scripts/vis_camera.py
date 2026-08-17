"""
Interactive GT label viewer — find the best camera angle.
Applies NIfTI affine (spacing + origin) for correct world-coordinate rendering.
Close the window to print the current camera parameters,
which you can then paste into vis_figure.py.
"""

import os
import numpy as np
import nibabel as nib
import pyvista as pv

BASE_DIR = "/media/data3/sj/Code/GS-dev-contrast-flow"
BRANCH = "RCA"   # change to "RCA" for the other branch
PAD = 10

# Load GT with affine
gt_path = os.path.join(
    BASE_DIR, "data/gen_4d_output_all/flow",
    f"asoca-diseased__Diseased_17__{BRANCH}", "coronary_label.nii.gz"
)
gt_nii = nib.load(gt_path)
gt_vol = np.asanyarray(gt_nii.get_fdata(), dtype=np.float32)

A = gt_nii.affine[:3, :3]
spacing = tuple(abs(A[i, i]) for i in range(3))   # voxel size
origin = tuple(gt_nii.affine[:3, 3])               # RAS origin

# AABB mask
coords = np.argwhere(gt_vol > 0.5)
x_min, y_min, z_min = coords.min(axis=0)
x_max, y_max, z_max = coords.max(axis=0)
x_min, x_max = max(0, x_min - PAD), min(gt_vol.shape[0], x_max + PAD)
y_min, y_max = max(0, y_min - PAD), min(gt_vol.shape[1], y_max + PAD)
z_min, z_max = max(0, z_min - PAD), min(gt_vol.shape[2], z_max + PAD)
masked = np.zeros_like(gt_vol)
masked[x_min:x_max, y_min:y_max, z_min:z_max] = gt_vol[x_min:x_max, y_min:y_max, z_min:z_max]

# Build ImageData with correct affine
nx, ny, nz = masked.shape
grid = pv.ImageData()
grid.dimensions = (nx + 1, ny + 1, nz + 1)
grid.spacing = spacing
grid.origin = origin
padded = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.float32)
padded[:nx, :ny, :nz] = masked
grid.point_data["values"] = padded.ravel(order="F")
mesh = grid.contour(isosurfaces=[0.5], scalars="values")

print(f"Mesh points: {mesh.n_points}")
print(f"Volume shape: {masked.shape}")
print(f"Spacing: {spacing}")
print(f"Origin:  {origin}")
print(f"AABB: x[{x_min}:{x_max}] y[{y_min}:{y_max}] z[{z_min}:{z_max}]")

# Interactive window
plotter = pv.Plotter(window_size=[800, 800])
plotter.set_background("white")
plotter.add_mesh(mesh, color="#c0392b", smooth_shading=True, ambient=0.3)

# Use tuned camera defaults
CAM_POSITION_DEFAULT = {
    "LCA": [
        (475.86768750317765, 250.29393304796548, 103.3320074385581),
        (236.1754550609433, 178.44540274746743, -1.3641787468238986),
        (-0.4482869625725806, 0.2413310815292358, 0.860696292704563),
    ],
    "RCA": [
        (-40.91710627519557, 140.2450152009345, -3.0019030553441235),
        (182.5454142876626, 155.4984609701409, 6.2253833220258565),
        (-0.03513773276097744, -0.08747089651827837, 0.9955471771838376),
    ],
}
plotter.camera_position = CAM_POSITION_DEFAULT[BRANCH]
plotter.camera.up = (0, 0, 1)

plotter.add_title(f"Diseased_17 {BRANCH} — Rotate to desired view, then close window", font_size=10)
plotter.show_axes()
plotter.show()

# ── On close, print camera config ──
print("\n" + "=" * 60)
print("Camera config — paste these into vis_figure.py:")
print("=" * 60)
cam = plotter.camera
pos = tuple(cam.position)
focal = tuple(cam.focal_point)
up = tuple(cam.up)
print(f'CAM_POSITION = [')
print(f'    {pos!r},   # camera position')
print(f'    {focal!r},   # focal point')
print(f'    {up!r},           # view up')
print(f']')
print(f"Focal distance: {np.linalg.norm(np.array(pos) - np.array(focal)):.1f}")
print("=" * 60)
