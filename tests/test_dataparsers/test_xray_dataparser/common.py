from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from internal.cameras import Cameras
from internal.dataparsers.dataparser import PointCloud
from internal.dataparsers.xray_dataparser.meta import XRayMeta, XRayMetaLoader

_MAX_CLOUD_POINTS = 100_000


def load_test_meta(data_root: Path) -> XRayMeta:
    return XRayMetaLoader().load(data_root)


def make_point_cloud(points: np.ndarray) -> PointCloud:
    features = np.ones_like(points, dtype=np.float64) * 127.0
    return PointCloud(xyz=points, feature=features)


def _camera_world_vectors(cameras: Cameras):
    """Return (forward, up, right) unit vectors in world coordinates.

    GS/COLMAP convention: X→right, Y→down, Z→forward.
    """
    device = cameras.R.device
    R_t = cameras.R.transpose(1, 2)
    forward = torch.matmul(R_t, torch.tensor([0.0, 0.0, 1.0], device=device))
    up = torch.matmul(R_t, torch.tensor([0.0, -1.0, 0.0], device=device))
    right = torch.matmul(R_t, torch.tensor([1.0, 0.0, 0.0], device=device))
    return (
        forward.detach().cpu().numpy(),
        up.detach().cpu().numpy(),
        right.detach().cpu().numpy(),
    )


def _compute_diag(pts: np.ndarray) -> float:
    """Return the diagonal (max Euclidean distance between any two axis-aligned extents)."""
    if pts.shape[0] == 0:
        return 0.0
    extent = pts.max(axis=0) - pts.min(axis=0)
    return max(float(np.linalg.norm(extent)), 1e-8)


def _draw_cloud(plotter: pv.Plotter, xyz: np.ndarray, diag: float) -> None:
    """Add point-cloud glyphs to *plotter*."""
    cloud_pd = pv.PolyData(xyz)
    plotter.add_points(cloud_pd, color="r", point_size=diag * 0.01, render_points_as_spheres=False, label="point cloud")


def _draw_cameras(
    plotter: pv.Plotter,
    centers: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    right: np.ndarray,
    diag: float,
) -> None:
    """Add camera-center spheres and orientation arrows to *plotter*."""
    arrow_len = max(diag * 0.05, 1.0)
    cam_pd = pv.PolyData(centers)
    plotter.add_points(cam_pd, color="red", point_size=diag * 0.001, render_points_as_spheres=True)
    plotter.add_arrows(centers, forward * arrow_len, color="red", label="forward")
    plotter.add_arrows(centers, up * arrow_len * 0.5, color="green", label="up")
    plotter.add_arrows(centers, right * arrow_len * 0.5, color="blue", label="right")


def _draw_bounds(plotter: pv.Plotter, pts: np.ndarray, color: str = "orange", label: str = "bounds") -> None:
    """Draw an axis-aligned bounding-box wireframe for *pts*."""
    if pts.shape[0] == 0:
        return
    bmin = pts.min(axis=0)
    bmax = pts.max(axis=0)
    box = pv.Box(bounds=[bmin[0], bmax[0], bmin[1], bmax[1], bmin[2], bmax[2]])
    plotter.add_mesh(box, color=color, style="wireframe", line_width=2, label=label)


def save_point_cloud(point_cloud: PointCloud, path: Path) -> None:
    """Save a standalone plot of the point cloud."""
    path.parent.mkdir(parents=True, exist_ok=True)

    xyz = point_cloud.xyz.astype(np.float32)
    if xyz.shape[0] > _MAX_CLOUD_POINTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(xyz.shape[0], _MAX_CLOUD_POINTS, replace=False)
        xyz = xyz[idx]

    if xyz.shape[0] == 0:
        return

    diag = _compute_diag(xyz)
    plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])
    plotter.add_title(path.stem, font_size=10)
    _draw_cloud(plotter, xyz, diag)
    plotter.show_grid()  # type: ignore[unused-ignore]
    plotter.add_legend()  # type: ignore[unused-ignore]
    plotter.screenshot(path, return_img=False)
    plotter.close()


def save_cameras(cameras: Cameras, path: Path) -> None:
    """Save a standalone plot of the camera rig."""
    path.parent.mkdir(parents=True, exist_ok=True)

    centers = cameras.camera_center.detach().cpu().numpy().astype(np.float32)
    forward, up, right = _camera_world_vectors(cameras)

    if centers.shape[0] == 0:
        return

    diag = _compute_diag(centers)
    plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])
    plotter.add_title(path.stem, font_size=10)
    _draw_cameras(plotter, centers, forward, up, right, diag)
    plotter.show_grid()  # type: ignore[unused-ignore]
    plotter.add_legend()  # type: ignore[unused-ignore]
    plotter.screenshot(path, return_img=False)
    plotter.close()


def save_point_cloud_and_cameras(point_cloud: PointCloud, cameras: Cameras, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- subsample cloud if too large ----------
    xyz = point_cloud.xyz.astype(np.float32)
    if xyz.shape[0] > _MAX_CLOUD_POINTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(xyz.shape[0], _MAX_CLOUD_POINTS, replace=False)
        xyz = xyz[idx]

    centers = cameras.camera_center.detach().cpu().numpy().astype(np.float32)
    forward, up, right = _camera_world_vectors(cameras)

    has_cloud = xyz.shape[0] > 0
    has_cameras = centers.shape[0] > 0

    # ---------- decide: single plot or two-panel ----------
    diag_cloud = _compute_diag(xyz)
    diag_cam = _compute_diag(centers)

    _SPLIT_RATIO = 2.0
    should_split = (
        has_cloud
        and has_cameras
        and max(diag_cloud, diag_cam) > _SPLIT_RATIO * min(diag_cloud, diag_cam)
    )

    # =====================================================================
    #  Single-panel (original behaviour) – when sizes are comparable
    # =====================================================================
    if not should_split:
        all_points = np.concatenate([xyz, centers], axis=0) if has_cameras else xyz
        diag = _compute_diag(all_points)

        plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])
        plotter.add_title(path.stem, font_size=10)

        if has_cloud:
            _draw_cloud(plotter, xyz, diag)

        if has_cameras:
            _draw_cameras(plotter, centers, forward, up, right, diag)

        plotter.show_grid()  # type: ignore[unused-ignore]
        plotter.add_legend()  # type: ignore[unused-ignore]
        plotter.screenshot(path, return_img=False)
        plotter.close()
        return

    # =====================================================================
    #  Two-panel layout – sizes differ by >2×
    #  Left  : large entity + small-entity bounds
    #  Right : small entity alone (zoomed in)
    # =====================================================================
    cloud_is_large = diag_cloud > diag_cam

    plotter = pv.Plotter(off_screen=True, window_size=[2048, 1024], shape=(1, 2))

    # --- Left subplot ---
    plotter.subplot(0, 0)
    if cloud_is_large:
        plotter.add_title(f"{path.stem}  (cloud + camera bounds)", font_size=10)
        _draw_cloud(plotter, xyz, diag_cloud)
        _draw_bounds(plotter, centers, color="orange", label="camera bounds")
    else:
        plotter.add_title(f"{path.stem}  (cameras + cloud bounds)", font_size=10)
        _draw_cameras(plotter, centers, forward, up, right, diag_cam)
        _draw_bounds(plotter, xyz, color="orange", label="cloud bounds")
    plotter.show_grid()  # type: ignore[unused-ignore]
    plotter.add_legend()  # type: ignore[unused-ignore]

    # --- Right subplot ---
    plotter.subplot(0, 1)
    if cloud_is_large:
        plotter.add_title(f"{path.stem}  (cameras only)", font_size=10)
        _draw_cameras(plotter, centers, forward, up, right, diag_cam)
    else:
        plotter.add_title(f"{path.stem}  (cloud only)", font_size=10)
        _draw_cloud(plotter, xyz, diag_cloud)
    plotter.show_grid()  # type: ignore[unused-ignore]
    plotter.add_legend()  # type: ignore[unused-ignore]

    plotter.screenshot(path, return_img=False)
    plotter.close()