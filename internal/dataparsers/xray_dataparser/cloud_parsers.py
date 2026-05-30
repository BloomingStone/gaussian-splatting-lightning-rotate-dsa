from typing import cast
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from ..dataparser import PointCloud, CloudParser
from .meta import XRayMeta


def get_AABB_corners(
    shape: np.ndarray,  # (3,) array of volume size in mm
    affine: np.ndarray,  # (4, 4) affine matrix of the volume
) -> np.ndarray:  # (8, 3) array of AABB corners in world coordinates
    """Get the 8 corners of the axis-aligned bounding box of the volume."""
    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]) * shape
    corners_homogeneous = np.hstack([corners, np.ones((8, 1))])  # (8, 4)
    corners_world = (affine @ corners_homogeneous.T).T[:, :3]  # (8, 3)
    return corners_world


def _get_random_ball_cloud(num_points: int, R: float, center: np.ndarray, seed: int) -> np.ndarray:
    """
    For uniform distribution in a ball, we need to use spherical coordinates and sample r, theta, phi. 
    The radius r should be sampled from the distribution that gives  uniform density in the ball. 
    This can be achieved by sampling  u uniformly from [0, 1] and then setting r = u^(1/3) * R, where R 
    is the radius of the ball. 
    
    Args:
        num_points: number of points to sample
        R: radius of the ball
        center: center of the ball
        seed: random seed for reproducibility
    Returns:
        xyz: (num_points, 3) array of point coordinates
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, num_points)
    costheta = rng.uniform(-1, 1, num_points)
    u = rng.uniform(0, 1, num_points)
    theta = np.arccos(costheta)
    r = R * (u ** (1/3))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack([x, y, z]) + center


def _get_random_backgound_cloud(xyz: np.ndarray, seed: int) -> np.ndarray:
    """Get random background points in a ball that covers the given xyz points."""
    center = xyz.mean(axis=0, keepdims=True)
    d = np.linalg.norm(xyz - center, axis=1)
    radius = d.max() * 1.2
    N = xyz.shape[0]
    return _get_random_ball_cloud(num_points=N, R=radius, center=center.squeeze(), seed=seed)
    


@dataclass
class UniformCloudParser(CloudParser):
    num_points: int

    def get_point_cloud(self, data_dir: Path, meta: XRayMeta) -> PointCloud:
        size = int(round(self.num_points ** (1/3)))
        bounds = get_AABB_corners(meta.volume_size, meta.centering_affine)
        x0, y0, z0 = bounds.min(axis=0)
        x1, y1, z1 = bounds.max(axis=0)
        axes = [np.linspace(x0, x1, size), np.linspace(y0, y1, size), np.linspace(z0, z1, size)]
        xyz = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        rgb = np.ones(xyz.shape) * 127
        return PointCloud(xyz=xyz, feature=rgb)


@dataclass
class RandomCloudParser(CloudParser):
    num_points: int
    seed: int = 42

    def get_point_cloud(self, data_dir: Path, meta: XRayMeta) -> PointCloud:
        rng = np.random.default_rng(self.seed)
        bounds = get_AABB_corners(meta.volume_size, meta.centering_affine)
        xyz = rng.random((self.num_points, 3)) * (bounds.max(axis=0) - bounds.min(axis=0)) + bounds.min(axis=0)
        rgb = np.ones(xyz.shape) * 127
        return PointCloud(xyz=xyz, feature=rgb)


@dataclass
class BallRandomCloudParser(CloudParser):
    num_points: int
    R: float|None = None  # if None, will be set to the minimum dimension of the bounding box of the volume
    seed: int = 42

    def get_point_cloud(self, data_dir: Path, meta: XRayMeta) -> PointCloud:
        if self.R is None:
            aabb = get_AABB_corners(meta.volume_size, meta.centering_affine)
            bounds_axis = aabb.max(axis=0) - aabb.min(axis=0)
            R: float = bounds_axis.min()
        else:
            R = self.R
        
        xyz = _get_random_ball_cloud(self.num_points, R, center=np.zeros(3), seed=self.seed)
        rgb = np.ones(xyz.shape) * 127
        return PointCloud(xyz=xyz, feature=rgb)


@dataclass
class LabelCloudParser(CloudParser):
    num_points: int
    label_nii_filename: str = "coronary_label.nii.gz"
    label_value: int | None = None  # if not None, only keep points with this label value in the label_nii
    seed: int = 42
    add_random_background_points: bool = True

    def get_point_cloud(self, data_dir: Path, meta: XRayMeta) -> PointCloud:
        import nibabel as nib
        
        affine = meta.centering_affine
        label_nii_path = data_dir / self.label_nii_filename
        nii_img = cast(nib.Nifti1Image, nib.load(label_nii_path))
        data = nii_img.get_fdata().astype(np.uint8)
        
        if self.label_value is not None:
            mask = (data == self.label_value)
        else:
            mask = (data > 0)
        
        idxs = np.argwhere(mask)
        if idxs.shape[0] == 0:
            raise ValueError(f"No points found in label nii with label_value={self.label_value}")
        
        A = affine[:3, :3]
        T = affine[:3, 3]
        xyz = (A @ idxs.T).T + T  # (N, 3) in world coordinates
        
        if self.add_random_background_points:
            xyz_background = _get_random_backgound_cloud(xyz, seed=self.seed)
            xyz = np.concatenate([xyz, xyz_background], axis=0)
        
        if idxs.shape[0] > self.num_points:
            rng = np.random.default_rng(self.seed)
            selected_idxs = rng.choice(idxs.shape[0], self.num_points, replace=False)
            xyz = xyz[selected_idxs]
        
        rgb = np.ones(xyz.shape) * 127
        return PointCloud(xyz=xyz, feature=rgb)


@dataclass
class CentralLineCloudParser(CloudParser):
    num_points: int
    central_line_filename: str = "central_line.npz"
    seed: int = 42
    add_random_background_points: bool = True

    def get_point_cloud(self, data_dir: Path, meta: XRayMeta) -> PointCloud:
        central_line_path = data_dir / self.central_line_filename
        xyz = np.load(central_line_path)["arr_0"]
        
        if self.add_random_background_points:
            xyz_background = _get_random_backgound_cloud(xyz, seed=self.seed)
            xyz = np.concatenate([xyz, xyz_background], axis=0)
        
        rgb = np.ones(xyz.shape) * 127
        return PointCloud(xyz=xyz, feature=rgb)