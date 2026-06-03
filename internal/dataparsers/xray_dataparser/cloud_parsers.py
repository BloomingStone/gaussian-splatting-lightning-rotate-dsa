from typing import cast
from pathlib import Path
from dataclasses import dataclass
import math

import numpy as np

from ..dataparser import PointCloud, CloudParser, Stage
from .meta import XRayMeta
from .conebeam import ConeBeamParams, PngOdlTransform, ConeBeamProjector

DEFAULT_NUM_POINTS = 100_000


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
    r = R * (np.power(u, 1/3))
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
class XRayCloudParser(CloudParser[XRayMeta]):
    """
    Base class for XRay point cloud parsers.  lightning jsonargparser's typing does not support Protocols, so we 
    use a base class instead of a Protocol for parsers.
    """
    pass    


@dataclass
class UniformCloudParser(XRayCloudParser):
    num_points: int = DEFAULT_NUM_POINTS

    def get_point_cloud(self, data_dir: Path, meta: XRayMeta, splits: None|dict[Stage, list[int]]=None) -> PointCloud:
        size = int(round(self.num_points ** (1/3)))
        bounds = get_AABB_corners(meta.volume_size, meta.centering_affine)
        x0, y0, z0 = bounds.min(axis=0)
        x1, y1, z1 = bounds.max(axis=0)
        axes = [np.linspace(x0, x1, size), np.linspace(y0, y1, size), np.linspace(z0, z1, size)]
        xyz = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        rgb = np.ones(xyz.shape) * 127
        return PointCloud(xyz=xyz, feature=rgb)


@dataclass
class RandomCloudParser(XRayCloudParser):
    num_points: int = DEFAULT_NUM_POINTS
    seed: int = 0

    def get_point_cloud(self, data_dir: Path, meta: XRayMeta, splits: None|dict[Stage, list[int]]=None) -> PointCloud:
        rng = np.random.default_rng(self.seed)
        bounds = get_AABB_corners(meta.volume_size, meta.centering_affine)
        xyz = rng.random((self.num_points, 3)) * (bounds.max(axis=0) - bounds.min(axis=0)) + bounds.min(axis=0)
        rgb = np.ones(xyz.shape) * 127
        return PointCloud(xyz=xyz, feature=rgb)


@dataclass
class BallRandomCloudParser(XRayCloudParser):
    num_points: int = DEFAULT_NUM_POINTS
    R: float|None = None  # if None, will be set to the minimum dimension of the bounding box of the volume
    seed: int = 0

    def get_point_cloud(self, data_dir: Path, meta: XRayMeta, splits: None|dict[Stage, list[int]]=None) -> PointCloud:
        if self.R is None:
            aabb = get_AABB_corners(meta.volume_size, meta.centering_affine)
            bounds_axis = aabb.max(axis=0) - aabb.min(axis=0)
            R: float = bounds_axis.min() / 2
        else:
            R = self.R
        
        xyz = _get_random_ball_cloud(self.num_points, R, center=np.zeros(3), seed=self.seed)
        rgb = np.ones(xyz.shape) * 127
        return PointCloud(xyz=xyz, feature=rgb)


@dataclass
class LabelCloudParser(XRayCloudParser):
    num_points: int = DEFAULT_NUM_POINTS
    label_value: int | None = None  # if not None, only keep points with this label value in the label_nii
    seed: int = 0
    add_random_background_points: bool = True

    def get_point_cloud(self, data_dir: Path, meta: XRayMeta, splits: None|dict[Stage, list[int]]=None) -> PointCloud:
        import nibabel as nib
        
        affine = meta.centering_affine
        assert meta.label_3d_info is not None, "label_3d_info must be provided in meta for LabelCloudParser"
        label_nii_path = data_dir / meta.label_3d_info.filename
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
class CentralLineCloudParser(XRayCloudParser):
    num_points: int = DEFAULT_NUM_POINTS
    central_line_filename: str = "central_line.npz"
    seed: int = 0
    add_random_background_points: bool = True

    def get_point_cloud(self, data_dir: Path, meta: XRayMeta, splits: None|dict[Stage, list[int]]=None) -> PointCloud:
        central_line_path = data_dir / self.central_line_filename
        xyz = np.load(central_line_path)["arr_0"]
        
        if self.add_random_background_points:
            xyz_background = _get_random_backgound_cloud(xyz, seed=self.seed)
            xyz = np.concatenate([xyz, xyz_background], axis=0)
        
        rgb = np.ones(xyz.shape) * 127
        return PointCloud(xyz=xyz, feature=rgb)


@dataclass
class FdkCloudParser(XRayCloudParser):
    num_points: int = DEFAULT_NUM_POINTS
    seed: int = 0
    use_filter: bool = True
    phase_min: float = 0.0
    phase_max: float = 0.5
    
    image_dir_name: str|None = "rotate_dsa"
    tiff_file_name: str|None = None  # if not None, will load the tiff file instead of individual png frames for FBP initialization
    
    def get_point_cloud(self, data_dir: Path, meta: XRayMeta, splits: None|dict[Stage, list[int]]=None) -> PointCloud:
        xyz, volume = self._init_point_cloud_from_fbp(data_dir, meta, splits)
        # TODO extract density from volume as feature, instead of using dummy constant feature
        feature = np.ones_like(xyz) * 127
        return PointCloud(xyz=xyz, feature=feature)

    @staticmethod
    def _load_png_projections(image_dir: Path, indices: list[int]) -> np.ndarray:
        from PIL import Image

        image_paths = sorted(image_dir.glob("*.png"))
        if len(image_paths) == 0:
            raise ValueError(f"No PNG files found in {image_dir}")

        return np.stack(
            [
                np.asarray(Image.open(image_paths[i]).convert("L"))
                for i in indices
            ],
            axis=0,
        )

    @staticmethod
    def _preprocess_indices_alphas(
        indices: list[int],
        meta: XRayMeta,
        phase_min: float,
        phase_max: float,
    ) -> tuple[list[int], list[float]]:
        assert phase_min >= 0.0 and phase_max <= 0.5 and phase_min <= phase_max, \
            f"Invalid phase range: [{phase_min}, {phase_max}]"
        
        alphas = meta.alphas_radians[indices].tolist()
        phases = meta.phase_array[indices].tolist()
        
        # DSA 中 alpha 角为从前向右转。在 RAS 坐标系中即旋转方向从 前（A +Y）转到右（R +X），即绕 Z 轴负向旋转。
        # degree 转 radian
        alphas = [-a for a in alphas]
        
        data = zip(indices, alphas, phases)
        
        # 收缩期和舒张期大致对称，因此也可以考虑使用对称的 phase 范围 [1-phase_max, 1] 来增加可用的视角数量
        y0 = 1 - phase_min
        y1 = 1 - phase_max
        
        phase_min_sym = min(y0, y1)
        phase_max_sym = max(y0, y1)
        
        data_selected = [
            (i, a, phi) for i, a, phi in data
            if (
                (phi <= phase_max + 1e-8 and phi >= phase_min - 1e-8) or
                (phi <= phase_max_sym + 1e-8 and phi >= phase_min_sym - 1e-8)
            )
        ]

        if len(data_selected) == 0:
            raise ValueError(f"No frames found in phase range [0, {phase_max}]")

        # ODL 仅接受单调递增的视角列表，因此根据 alpha 角对选定的帧进行排序
        data_sorted = sorted(data_selected, key=lambda d: float(d[1]))
        
        indices, alphas, phases = map(list, zip(*data_sorted))

        return indices, alphas


    def _init_point_cloud_from_fbp(self, data_dir: Path, meta: XRayMeta, splits: None|dict[Stage, list[int]]) -> tuple[np.ndarray, np.ndarray]:
        # FBP 初始化仅使用训练集的图像，防止可能的数据泄露造成的 valid/test 评估不公平。        
        indices, alphas = self._preprocess_indices_alphas(
            indices     =   splits["train"] if splits is not None else list(range(meta.num_frames)),
            meta        =   meta,
            phase_min   =   self.phase_min,
            phase_max   =   self.phase_max
        )

        if self.image_dir_name is not None:
            image_dir = data_dir / self.image_dir_name
            assert image_dir.exists(), f"Image directory {image_dir} does not exist"
            projections = self._load_png_projections(image_dir, indices)
        elif self.tiff_file_name is not None:
            import tifffile as tiff
            tiff_path = data_dir / self.tiff_file_name
            assert tiff_path.exists(), f"TIFF file {tiff_path} does not exist"
            projections = tiff.imread(tiff_path)
            projections = projections[indices]
        else:
            raise ValueError("Either image_dir_name or tiff_file_name must be provided for FdkCloudParser")
        
        
        affine = meta.centering_affine   # 注意这里使用 centering affine 来保证 FBP 初始化的点云与后续训练使用的坐标系一致
        geom = meta.c_arm_geometry
        projector = ConeBeamProjector(
            param = ConeBeamParams.init_from(
                shape       =   tuple(meta.volume_size),
                affine      =   affine,    
                alphas      =   np.asarray(alphas),
                proj_size   =   projections.shape[1:],
                dh          =   geom.dely,
                dw          =   geom.delx,
                dde         =   geom.sdd - geom.sod,
                dso         =   geom.sod,
            ), 
            img_transform = PngOdlTransform()
        )
        
        volume = projector.backward_proj(projections, use_filter=self.use_filter)
        xyz = self._sample_points_from_volume(
            volume=volume,
            affine=affine,
            num_points=self.num_points,
            seed=self.seed,
        )
        
        return xyz, volume

    @staticmethod
    def _sample_points_from_volume(
        volume: np.ndarray,
        affine: np.ndarray,
        num_points: int,
        seed: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        
        volume = volume.astype(np.float32)
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

        if volume.size == 0:
            raise ValueError("Cannot sample points from an empty volume")

        volume = volume - volume.min()
        
        threshold = np.percentile(volume, 75)
        mask = volume > threshold
        indices = np.argwhere(mask)
        weights = volume[mask]
        weights /= weights.sum()
        
        chosen = rng.choice(len(indices), size=num_points, p=weights)
        coords = indices[chosen]

        coords_h = np.concatenate([coords, np.ones((coords.shape[0], 1), dtype=np.float64)], axis=1)
        xyz = (coords_h @ np.asarray(affine, dtype=np.float64).T)[:, :3]
        return xyz
    