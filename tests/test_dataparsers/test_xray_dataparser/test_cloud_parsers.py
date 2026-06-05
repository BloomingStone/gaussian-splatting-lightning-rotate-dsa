from pathlib import Path

import numpy as np
import pytest

from internal.dataparsers.xray_dataparser.cameras_builder import RotateXRayCamerasBuilder
from internal.dataparsers.xray_dataparser.cloud_parsers import (
    BallRandomCloudParser,
    CentralLineCloudParser,
    LabelCloudParser,
    RandomCloudParser,
    UniformCloudParser,
    get_AABB_corners,
)

from ..common import load_test_meta, save_point_cloud_and_cameras


def test_uniform_cloud_parser_creates_regular_grid(test_xray_data_root: Path, output_root: Path):
    meta = load_test_meta(test_xray_data_root)
    num_points=5**3
    point_cloud = UniformCloudParser(num_points=num_points).get_point_cloud(Path("unused"), meta)

    assert point_cloud.xyz.shape == (num_points, 3)
    corners = get_AABB_corners(meta.volume_size, meta.centering_affine)
    assert np.allclose(point_cloud.xyz.min(axis=0), corners.min(axis=0))
    assert np.allclose(point_cloud.xyz.max(axis=0), corners.max(axis=0))
    save_point_cloud_and_cameras(point_cloud, RotateXRayCamerasBuilder().build_cameras(meta), output_root / "uniform_cloud.png")


def test_random_cloud_parser_is_seeded(test_xray_data_root: Path, output_root: Path):
    meta = load_test_meta(test_xray_data_root)
    parser = RandomCloudParser(num_points=500, seed=7)

    first = parser.get_point_cloud(Path("unused"), meta)
    second = parser.get_point_cloud(Path("unused"), meta)

    assert np.allclose(first.xyz, second.xyz)
    corners = get_AABB_corners(meta.volume_size, meta.centering_affine)
    assert np.all(first.xyz >= corners.min(axis=0))
    assert np.all(first.xyz <= corners.max(axis=0))
    save_point_cloud_and_cameras(first, RotateXRayCamerasBuilder().build_cameras(meta), output_root / "random_cloud.png")


def test_ball_random_cloud_parser_stays_inside_radius(test_xray_data_root: Path, output_root: Path):
    meta = load_test_meta(test_xray_data_root)
    R=50.0
    parser = BallRandomCloudParser(num_points=500, R=R, seed=11)

    point_cloud = parser.get_point_cloud(Path("unused"), meta)

    assert point_cloud.xyz.shape == (500, 3)
    radii = np.linalg.norm(point_cloud.xyz, axis=1)
    assert np.all(radii <= R + 1e-6)

    save_point_cloud_and_cameras(point_cloud, RotateXRayCamerasBuilder().build_cameras(meta), output_root / "ball_random_cloud.png")

def test_label_cloud_parser_uses_label_mask_and_background(test_xray_data_root: Path, tmp_path: Path):
    nibabel = pytest.importorskip("nibabel")
    meta = load_test_meta(test_xray_data_root)
    label_data = np.zeros((6, 6, 6), dtype=np.uint8)
    label_data[1:3, 1:3, 1:3] = 1
    label_data[4, 4, 4] = 2
    nibabel.save(nibabel.Nifti1Image(label_data, affine=np.eye(4)), tmp_path / "coronary_label.nii.gz")

    parser = LabelCloudParser(num_points=500, label_value=1, add_random_background_points=False)
    point_cloud = parser.get_point_cloud(tmp_path, meta)

    assert point_cloud.xyz.shape[0] == 8

    parser_with_bg = LabelCloudParser(num_points=500, label_value=1, add_random_background_points=True)
    point_cloud_with_bg = parser_with_bg.get_point_cloud(tmp_path, meta)
    assert point_cloud_with_bg.xyz.shape[0] > point_cloud.xyz.shape[0]

def test_label_cloud_parser_real_data(test_xray_data_root: Path, output_root: Path):
    nibabel = pytest.importorskip("nibabel")
    meta = load_test_meta(test_xray_data_root)

    parser = LabelCloudParser(num_points=500, label_value=1, add_random_background_points=False)
    point_cloud = parser.get_point_cloud(test_xray_data_root, meta)

    assert point_cloud.xyz.shape[0] > 0

    save_point_cloud_and_cameras(point_cloud, RotateXRayCamerasBuilder().build_cameras(meta), output_root / "label_cloud.png")


def test_central_line_cloud_parser_reads_npz_and_background(test_xray_data_root: Path, tmp_path: Path):
    meta = load_test_meta(test_xray_data_root)
    central_line = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    np.savez(tmp_path / "central_line.npz", central_line)

    parser = CentralLineCloudParser(num_points=500, add_random_background_points=False)
    point_cloud = parser.get_point_cloud(tmp_path, meta)
    assert np.allclose(point_cloud.xyz, central_line)

    parser_with_bg = CentralLineCloudParser(num_points=500, add_random_background_points=True)
    point_cloud_with_bg = parser_with_bg.get_point_cloud(tmp_path, meta)
    assert point_cloud_with_bg.xyz.shape[0] > point_cloud.xyz.shape[0]


def test_central_line_cloud_parser_real_data(test_xray_data_root: Path, output_root: Path):
    meta = load_test_meta(test_xray_data_root)

    parser = CentralLineCloudParser(num_points=500, add_random_background_points=False)
    point_cloud = parser.get_point_cloud(test_xray_data_root, meta)

    assert point_cloud.xyz.shape[0] > 0

    save_point_cloud_and_cameras(point_cloud, RotateXRayCamerasBuilder().build_cameras(meta), output_root / "central_line_cloud.png")