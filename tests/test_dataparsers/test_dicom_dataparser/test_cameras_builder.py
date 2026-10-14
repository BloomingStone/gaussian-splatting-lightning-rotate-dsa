from pathlib import Path

import torch
import pytest

from internal.dataparsers.dicom_dataparser.cameras_builder import DicomCamerasBuilder
from internal.dataparsers.dicom_dataparser.meta import DicomMetaLoader
from internal.dataparsers.xray_dataparser.cameras_builder import RotateXRayCamerasBuilder


@pytest.fixture
def chest_meta(test_dicom_data_root: Path):
    return DicomMetaLoader().load(test_dicom_data_root)


def test_camera_count_and_intrinsics(chest_meta):
    cameras = DicomCamerasBuilder().build_cameras(chest_meta)
    geometry = chest_meta.c_arm_geometry

    assert len(cameras) == chest_meta.num_frames
    assert float(cameras.fx[0]) == pytest.approx(geometry.sdd / geometry.delx, rel=1e-5)
    assert float(cameras.fy[0]) == pytest.approx(geometry.sdd / geometry.dely, rel=1e-5)
    # No ROI -> principal point at the image centre.
    assert float(cameras.cx[0]) == pytest.approx(geometry.width / 2)
    assert float(cameras.cy[0]) == pytest.approx(geometry.height / 2)


def test_camera_fov_matches_focal_length(chest_meta):
    cameras = DicomCamerasBuilder().build_cameras(chest_meta)
    geometry = chest_meta.c_arm_geometry

    expected_fov_x = 2.0 * torch.atan(torch.tensor(geometry.width / 2.0) / cameras.fx[0])
    assert float(cameras.fov_x[0]) == pytest.approx(float(expected_fov_x), rel=1e-6)


def test_camera_center_is_sod_from_isocentre(chest_meta):
    cameras = DicomCamerasBuilder().build_cameras(chest_meta)

    distances = torch.linalg.norm(cameras.camera_center, dim=-1)
    assert torch.allclose(distances, torch.full_like(distances, chest_meta.c_arm_geometry.sod), atol=1e-2)


def test_camera_rotations_are_orthonormal(chest_meta):
    cameras = DicomCamerasBuilder().build_cameras(chest_meta)

    identity = torch.eye(3).expand(len(cameras), 3, 3)
    assert torch.allclose(cameras.R @ cameras.R.transpose(1, 2), identity, atol=1e-5)


def test_default_matches_xray_angle_model(chest_meta):
    """With `use_projection_matrix=False` the builder must reproduce the xray
    ideal-circle camera model exactly."""
    dicom_cameras = DicomCamerasBuilder().build_cameras(chest_meta)
    xray_cameras = RotateXRayCamerasBuilder().build_cameras(chest_meta)

    assert torch.allclose(dicom_cameras.R, xray_cameras.R, atol=1e-5)
    assert torch.allclose(dicom_cameras.T, xray_cameras.T, atol=1e-3)
    assert torch.allclose(dicom_cameras.camera_center, xray_cameras.camera_center, atol=1e-2)
    assert torch.allclose(dicom_cameras.fx, xray_cameras.fx, atol=1e-3)


def test_projection_matrix_cameras_are_plausible(chest_meta):
    """The vendor-matrix path must produce a well-formed camera per frame."""
    assert chest_meta.projection_matrices is not None
    cameras = DicomCamerasBuilder(use_projection_matrix=True).build_cameras(chest_meta)

    assert len(cameras) == chest_meta.num_frames
    assert torch.all(torch.isfinite(cameras.camera_center))
    assert torch.all(torch.isfinite(cameras.R))
    assert torch.all(cameras.fx > 0)
    assert torch.all(cameras.fy > 0)

    identity = torch.eye(3).expand(len(cameras), 3, 3)
    assert torch.allclose(cameras.R @ cameras.R.transpose(1, 2), identity, atol=1e-4)

    # The source should still sit roughly a SOD away from the iso-centre.
    distances = torch.linalg.norm(cameras.camera_center, dim=-1)
    assert torch.allclose(
        distances, torch.full_like(distances, chest_meta.c_arm_geometry.sod), atol=25.0
    )
