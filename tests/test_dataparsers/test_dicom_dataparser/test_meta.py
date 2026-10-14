from pathlib import Path

import numpy as np
import pytest

from internal.dataparsers.dicom_dataparser.meta import DicomMetaLoader

EXPECTED_FRAMES = 187
EXPECTED_HEIGHT = 634
EXPECTED_WIDTH = 862


def test_meta_loader_geometry(test_dicom_data_root: Path):
    meta = DicomMetaLoader().load(test_dicom_data_root)

    assert meta.num_frames == EXPECTED_FRAMES
    geometry = meta.c_arm_geometry
    assert geometry.sdd == pytest.approx(1199.99, abs=1e-2)
    assert geometry.sod == pytest.approx(760.0)
    assert (geometry.height, geometry.width) == (EXPECTED_HEIGHT, EXPECTED_WIDTH)
    assert geometry.orientation_type == "AP"
    assert meta.rotated_parameters.convention == "ZXY"
    assert meta.rotated_parameters.fps == pytest.approx(60.0)


def test_meta_loader_pixels(test_dicom_data_root: Path):
    meta = DicomMetaLoader().load(test_dicom_data_root)

    assert meta.pixel_data is not None
    assert meta.pixel_data.shape == (EXPECTED_FRAMES, EXPECTED_HEIGHT, EXPECTED_WIDTH)
    assert meta.pixel_data.dtype == np.float32
    assert 0.0 <= float(meta.pixel_data.min())
    assert float(meta.pixel_data.max()) <= 1.0
    assert float(meta.pixel_data.max()) == pytest.approx(1.0)
    assert meta.normalization == "minmax"
    assert meta.gamma is None
    assert meta.raw_pixel_max > meta.raw_pixel_min


def test_meta_loader_arrays(test_dicom_data_root: Path):
    meta = DicomMetaLoader().load(test_dicom_data_root)

    assert len(meta.alphas_radians) == EXPECTED_FRAMES
    assert len(meta.betas_radians) == EXPECTED_FRAMES
    assert len(meta.phase_array) == EXPECTED_FRAMES
    assert len(meta.time_array) == EXPECTED_FRAMES
    assert len(meta.frame_indices) == EXPECTED_FRAMES
    assert meta.frame_indices[0] == 0
    assert meta.frame_indices[-1] == EXPECTED_FRAMES - 1
    assert meta.frames[1].time_s == pytest.approx(1.0 / 60.0, abs=1e-9)


def test_meta_loader_default_volume(test_dicom_data_root: Path):
    meta = DicomMetaLoader().load(test_dicom_data_root)

    magnification = meta.c_arm_geometry.sod / meta.c_arm_geometry.sdd
    expected_transverse = EXPECTED_WIDTH * 0.462 * magnification
    expected_axial = EXPECTED_HEIGHT * 0.462 * magnification

    assert meta.volume_size == pytest.approx(
        np.array([expected_transverse, expected_transverse, expected_axial])
    )
    # Volume is centred on the iso-centre at the world origin.
    assert np.allclose(meta.centering_affine[:3, :3], np.eye(3))
    assert np.allclose(meta.centering_affine[:3, 3], -meta.volume_size / 2.0)


def test_meta_loader_volume_size_override(test_dicom_data_root: Path):
    meta = DicomMetaLoader(volume_size=(100.0, 200.0, 300.0)).load(test_dicom_data_root)

    assert np.allclose(meta.volume_size, [100.0, 200.0, 300.0])
    assert np.allclose(meta.centering_affine[:3, 3], [-50.0, -100.0, -150.0])


def test_meta_loader_flip_u(test_dicom_data_root: Path):
    flipped = DicomMetaLoader(flip_u=True).load(test_dicom_data_root)
    plain = DicomMetaLoader(flip_u=False).load(test_dicom_data_root)

    assert not np.allclose(flipped.pixel_data, plain.pixel_data)
    assert np.allclose(flipped.pixel_data[:, :, ::-1], plain.pixel_data)


def test_meta_loader_percentile_and_gamma(test_dicom_data_root: Path):
    meta = DicomMetaLoader(
        normalization="percentile",
        percentile=(0.05, 0.95),
        gamma_target=0.5,
    ).load(test_dicom_data_root)

    assert meta.normalization == "percentile"
    assert meta.gamma is not None
    assert 0.05 <= meta.gamma <= 1.0
    median = float(np.median(meta.pixel_data))
    # gamma was chosen so the pre-gamma median maps to 0.5.
    assert 0.3 < median < 0.7


def test_meta_loader_roi_folds_into_geometry(test_dicom_data_root: Path):
    roi = (100, 50, 400, 300)
    meta = DicomMetaLoader(roi=roi).load(test_dicom_data_root)
    rx0, ry0, rw, rh = roi

    assert meta.pixel_data.shape == (EXPECTED_FRAMES, rh, rw)
    geometry = meta.c_arm_geometry
    assert (geometry.height, geometry.width) == (rh, rw)
    # cx = x0 + width / 2 must equal the full-detector principal point shifted
    # by the crop origin.
    assert geometry.x0 + geometry.width / 2 == pytest.approx(EXPECTED_WIDTH / 2 - rx0)
    assert geometry.y0 + geometry.height / 2 == pytest.approx(EXPECTED_HEIGHT / 2 - ry0)
    assert meta.roi_offset == (rx0, ry0)
    assert meta.full_detector_size == (EXPECTED_HEIGHT, EXPECTED_WIDTH)


def test_meta_loader_roi_out_of_bounds(test_dicom_data_root: Path):
    with pytest.raises(ValueError):
        DicomMetaLoader(roi=(800, 0, 400, 100)).load(test_dicom_data_root)


def test_meta_loader_resolves_glob(test_dicom_data_root: Path):
    meta = DicomMetaLoader(dcm_filename="RXA_ch*.dcm").load(test_dicom_data_root)

    assert meta.dcm_path is not None
    assert meta.dcm_path.name == "RXA_chest.dcm"
