from pathlib import Path

import numpy as np
import pytest

from internal.dataparsers.dicom_dataparser.dicom_reader import (
    DicomReadError,
    extract_pixels,
    normalize_01,
    normalize_01_percentile,
    parse_dicom_header,
)

EXPECTED_FRAMES = 187
EXPECTED_HEIGHT = 634
EXPECTED_WIDTH = 862


def test_parse_header_geometry(test_dicom_chest_file: Path):
    header = parse_dicom_header(test_dicom_chest_file)

    assert header.num_frames == EXPECTED_FRAMES
    assert header.shape == (EXPECTED_FRAMES, EXPECTED_HEIGHT, EXPECTED_WIDTH)
    assert header.sdd == pytest.approx(1199.99, abs=1e-2)
    assert header.sod == pytest.approx(760.0, abs=1e-6)
    assert header.delx == pytest.approx(0.462, abs=1e-9)
    assert header.dely == pytest.approx(0.462, abs=1e-9)
    assert header.fps == pytest.approx(60.0, abs=1e-6)


def test_parse_header_frames(test_dicom_chest_file: Path):
    header = parse_dicom_header(test_dicom_chest_file)

    assert len(header.alphas_degree) == EXPECTED_FRAMES
    assert header.alphas_degree[0] == pytest.approx(-99.42, abs=1e-6)
    assert header.alphas_degree[-1] == pytest.approx(101.26, abs=1e-6)
    assert np.all(np.diff(header.alphas_degree) > 0.0)
    assert np.allclose(header.betas_degree, 0.0)

    # FrameTimeVector is degenerate in this file -> uniform i / fps fallback.
    assert header.times_s[0] == pytest.approx(0.0)
    assert header.times_s[1] == pytest.approx(1.0 / 60.0, abs=1e-9)
    assert np.all(np.diff(header.times_s) > 0.0)

    # No private phase tag in this file -> all-zero phases.
    assert np.allclose(header.phases, 0.0)


def test_parse_header_projection_matrices(test_dicom_chest_file: Path):
    header = parse_dicom_header(test_dicom_chest_file)

    assert header.projection_matrices is not None
    assert header.projection_matrices.shape == (EXPECTED_FRAMES, 12)
    assert np.all(np.isfinite(header.projection_matrices))


def test_parse_header_accepts_bytes(test_dicom_chest_file: Path):
    from_bytes = parse_dicom_header(test_dicom_chest_file.read_bytes())
    from_path = parse_dicom_header(test_dicom_chest_file)

    assert from_bytes.num_frames == from_path.num_frames
    assert from_bytes.sdd == from_path.sdd
    assert np.array_equal(from_bytes.alphas_degree, from_path.alphas_degree)


def test_extract_pixels(test_dicom_chest_file: Path):
    pixels = extract_pixels(test_dicom_chest_file)

    assert pixels.shape == (EXPECTED_FRAMES, EXPECTED_HEIGHT, EXPECTED_WIDTH)
    assert pixels.dtype == np.uint16
    assert pixels.max() > pixels.min()


def test_parse_header_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_dicom_header(tmp_path / "does_not_exist.dcm")


def test_normalize_01_min_max():
    values = np.array([10, 20, 30, 40], dtype=np.uint16)
    normalized = normalize_01(values)

    assert normalized.dtype == np.float32
    assert normalized[0] == pytest.approx(0.0)
    assert normalized[-1] == pytest.approx(1.0)
    assert normalized[1] == pytest.approx(1.0 / 3.0)


def test_normalize_01_constant_is_zero():
    assert np.allclose(normalize_01(np.full((4,), 5, dtype=np.uint16)), 0.0)


def test_normalize_01_percentile_clips_outliers():
    # A broad, well-sampled intensity distribution plus one very bright outlier.
    base = np.arange(1000, dtype=np.uint16) + 100  # 100 .. 1099
    values = np.concatenate([base, np.array([60000], dtype=np.uint16)])

    plain = normalize_01(values)
    robust = normalize_01_percentile(values, 0.01, 0.99)

    # The outlier inflates the global max and crushes the plain min-max median...
    assert float(np.median(plain)) < 0.02
    # ...while percentile clipping keeps the useful intensity range visible.
    assert float(robust.min()) == pytest.approx(0.0)
    assert float(robust.max()) == pytest.approx(1.0)
    assert float(np.median(robust)) > 0.3
    assert float(np.median(robust)) > float(np.median(plain))
