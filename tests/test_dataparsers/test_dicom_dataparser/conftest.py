from pathlib import Path

import pytest


@pytest.fixture
def test_dicom_data_root() -> Path:
    root = Path("data/dicom")
    if not root.is_dir():
        pytest.skip(f"DICOM test data directory not found: {root}")
    return root


@pytest.fixture
def test_dicom_chest_file(test_dicom_data_root: Path) -> Path:
    path = test_dicom_data_root / "RXA_chest.dcm"
    if not path.is_file():
        pytest.skip(f"DICOM test file not found: {path}")
    return path
