from pathlib import Path

from pytest import fixture

@fixture
def test_xray_data_root():
    return Path("data/rbf_reader_flow_contrast_LCA")

@fixture
def test_xray_data_no_flow_root():
    return Path("data/Diseased_17")

@fixture
def test_pigdata_root():
    return Path("data/pigdata")

@fixture
def output_root():
    res = Path("tests/output/xray")
    res.mkdir(parents=True, exist_ok=True)
    return res