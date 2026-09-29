from pathlib import Path

from pytest import fixture

@fixture
def test_xray_data_root():
    return Path("data/test_data")

@fixture
def test_xray_data_no_flow_root():
    return Path("data/test_data_motion")

@fixture
def test_pigdata_root():
    return Path("data/pigdata")

@fixture
def output_root():
    res = Path("tests/output/xray")
    res.mkdir(parents=True, exist_ok=True)
    return res