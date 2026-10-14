import importlib
from pathlib import Path

import numpy as np
import torch
import yaml

from internal.dataparsers.dicom_dataparser import (
    DicomCamerasBuilder,
    DicomDataParserBuilder,
    DicomDatasetBuilder,
    DicomDatasetConfig,
    DicomMetaLoader,
)
from internal.dataparsers.xray_dataparser.cloud_parsers import RandomCloudParser
from internal.dataparsers.xray_dataparser.splitters import UniformIntervalSpliter

EXPECTED_FRAMES = 187
CONFIG_PATH = Path("configs/dicom/dicom_chest.yaml")


def test_parser_builds_all_outputs(test_dicom_data_root: Path):
    parser = DicomDataParserBuilder(
        cloud_parser=RandomCloudParser(num_points=1_000),
    ).build()
    outputs = parser.get_outputs(test_dicom_data_root)

    assert set(outputs.splits.keys()) == {"train", "val", "test"}
    # The default ReconstructionSpliter uses every frame everywhere.
    assert len(outputs.splits["train"]) == EXPECTED_FRAMES
    assert len(outputs.train_set) == EXPECTED_FRAMES
    assert outputs.camera_extent is not None and outputs.camera_extent > 0

    item = outputs.train_set[0]
    assert int(item.camera.idx) == 0
    assert item.image.gt_image.shape == (3, 634, 862)
    assert 0.0 <= float(item.image.gt_image.min())
    assert float(item.image.gt_image.max()) <= 1.0
    assert item.image.mask is None

    assert outputs.point_cloud.xyz.shape == (1_000, 3)
    bounds_min = outputs.point_cloud.xyz.min(axis=0)
    bounds_max = outputs.point_cloud.xyz.max(axis=0)
    assert np.all(bounds_min >= -outputs.meta.volume_size / 2 - 1e-6)
    assert np.all(bounds_max <= outputs.meta.volume_size / 2 + 1e-6)


def test_parser_with_mixed_spliter_and_projection_matrices(test_dicom_data_root: Path):
    parser = DicomDataParserBuilder(
        spliter=UniformIntervalSpliter(train_ratio=0.8),
        cameras_builder=DicomCamerasBuilder(use_projection_matrix=True),
        dataset_builder=DicomDatasetBuilder(),
    ).build()
    outputs = parser.get_outputs(test_dicom_data_root)

    train = outputs.splits["train"]
    val = outputs.splits["val"]
    assert len(train) + len(val) == EXPECTED_FRAMES
    assert len(train) > len(val)
    assert len(outputs.train_set) == len(train)

    item = outputs.train_set[0]
    assert item.image.gt_image.shape[0] == 3
    assert torch.isfinite(item.camera.camera_center).all()


def test_dataset_channels_option(test_dicom_data_root: Path):
    """`channels=1` yields a single-channel target; the default is 3 so the
    SaveImage callback can concatenate it with the renderer's RGB outputs."""
    parser = DicomDataParserBuilder(
        dataset_builder=DicomDatasetBuilder(
            dataset_config=DicomDatasetConfig(channels=1),
        ),
        cloud_parser=RandomCloudParser(num_points=100),
    ).build()
    outputs = parser.get_outputs(test_dicom_data_root)

    assert outputs.train_set[0].image.gt_image.shape == (1, 634, 862)


def test_parser_meta_loader_options_flow_through(test_dicom_data_root: Path):
    parser = DicomDataParserBuilder(
        meta_loader=DicomMetaLoader(roi=(100, 50, 400, 300), flip_u=False),
    ).build()
    outputs = parser.get_outputs(test_dicom_data_root)

    assert outputs.meta.pixel_data.shape == (EXPECTED_FRAMES, 300, 400)
    assert outputs.train_set[0].image.gt_image.shape == (3, 300, 400)


def test_example_config_class_paths_are_importable():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    class_paths: list[str] = []

    def collect(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "class_path":
                    class_paths.append(value)
                else:
                    collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(config)
    assert class_paths, "config contains no class_path entries"
    for class_path in class_paths:
        module_name, _, attribute = class_path.rpartition(".")
        module = importlib.import_module(module_name)
        assert hasattr(module, attribute), f"{class_path} does not resolve"
