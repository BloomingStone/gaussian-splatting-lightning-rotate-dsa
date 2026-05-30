from pathlib import Path

import numpy as np
import tifffile as tiff
import torch

from internal.dataparsers.xray_dataparser.cameras_builder import RotateXRayCamerasBuilder
from internal.dataparsers.xray_dataparser.datasets import (
    ImagesDatasetBuilder,
    ImagesDatasetConfig,
    PixelPosition,
    ROI,
    TiffDatasetBuilder,
    TiffDatasetConfig,
    apply_roi,
)
from internal.dataparsers.xray_dataparser.meta import XRayMetaLoader
from internal.dataparsers.xray_dataparser.splitters import ReconstructionSpliter

from .common import make_point_cloud, load_test_meta, save_cameras


def _make_cameras(test_pigdata_root: Path):
    return RotateXRayCamerasBuilder().build_cameras(load_test_meta(test_pigdata_root))


def test_apply_roi_crops_tiff_and_updates_camera_geometry(test_pigdata_root: Path):
    tiff_data = np.arange(2 * 6 * 8, dtype=np.float32).reshape(2, 6, 8)
    cameras = _make_cameras(test_pigdata_root).get_from_indices([0, 1])
    roi = ROI(top_left=PixelPosition(2, 1), bottom_right=PixelPosition(6, 5))

    cropped_data, cropped_cameras = apply_roi(tiff_data, cameras, roi)

    assert cropped_data.shape == (2, 4, 4)
    assert torch.allclose(cropped_cameras.cx, cameras.cx - 2)
    assert torch.allclose(cropped_cameras.cy, cameras.cy - 1)
    assert torch.allclose(cropped_cameras.width, torch.full((2,), 4.0).to(cameras.width))
    assert torch.allclose(cropped_cameras.height, torch.full((2,), 4.0).to(cameras.height))


def test_tiff_dataset_builder_resolves_and_reads_tiff(tmp_path: Path, output_root: Path, test_pigdata_root: Path):
    meta = load_test_meta(test_pigdata_root)
    tiff_data = np.stack(
        [
            np.arange(6 * 8, dtype=np.float32).reshape(6, 8),
            np.arange(6 * 8, dtype=np.float32).reshape(6, 8) + 100.0,
            np.arange(6 * 8, dtype=np.float32).reshape(6, 8) + 200.0,
        ],
        axis=0,
    )
    tiff.imwrite(tmp_path / "rotate_dsa.tif", tiff_data)

    builder = TiffDatasetBuilder(
        base_name="rotate_dsa",
        dataset_config=TiffDatasetConfig(
            camera_cache_device="cuda",
            image_cache_device="cuda",
            roi=ROI(top_left=PixelPosition(1, 1), bottom_right=PixelPosition(5, 4)),
        ),
    )
    cameras = RotateXRayCamerasBuilder().build_cameras(meta)
    dataset = builder.build_dataset(tmp_path, cameras, meta, [0, 1], "train")

    assert len(dataset) == 2
    assert dataset.image_names == ["rotate_dsa.tif_0", "rotate_dsa.tif_1"]
    item = dataset[0]
    assert item.image.gt_image.shape == (1, 3, 4)
    assert item.image.mask is None
    assert item.extra_data is None
    assert torch.allclose(item.camera.cx, cameras.cx[0] - 1)
    assert torch.allclose(item.camera.cy, cameras.cy[0] - 1)

    save_cameras(
        dataset.cameras,
        output_root / "tiff_dataset_cameras(2_cameras).png",
    )


def test_images_dataset_builder_reads_real_fixture_data(test_xray_data_root: Path, output_root: Path):
    meta = XRayMetaLoader().load(test_xray_data_root)
    cameras = RotateXRayCamerasBuilder().build_cameras(meta)
    indices = ReconstructionSpliter().split(test_xray_data_root, meta)["train"]

    builder = ImagesDatasetBuilder(
        image_dir_name="rotate_dsa",
        mask_dir_name="label",
        image_suffix="*.png",
        use_depth_map=True,
        depth_map_filename="depth_map.npz",
        dataset_config=ImagesDatasetConfig(
            camera_cache_device="cuda",
            image_cache_device="cuda",
            image_uint8=False,
        ),
    )
    dataset = builder.build_dataset(test_xray_data_root, cameras, meta, indices, "train")

    assert len(dataset) == len(indices)
    item = dataset[0]
    assert item.image.gt_image.shape[0] == 3
    assert item.image.mask is not None and item.image.mask.shape == item.image.gt_image.shape
    assert item.extra_data is not None and "depth" in item.extra_data
    assert item.extra_data["depth"].ndim == 2
    assert torch.is_floating_point(item.image.gt_image)

    save_cameras(
        dataset.cameras,
        output_root / "images_dataset_cameras.png",
    )