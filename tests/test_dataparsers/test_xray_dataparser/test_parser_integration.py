from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch
from torch.utils.data import DataLoader

from internal.dataparsers.dataparser import collate_fn
from internal.dataparsers.xray_dataparser import XRayDataParser
from internal.dataparsers.xray_dataparser.cameras_builder import RotateXRayCamerasBuilder
from internal.dataparsers.xray_dataparser.cloud_parsers import RandomCloudParser
from internal.dataparsers.xray_dataparser.datasets import (
    ImagesDatasetBuilder,
    ImagesDatasetConfig,
    TiffDatasetBuilder,
    TiffDatasetConfig,
)
from internal.dataparsers.xray_dataparser.meta import XRayMetaLoader
from internal.dataparsers.xray_dataparser.splitters import ReconstructionSpliter, RenderNewViewsSpliter

ROTATED_XRAY_ROOTS = [
    Path("data/rbf_reader_flow_contrast_LCA"),
    Path("data/rbf_reader_flow_contrast_RCA"),
]
TIFF_ROOTS = [
    Path("data/pigdata"),
]


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().cpu().float()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze(-1)
    plt.imsave(path, image.numpy(), cmap="gray" if image.ndim == 2 else None)


def _save_depth(depth: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_image = depth.detach().cpu().float()
    depth_min = depth_image.min()
    depth_max = depth_image.max()
    if float(depth_max - depth_min) > 0:
        depth_image = (depth_image - depth_min) / (depth_max - depth_min)
    else:
        depth_image = torch.zeros_like(depth_image)
    plt.imsave(path, depth_image.numpy(), cmap="magma")


def _save_cloud_xy(xyz, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(xyz, torch.Tensor):
        points = xyz.detach().cpu().numpy()
    else:
        points = xyz
    plt.figure(figsize=(6, 4))
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _make_rotated_xray_parser() -> XRayDataParser:
    return XRayDataParser(
        meta_loader=XRayMetaLoader(),
        cloud_parser=RandomCloudParser(num_points=100_000),
        spliter=RenderNewViewsSpliter(train_ratio=0.8),
        cameras_builder=RotateXRayCamerasBuilder(),
        dataset_builder=ImagesDatasetBuilder(
            dataset_config=ImagesDatasetConfig(
                image_uint8=False,
            )
        ),
        filter_visible_points=True,
    )


def _make_tiff_parser() -> XRayDataParser:
    return XRayDataParser(
        meta_loader=XRayMetaLoader(),
        cloud_parser=RandomCloudParser(num_points=100_000),
        spliter=ReconstructionSpliter(),
        cameras_builder=RotateXRayCamerasBuilder(),
        dataset_builder=TiffDatasetBuilder(
            base_name="rotate_dsa",
            dataset_config=TiffDatasetConfig(),
        ),
        filter_visible_points=True,
    )


@pytest.mark.parametrize("data_root", ROTATED_XRAY_ROOTS)
def test_rotated_xray_parser_outputs_and_collate(data_root: Path, output_root: Path):
    parser = _make_rotated_xray_parser()
    outputs = parser.get_outputs(data_root)
    output_dir = output_root / "rotated_xray" / data_root.name

    assert len(outputs.train_set) > 0
    assert len(outputs.val_set) > 0
    assert len(outputs.test_set) > 0
    assert outputs.point_cloud.xyz.shape[1] == 3
    assert outputs.camera_extent is not None and outputs.camera_extent > 0

    _save_cloud_xy(outputs.point_cloud.xyz, output_dir / "point_cloud_xy.png", f"point cloud: {data_root.name}")

    item0 = outputs.train_set[0]
    assert isinstance(item0.image.image_name, str)
    assert item0.image.gt_image.ndim == 3 and item0.image.gt_image.shape[0] == 3
    assert item0.image.mask is not None and item0.image.mask.shape == item0.image.gt_image.shape
    assert item0.extra_data is not None and "depth" in item0.extra_data
    assert item0.extra_data["depth"].shape == item0.image.gt_image.shape[1:]

    _save_image(item0.image.gt_image, output_dir / "train_item_0.png")
    _save_image(item0.image.mask.float(), output_dir / "train_mask_0.png")
    _save_depth(item0.extra_data["depth"], output_dir / "train_depth_0.png")

    loader = DataLoader(
        outputs.train_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))

    assert batch.camera.R.dim() == 2
    assert batch.image.gt_image.shape == (
        3,
        item0.image.gt_image.shape[1],
        item0.image.gt_image.shape[2],
    )
    assert batch.image.mask is not None
    assert batch.extra_data is not None and "depth" in batch.extra_data
    assert batch.image.image_name == item0.image.image_name
    assert torch.allclose(batch.image.gt_image, item0.image.gt_image)

    _save_image(batch.image.gt_image, output_dir / "loader_batch_0.png")
    _save_depth(batch.extra_data["depth"], output_dir / "loader_depth_0.png")


@pytest.mark.parametrize("data_root", TIFF_ROOTS)
def test_tiff_parser_outputs_and_collate(data_root: Path, output_root: Path):
    parser = _make_tiff_parser()
    outputs = parser.get_outputs(data_root)
    output_dir = output_root / "tiff" / data_root.name

    assert len(outputs.train_set) > 0
    assert len(outputs.val_set) > 0
    assert len(outputs.test_set) > 0
    assert outputs.point_cloud.xyz.shape[1] == 3
    assert outputs.camera_extent is not None and outputs.camera_extent > 0

    _save_cloud_xy(outputs.point_cloud.xyz, output_dir / "point_cloud_xy.png", f"point cloud: {data_root.name}")

    item0 = outputs.train_set[0]
    assert isinstance(item0.image.image_name, str)
    assert item0.image.gt_image.ndim == 3 and item0.image.gt_image.shape[0] == 1
    assert item0.image.mask is None
    assert item0.extra_data is None

    _save_image(item0.image.gt_image, output_dir / "train_item_0.png")

    loader = DataLoader(
        outputs.train_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))

    assert batch.camera.R.dim() == 2
    assert batch.image.gt_image.shape == (
        1,
        item0.image.gt_image.shape[1],
        item0.image.gt_image.shape[2],
    )
    assert batch.image.mask is None
    assert batch.extra_data is None
    assert batch.image.image_name == item0.image.image_name
    assert torch.allclose(batch.image.gt_image, item0.image.gt_image)

    _save_image(batch.image.gt_image, output_dir / "loader_batch_0.png")