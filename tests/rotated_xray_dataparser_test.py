from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch
from torch.utils.data import DataLoader

from internal.dataparsers.rotated_xray_dataparser import RotatedXRay
from internal.datasets.gs_dataset import collate_fn
from internal.datasets.images_dataset import ImagesDatasetConfig
from internal.datasets.gs_dataset import BatchT


DATA_ROOTS = [
    Path("data/rbf_reader_flow_contrast_LCA"),
    Path("data/rbf_reader_flow_contrast_RCA"),
]
OUTPUT_ROOT = Path("tests/output/rotated_xray")


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


def _make_parser(data_root: Path):
    return RotatedXRay(
        dataset_config=ImagesDatasetConfig(
            image_cache_device="cpu",
            camera_cache_device="cpu",
            image_uint8=False,
        ),
        init_point_cloud_mode="central-line",
    ).instantiate(
        path=str(data_root),
        output_path=str(OUTPUT_ROOT / data_root.name),
        global_rank=0,
    )


@pytest.mark.parametrize("data_root", DATA_ROOTS)
def test_rotated_xray_dataparser_outputs(data_root: Path):
    parser = _make_parser(data_root)
    outputs = parser.get_outputs()

    assert len(outputs.train_set) > 0
    assert len(outputs.val_set) > 0
    assert len(outputs.test_set) > 0
    assert outputs.point_cloud.xyz.shape[1] == 3
    assert outputs.camera_extent is not None and outputs.camera_extent > 0

    camera, image_info, extra_data = outputs.train_set[0]
    image_name, gt_image, mask = image_info

    assert isinstance(image_name, str)
    assert gt_image.ndim == 3 and gt_image.shape[0] == 3
    assert mask is not None and mask.shape == gt_image.shape
    assert extra_data is not None and "depth" in extra_data
    assert extra_data["depth"].shape == gt_image.shape[1:]

    _save_image(gt_image, OUTPUT_ROOT / data_root.name / "train_item_0.png")
    _save_image(mask.float(), OUTPUT_ROOT / data_root.name / "train_mask_0.png")
    _save_depth(extra_data["depth"], OUTPUT_ROOT / data_root.name / "train_depth_0.png")


@pytest.mark.parametrize("data_root", DATA_ROOTS)
def test_rotated_xray_dataloader_collate_fn(data_root: Path):
    parser = _make_parser(data_root)
    outputs = parser.get_outputs()

    loader = DataLoader(
        outputs.train_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    batch: BatchT = next(iter(loader))

    item0 = outputs.train_set[0]

    assert batch.camera.R.dim() == 2
    assert batch.image.gt_image.shape == (3, item0.image.gt_image.shape[1], item0.image.gt_image.shape[2])
    assert batch.image.mask is not None
    assert batch.extra_data is not None and "depth" in batch.extra_data
    assert torch.allclose(batch.image.gt_image, item0.image.gt_image)

    _save_image(batch.image.gt_image, OUTPUT_ROOT / data_root.name / "loader_batch_0.png")
    _save_depth(batch.extra_data["depth"], OUTPUT_ROOT / data_root.name / "loader_depth_0.png")
