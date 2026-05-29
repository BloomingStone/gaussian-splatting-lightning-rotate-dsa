from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch
from torch.utils.data import DataLoader

from internal.dataparsers.tiff_dataparser import TiffDataParserConfig
from internal.datasets.gs_dataset import collate_fn
from internal.datasets.tiff_dataset import TiffDatasetConfig


DATA_ROOTS = [
    Path("data/rbf_reader_flow_contrast_LCA"),
    Path("data/rbf_reader_flow_contrast_RCA"),
]
OUTPUT_ROOT = Path("tests/output/tiff")


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().cpu().float()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze(-1)
    plt.imsave(path, image.numpy(), cmap="gray" if image.ndim == 2 else None)


def _make_parser(data_root: Path):
    return TiffDataParserConfig(
        dataset_config=TiffDatasetConfig(
            camera_cache_device="cpu",
            image_cache_device="cpu",
        ),
        base_name="rotate_dsa",
        mode="reconstruction",
        init_point_cloud_mode="central-line",
    ).instantiate(
        path=str(data_root),
        output_path=str(OUTPUT_ROOT / data_root.name),
        global_rank=0,
    )


@pytest.mark.parametrize("data_root", DATA_ROOTS)
def test_tiff_dataparser_outputs(data_root: Path):
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
    assert gt_image.ndim == 3 and gt_image.shape[0] == 1
    assert mask is None
    assert extra_data is None

    _save_image(gt_image, OUTPUT_ROOT / data_root.name / "train_item_0.png")


@pytest.mark.parametrize("data_root", DATA_ROOTS)
def test_tiff_dataloader_collate_fn(data_root: Path):
    parser = _make_parser(data_root)
    outputs = parser.get_outputs()

    loader = DataLoader(
        outputs.train_set,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))

    item0 = outputs.train_set[0]
    item1 = outputs.train_set[1]

    assert batch.cameras.R.shape[0] == 2
    assert batch.images.gt_images.shape == (2, 1, item0.image.gt_image.shape[1], item0.image.gt_image.shape[2])
    assert batch.images.masks is None
    assert batch.extra_data is None
    assert batch.images.image_names == [item0.image.image_name, item1.image.image_name]
    assert torch.allclose(batch.images.gt_images[0], item0.image.gt_image)
    assert torch.allclose(batch.images.gt_images[1], item1.image.gt_image)

    _save_image(batch.images.gt_images[0], OUTPUT_ROOT / data_root.name / "loader_batch_0.png")