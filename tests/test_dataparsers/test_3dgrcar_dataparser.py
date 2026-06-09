from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from .common import save_cameras, save_point_cloud, save_point_cloud_and_cameras
from internal.cameras import Cameras
from internal.dataparsers.dataparser import DataParser
from internal.dataparsers.threeDGR_parser import (
    MU_IDODINE,
    ThreeDGRCarMetaLoader,
    ThreeDGRCarCamerasBuilder,
    UniformCloudParser,
    SparseViewPickSpliter,
    ThreeDGRCarDataset,
    ThreeDGRCarDatasetBuilder,
    ThreeDGRCarDataParserBuilder,
    ThreeDGRCarMeta,
)

_NUM_VIEWS = 4


@pytest.fixture
def case_name() -> str:
    return "Diseased_1_lca"


@pytest.fixture
def data_dir() -> Path:
    return Path("data/asoca")


@pytest.fixture
def output_root() -> Path:
    res = Path("tests/output/3dgrcar")
    res.mkdir(parents=True, exist_ok=True)
    return res


@pytest.fixture
def meta(data_dir: Path, case_name: str) -> ThreeDGRCarMeta:
    return ThreeDGRCarMetaLoader(case_name).load(data_dir)


@pytest.fixture
def cameras(meta: ThreeDGRCarMeta) -> Cameras:
    return ThreeDGRCarCamerasBuilder().build_cameras(meta)


# ============================================================
#  Unit tests
# ============================================================

def test_meta_loader(meta: ThreeDGRCarMeta):
    assert meta.xca_projs is not None
    assert meta.ori_projs is not None
    assert meta.label_projs is not None
    assert meta.ori_projs_meta is not None
    assert meta.label_projs_meta is not None
    assert meta.default_use_proj in ("ori", "label", "xca", "xca-raw")


def test_cameras_builder(meta: ThreeDGRCarMeta, cameras: Cameras, output_root: Path):
    save_cameras(cameras, output_root / "cameras_builder.png")


def test_cloud_parser(meta: ThreeDGRCarMeta, cameras: Cameras, output_root: Path):
    cloud_parser = UniformCloudParser(num_points=1000)
    point_cloud = cloud_parser.get_point_cloud(data_dir=None, meta=meta)  # type: ignore[arg-type]

    assert point_cloud.xyz.shape == (1000, 3)
    assert point_cloud.feature.shape == (1000, 3)

    save_point_cloud_and_cameras(
        point_cloud, cameras,
        output_root / "cloud_parser.png",
    )


def test_spliter(meta: ThreeDGRCarMeta):
    spliter = SparseViewPickSpliter(num_views=_NUM_VIEWS, val_on_selected_views=True)
    splits = spliter.split(data_dir=None, meta=meta)  # type: ignore[arg-type]

    assert set(splits.keys()) == {"train", "val", "test"}
    assert len(splits["train"]) == _NUM_VIEWS
    assert splits["val"] == splits["train"]
    assert splits["test"] == splits["train"]

    # val_on_selected_views=False — val uses unseen views
    spliter2 = SparseViewPickSpliter(num_views=_NUM_VIEWS, val_on_selected_views=False)
    splits2 = spliter2.split(data_dir=None, meta=meta)  # type: ignore[arg-type]
    assert len(splits2["train"]) == _NUM_VIEWS
    assert len(splits2["val"]) == meta.projs_meta.param.num_proj - _NUM_VIEWS

    # error case: num_views > total
    n_total = meta.projs_meta.param.num_proj
    with pytest.raises(ValueError, match="greater than the total"):
        SparseViewPickSpliter(num_views=n_total + 1).split(data_dir=None, meta=meta)  # type: ignore[arg-type]

def test_vis_splits_cameras(meta: ThreeDGRCarMeta, output_root: Path):
    spliter = SparseViewPickSpliter(num_views=_NUM_VIEWS, val_on_selected_views=True)
    splits = spliter.split(data_dir=None, meta=meta)  # type: ignore[arg-type]
    
    cameras = ThreeDGRCarCamerasBuilder().build_cameras(meta)

    for split_name, indices in splits.items():
        save_cameras(cameras.get_from_indices(indices), output_root / f"cameras_{_NUM_VIEWS}_val-on-train_{split_name}.png")
        
    spliter = SparseViewPickSpliter(num_views=2, val_on_selected_views=False)
    splits = spliter.split(data_dir=None, meta=meta)  # type: ignore[arg-type]
    
    for split_name, indices in splits.items():
        save_cameras(cameras.get_from_indices(indices), output_root / f"cameras_2_val-on-unseen_{split_name}.png")


def test_dataset(meta: ThreeDGRCarMeta, cameras: Cameras, output_root: Path):
    indices = [0, 2, 4]
    dataset = ThreeDGRCarDataset(meta=meta, cameras=cameras, indices=indices)

    assert len(dataset) == 3
    assert dataset.image_names == ["proj_000", "proj_002", "proj_004"]

    item = dataset[0]
    assert item.camera.idx.item() == 0
    assert item.image.image_name == "proj_000"
    assert item.image.gt_image.shape == (1, meta.projs_meta.param.nh, meta.projs_meta.param.nw)
    
    from matplotlib import pyplot as plt
    plt.imshow(item.image.gt_image.cpu().numpy()[0], cmap="gray")
    plt.axis("off")
    plt.savefig(output_root / "dataset_item.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    
    
    from internal.visualizers import FloatColormapVisualizer, ColorMapName, Visualizer
    from PIL import Image
    visualizer = FloatColormapVisualizer(ColorMapName.GRAY)
    vis_img = visualizer.process(item.image.gt_image.cpu())
    vis_img_pil = Image.fromarray((vis_img*255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8))
    vis_img_pil.save(output_root / "dataset_item_visualized.png")


# ============================================================
#  xca-raw option tests
# ============================================================

def test_meta_projs_xca_raw(meta: ThreeDGRCarMeta, output_root: Path):
    """Test projs and projs_meta properties with xca-raw option."""
    meta.default_use_proj = "xca-raw"

    # projs_meta should return ori_projs_meta (same as xca)
    assert meta.projs_meta is meta.ori_projs_meta

    # projs should return exp(-(ori + label * MU_IDODINE))
    expected = torch.exp(-(meta.ori_projs + meta.label_projs * MU_IDODINE))
    torch.testing.assert_close(meta.projs, expected)

    # Visualize: side-by-side comparison of ori, label, and xca-raw
    import matplotlib.pyplot as plt
    from internal.visualizers import FloatColormapVisualizer, ColorMapName

    visualizer = FloatColormapVisualizer(ColorMapName.GRAY)
    proj_idx = 0

    ori_img = visualizer.process(meta.ori_projs[proj_idx].cpu())
    label_img = visualizer.process(meta.label_projs[proj_idx].cpu())
    xca_raw_img = visualizer.process(meta.projs[proj_idx].cpu())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, [ori_img, label_img, xca_raw_img],
                               ["ori", "label", "xca-raw"]):
        ax.imshow(img.permute(1, 2, 0).numpy(), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_root / "meta_projs_xca_raw_comparison.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def test_meta_projs_all_options(meta: ThreeDGRCarMeta):
    """Test projs property returns the correct tensor for each option (including xca-raw)."""
    # ori
    meta.default_use_proj = "ori"
    assert meta.projs is meta.ori_projs
    assert meta.projs_meta is meta.ori_projs_meta

    # label
    meta.default_use_proj = "label"
    assert meta.projs is meta.label_projs
    assert meta.projs_meta is meta.label_projs_meta

    # xca
    meta.default_use_proj = "xca"
    assert meta.projs is meta.xca_projs
    assert meta.projs_meta is meta.ori_projs_meta

    # xca-raw
    meta.default_use_proj = "xca-raw"
    assert meta.projs_meta is meta.ori_projs_meta
    expected = torch.exp(-(meta.ori_projs + meta.label_projs * MU_IDODINE))
    torch.testing.assert_close(meta.projs, expected)


def test_data_parser_builder_visualizer_xca_raw(data_dir: Path, output_root: Path, case_name: str):
    """Test that xca-raw uses GammaVisualizer for gt_image_visualizer."""
    from internal.visualizers import GammaVisualizer, ColorMapName
    from PIL import Image

    builder = ThreeDGRCarDataParserBuilder(
        meta_loader=ThreeDGRCarMetaLoader(case_name=case_name, default_use_proj="xca-raw"),
        cloud_parser=UniformCloudParser(num_points=1000),
        spliter=SparseViewPickSpliter(num_views=_NUM_VIEWS, val_on_selected_views=True),
    )
    parser = builder.build()
    assert isinstance(parser.gt_image_visualizer, GammaVisualizer)
    # Verify GammaVisualizer has the expected parameters
    assert parser.gt_image_visualizer.gamma == pytest.approx(0.1)
    assert parser.gt_image_visualizer.colormap == ColorMapName.GRAY

    # Visualize: apply the GammaVisualizer to an actual image
    outputs = parser.get_outputs(data_dir)
    item = outputs.train_set[0]
    vis_img = parser.gt_image_visualizer.process(item.image.gt_image.cpu())
    vis_img_pil = Image.fromarray(
        (vis_img * 255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
    )
    vis_img_pil.save(output_root / "visualizer_xca_raw.png")


def test_data_parser_builder_visualizer_default(data_dir: Path, output_root: Path, case_name: str):
    """Test that non-xca-raw uses FloatColormapVisualizer for gt_image_visualizer."""
    from internal.visualizers import FloatColormapVisualizer, ColorMapName
    from PIL import Image

    builder = ThreeDGRCarDataParserBuilder(
        meta_loader=ThreeDGRCarMetaLoader(case_name=case_name),
        cloud_parser=UniformCloudParser(num_points=1000),
        spliter=SparseViewPickSpliter(num_views=_NUM_VIEWS, val_on_selected_views=True),
    )
    parser = builder.build()
    visualizer = parser.gt_image_visualizer
    assert isinstance(visualizer, FloatColormapVisualizer)
    assert visualizer.colormap == ColorMapName.GRAY

    # Visualize: apply the FloatColormapVisualizer to an actual image
    outputs = parser.get_outputs(data_dir)
    item = outputs.train_set[0]
    vis_img = parser.gt_image_visualizer.process(item.image.gt_image.cpu())
    vis_img_pil = Image.fromarray(
        (vis_img * 255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
    )
    vis_img_pil.save(output_root / "visualizer_default.png")


def test_data_parser_builder_visualizer_xca_raw_via_meta(
    meta: ThreeDGRCarMeta, cameras: Cameras, output_root: Path
):
    """Verify that a dataset with xca-raw meta produces the correct (transformed) pixel values."""
    from internal.visualizers import FloatColormapVisualizer, ColorMapName
    from PIL import Image
    import matplotlib.pyplot as plt

    meta.default_use_proj = "xca-raw"

    indices = [0, 2]
    dataset = ThreeDGRCarDataset(meta=meta, cameras=cameras, indices=indices)
    item = dataset[0]

    # xca-raw = exp(-(ori + label * MU_IDODINE)), so values must be in (0, 1]
    img = item.image.gt_image
    assert img.min() > 0.0
    assert img.max() <= 1.0

    # Verify against the known formula
    expected = torch.exp(-(meta.ori_projs[indices[0]] + meta.label_projs[indices[0]] * MU_IDODINE))
    expected = torch.rot90(expected, k=1, dims=(-2, -1))[None]  # match __getitem__ transform
    torch.testing.assert_close(img, expected)

    # Visualize: side-by-side of raw xca-raw image and gamma-visualized version
    visualizer = FloatColormapVisualizer(ColorMapName.GRAY)
    raw_vis = visualizer.process(img.cpu())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(raw_vis.permute(1, 2, 0).numpy(), cmap="gray")
    axes[0].set_title("xca-raw (colormap)")
    axes[0].axis("off")

    axes[1].imshow(img.cpu().numpy()[0], cmap="gray")
    axes[1].set_title("xca-raw (raw)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_root / "xca_raw_via_meta.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # Also save the raw image via PIL for easy inspection
    raw_img_pil = Image.fromarray(
        (img * 255).clamp(0, 255).squeeze(0).numpy().astype(np.uint8), mode="L"
    )
    raw_img_pil.save(output_root / "xca_raw_via_meta_raw.png")


# ============================================================
#  Integration test: DataParser + DataLoader
# ============================================================

def test_integration(data_dir: Path, output_root: Path, case_name: str):
    builder = ThreeDGRCarDataParserBuilder(
        meta_loader=ThreeDGRCarMetaLoader(case_name=case_name),
        cloud_parser=UniformCloudParser(num_points=1000),
        spliter=SparseViewPickSpliter(num_views=_NUM_VIEWS, val_on_selected_views=True),
        cameras_builder=ThreeDGRCarCamerasBuilder(),
        dataset_builder=ThreeDGRCarDatasetBuilder(),
        filter_visible_points=False,
    )
    parser: DataParser = builder.build()
    outputs = parser.get_outputs(data_dir)

    # --- structure ---
    assert set(outputs.splits.keys()) == {"train", "val", "test"}
    assert outputs.meta is not None

    # --- point cloud ---
    pc = outputs.point_cloud
    assert pc.xyz.shape[0] == 1000
    assert pc.xyz.shape[1] == 3

    # --- datasets ---
    train_set = outputs.train_set
    val_set = outputs.val_set

    assert len(train_set) == _NUM_VIEWS
    assert len(val_set) == _NUM_VIEWS

    # --- DataLoader smoke test ---
    train_loader = DataLoader(train_set, batch_size=1, collate_fn=train_set.batch_one_collate_fn)
    batch = next(iter(train_loader))
    assert batch.camera is not None
    assert batch.image is not None
    img = batch.image.gt_image
    assert isinstance(img, torch.Tensor)
    assert img.dim() == 3

    # --- visualise cloud + cameras ---
    save_point_cloud_and_cameras(
        pc, train_set.cameras,
        output_root / "integration.png",
    )
