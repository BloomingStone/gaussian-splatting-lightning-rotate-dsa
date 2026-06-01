from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from matplotlib import pyplot as plt

from internal.dataparsers.xray_dataparser.cloud_parsers import FdkCloudParser
from internal.dataparsers.xray_dataparser.meta import XRayMetaLoader
from internal.dataparsers.xray_dataparser.splitters import ReconstructionSpliter

from .common import make_point_cloud, save_point_cloud


def test_fdk_cloud_parser_loads_png_file_list_in_sorted_order(tmp_path: Path) -> None:
    pillow = pytest.importorskip("PIL.Image")

    values = [20, 0, 10]
    names = ["2.png", "0.png", "1.png"]
    for name, value in zip(names, values, strict=True):
        image = np.full((4, 5), value, dtype=np.uint8)
        pillow.fromarray(image).save(tmp_path / name)

    projections = FdkCloudParser._load_png_projections(tmp_path, [0, 1, 2])

    assert projections.shape == (3, 4, 5)
    assert projections[:, 0, 0].tolist() == [0, 10, 20]


def test_fbp_reconstruction_saves_volume_and_init_point_cloud(
    test_xray_data_root: Path,
    output_root: Path,
) -> None:
    meta = XRayMetaLoader().load(test_xray_data_root)
    splits = ReconstructionSpliter().split(test_xray_data_root, meta)

    parser = FdkCloudParser(
        num_points=1000,
        seed=42,
        use_filter=True,
        phase_min=0.0,
        phase_max=0.2,
        image_dir_name=None,
        tiff_file_name="rotate_dsa.tif",
    )

    xyz, volume = parser._init_point_cloud_from_fbp(test_xray_data_root, meta, splits)

    output_dir = output_root / "fbp_reconstruction" / test_xray_data_root.name
    output_dir.mkdir(parents=True, exist_ok=True)

    volume_slice_path = output_dir / "volume_slice.png"
    volume_slice_z = volume.shape[2] // 2
    plt.figure(figsize=(6, 6))
    plt.imshow(volume[:, :, volume_slice_z], cmap="gray")
    plt.axis("off")
    plt.savefig(volume_slice_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    volume_path = output_dir / "volume.nii.gz"
    point_cloud_path = output_dir / "point_cloud.npz"
    point_cloud_image_path = output_dir / "point_cloud.png"

    nib.save(nib.Nifti1Image(volume.astype(np.float32), meta.centering_affine), volume_path)
    np.savez_compressed(point_cloud_path, xyz=xyz.astype(np.float32))
    save_point_cloud(make_point_cloud(xyz), point_cloud_image_path)

    assert volume_path.exists()
    assert point_cloud_path.exists()
    assert point_cloud_image_path.exists()
    assert volume_slice_path.exists()
    assert volume.shape == tuple(meta.volume_size)
    assert xyz.shape[0] == 1000
    assert xyz.shape[1] == 3
    assert np.isfinite(volume).all()
