from pathlib import Path

import numpy as np
import torch

from internal.dataparsers.xray_dataparser.cameras_builder import RotateXRayCamerasBuilder

from .common import load_test_meta, make_point_cloud, save_cameras


def test_cameras_builder_populates_intrinsics_and_time(test_xray_data_root: Path, output_root: Path):
    meta = load_test_meta(test_xray_data_root)
    cameras = RotateXRayCamerasBuilder().build_cameras(meta)

    assert len(cameras) == meta.num_frames
    assert torch.allclose(cameras.fx, torch.full((meta.num_frames,), meta.c_arm_geometry.sdd / meta.c_arm_geometry.delx))
    assert torch.allclose(cameras.fy, torch.full((meta.num_frames,), meta.c_arm_geometry.sdd / meta.c_arm_geometry.dely))
    assert torch.allclose(cameras.cx, torch.full((meta.num_frames,), meta.c_arm_geometry.x0 + meta.c_arm_geometry.width / 2))
    assert torch.allclose(cameras.cy, torch.full((meta.num_frames,), meta.c_arm_geometry.y0 + meta.c_arm_geometry.height / 2))
    expected_time = torch.from_numpy((meta.time_array - meta.time_array.min()) / (meta.time_array.max() - meta.time_array.min())).float()
    assert torch.allclose(cameras.time, expected_time)
    assert torch.allclose(cameras.phase, torch.from_numpy(meta.phase_array).float())
    # Camera center is at distance sod from origin (for the RAS default pose)
    dist = torch.linalg.norm(cameras.camera_center[0].float())
    assert torch.allclose(dist, torch.tensor(float(meta.c_arm_geometry.sod)), atol=1e-4)

    save_cameras(
        cameras,
        output_root / "cameras_builder.png",
    )
