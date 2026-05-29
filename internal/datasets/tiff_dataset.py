from typing import NamedTuple
from pathlib import Path
from dataclasses import dataclass

import torch
from PIL import Image
import numpy as np
import tifffile as tiff

from ..cameras import Cameras
from .gs_dataset import GSImageDataset, GSImageDatasetConfig, ItemT, ImageItemT


PixelPosition = NamedTuple("PixelPosition", [
    ("x", int),
    ("y", int),
])


ROI = NamedTuple("ROI", [
    ("top_left", PixelPosition),
    ("bottom_right", PixelPosition),
])


def apply_roi(
    tiff_data: np.ndarray,
    cameras: Cameras,
    roi: ROI,
) -> tuple[np.ndarray, Cameras]:
    x0, y0 = roi.top_left
    x1, y1 = roi.bottom_right
    
    r"""
    矩阵（numpy）使用行列坐标，读取方式为data[row, col]
    绘图时, x 轴向右，y 轴向下, 坐标为 (x, y)，row对应y轴，x对应x轴

        0    1    2    3    4   
    --------------------------→ x
    0 | [a00, a01, a02, a03, a04],
    1 | [a10, a11, a12, a13, a14],
    2 | [a20, a21, a22, a23, a24],
    3 | [a30, a31, a32, a33, a34]
      ▼
      y
    
    a03 对应的坐标是 (3, 0)
    data.shape = (4, 5)  # (height, width)
    x ∈ {0, 1, 2, 3, 4}
    y ∈ {0, 1, 2, 3}

    """
    _, H, W = tiff_data.shape
    assert 0 <= x0 < x1 <= W, f"Invalid ROI x coordinates: {x0}, {x1}, image width: {W}"
    assert 0 <= y0 < y1 <= H, f"Invalid ROI y coordinates: {y0}, {y1}, image height: {H}"
    
    tiff_data = tiff_data[:, y0:y1, x0:x1]
    
    width = x1 - x0
    height = y1 - y0
    n_camras = len(cameras)
    camera = Cameras.build(
        idx = cameras.idx,
        R = cameras.R,
        T = cameras.T,
        fx = cameras.fx,
        fy = cameras.fy,
        cx = cameras.cx - x0,   # 注意这里要减去x0，因为ROI裁剪后，新的坐标系原点是ROI的左上角
        cy = cameras.cy - y0,
        width = torch.ones(n_camras) * width,
        height = torch.ones(n_camras) * height,
        time=cameras.time,
        zfar=cameras.zfar,
        znear=cameras.znear
    )
    return tiff_data, camera

@dataclass
class TiffDatasetConfig(GSImageDatasetConfig):
    roi: ROI = ROI(
        top_left=PixelPosition(200, 200),
        bottom_right=PixelPosition(450, 400),
    )
    val_min: float|None = None
    val_max: float|None = None


class TiffDataset(GSImageDataset):
    cameras: Cameras
    cfg: TiffDatasetConfig
    
    tiff_path: Path
    tiff_data: torch.Tensor
    
    def __init__(
        self, 
        cameras: Cameras, 
        cfg: TiffDatasetConfig,
        tiff_path: Path,
    ):
        super().__init__(cameras, cfg)
        self.tiff_path = tiff_path
        
        # read tiff data to get image shape
        tiff_data = tiff.imread(self.tiff_path).astype(np.float32)
        
        tiff_data, self.cameras = apply_roi(tiff_data, cameras, cfg.roi)
        
        vol_max = tiff_data.max() if cfg.val_max is None else cfg.val_max
        vol_min = tiff_data.min() if cfg.val_min is None else cfg.val_min
        tiff_data = (tiff_data - vol_min) / (vol_max - vol_min)  # normalize to [0, 1]
        
        tiff_data = torch.from_numpy(tiff_data)
        if (device := self.cfg.image_cache_device) is not None:
            tiff_data = tiff_data.to(device)
        self.tiff_data = tiff_data
        
    def get_image(self, index: int) -> torch.Tensor:
        """ Return image tensor in float format, shape (1, H, W) """
        image = self.tiff_data[index].unsqueeze(0)  # (H, W) -> (1, H, W)
        return image
    
    def __getitem__(self, index: int) -> ItemT:
        image = self.get_image(index)
        batch = ItemT(
            camera=self.cameras[index],
            image=ImageItemT(
                image_name=f"{self.tiff_path.name}_{index}",
                gt_image=image,
                mask=None,
            ),
            extra_data=None,
        )
        return batch
