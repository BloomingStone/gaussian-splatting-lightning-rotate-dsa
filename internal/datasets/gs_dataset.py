from typing import NamedTuple, Generic, TypeVar
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from ..cameras import Camera, Cameras


ImageItemT = NamedTuple("ImageItemT", [
    ("image_name", str),
    ("gt_image", torch.Tensor),
    ("mask", torch.Tensor|None),  # bool mask, True is valid pixel, False is masked pixel
])

ItemT = NamedTuple("ItemT", [
    ("camera", Camera),
    ("image", ImageItemT),
    ("extra_data", dict[str, torch.Tensor]|None),
])

BatchT = ItemT

def collate_fn(batch: list[ItemT]) -> BatchT:
    if len(batch) != 1:
        raise ValueError(f"GSImageDataset only supports batch size 1, got {len(batch)}")
    return batch[0]


@dataclass
class GSImageDatasetConfig:
    camera_cache_device: str|None = "cuda"
    image_cache_device: str|None = "cuda"

DatasetCfgT = TypeVar("DatasetCfgT", bound=GSImageDatasetConfig)

class GSImageDataset(Dataset, Generic[DatasetCfgT]):
    cameras: Cameras
    cfg: DatasetCfgT
    
    def __init__(self, cameras: Cameras, cfg: DatasetCfgT):
        self.cameras = cameras
        self.cfg = cfg
        
        if (device := self.cfg.camera_cache_device) is not None:
            self.cameras = self.cameras.to(torch.device(device))
    
    def __len__(self):
        return len(self.cameras)
    
    def get_camera(self, index: int) -> Camera:
        return self.cameras[index]
    
    def __getitem__(self, index: int) -> ItemT:
        raise NotImplementedError("GSImageDataset is an abstract class. Please implement __getitem__ method.")

