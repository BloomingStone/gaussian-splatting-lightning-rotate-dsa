from typing import Any, Callable, cast, Final
from pathlib import Path
from dataclasses import dataclass

import torch
from PIL import Image
import numpy as np

from ..cameras import Cameras
from .gs_dataset import GSImageDataset, GSImageDatasetConfig, ItemT, ImageItemT


@dataclass
class ImagesDatasetConfig(GSImageDatasetConfig):
    image_uint8: bool = False


class ImagesDataset(GSImageDataset): 
    image_paths: list[Path]
    masks_paths: list[Path]|None
    other_data_closure: Callable[[int], dict[str, torch.Tensor]]|None
    
    cached_images: torch.Tensor|None    
    cached_masks: torch.Tensor|None
    
    def __init__(
        self, 
        cameras: Cameras, 
        cfg: ImagesDatasetConfig,
        image_paths: list[Path],
        masks_paths: list[Path]|None = None,
        other_data_closure: Callable[[int], dict[str, torch.Tensor]]|None = None,
    ):
        super().__init__(cameras, cfg)
        self.image_paths = image_paths
        self.masks_paths = masks_paths
        self.other_data_closure = other_data_closure
        self.cached_images = None
        self.cached_masks = None
    
        def _try_cache_images():
            if self.cfg.image_cache_device is None:
                return
            
            device = torch.device(self.cfg.image_cache_device)
            L = len(self.image_paths)
            self.cached_images = torch.stack([self.get_image(i) for i in range(L)]).to(device)
            
            if self.masks_paths is None:
                return
            
            masks = [self.get_mask(i) for i in range(L)]
            is_masks_valid = [mask is not None for mask in masks]
            assert all(is_masks_valid), "Some masks are invalid, cannot cache masks"
            
            self.cached_masks = torch.stack(cast(list[torch.Tensor], masks)).to(device)
        
        _try_cache_images()
    
    
    def get_image(self, index: int) -> torch.Tensor:
        """ Return image tensor in uint8 format, shape (3, H, W) """
        if self.cached_images is not None:
            return self.cached_images[index]
        
        image = Image.open(self.image_paths[index])
        image = np.array(image, dtype=np.uint8)
        image = torch.from_numpy(image)
        
        match image.shape:
            case (_, _):
                image = image.unsqueeze(-1).expand(-1, -1, 3)
            case (_, _, 1):
                image = image.expand(-1, -1, 3)
            case (_, _, 3):
                pass
            case _:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        
        image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        if self.cfg.image_uint8:
            return image
        else:
            return image.float() / 255.0

    
    def get_mask(self, index: int) -> torch.Tensor|None:
        """ Return mask tensor in bool format, (3, H, W) """
        if self.cached_masks is not None:
            return self.cached_masks[index]
        
        if self.masks_paths is None:
            return None
        
        mask = Image.open(self.masks_paths[index])
        mask = np.array(mask, dtype=np.uint8)
        mask = torch.from_numpy(mask).squeeze()
        
        match mask.shape:
            case (_, _):
                pass
            case (_, _, 3):
                mask = mask[:, :, 0]  # take the first channel as mask
            case _:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")
        
        mask = mask > 0  # convert to bool mask, True is valid pixel, False is masked pixel
        mask = mask.unsqueeze(0).expand(3, -1, -1)  # (H, W) -> (3, H, W)
        
        return mask.bool()
    
    
    def get_extra_data(self, index: int) -> None|dict[str, torch.Tensor]:
        if self.other_data_closure is None:
            return None
        res = self.other_data_closure(index)
        return {k: v for k, v in res.items()}


    def __getitem__(self, index: int) -> ItemT:
        return ItemT(
            camera=self.get_camera(index),
            image=ImageItemT(
                image_name=self.image_paths[index].name,
                gt_image=self.get_image(index),
                mask=self.get_mask(index),
            ),
            extra_data=self.get_extra_data(index),
        )
