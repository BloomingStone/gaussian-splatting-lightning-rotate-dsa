from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Any, Optional, override
import numpy as np
import torch
import torch.utils.data
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tqdm import tqdm
import tifffile as tiff

from ..cameras import Camera, Cameras
from ..dataparsers import  ImageSet
from .vanilla_dataset import DataModule, CacheDataLoader

def reset_cameras(cameras: Cameras, width, height) -> Cameras:
    n_camras = len(cameras)
    return Cameras(
        R = cameras.R,
        T = cameras.T,
        fx = cameras.fx,
        fy = cameras.fy,
        cx = cameras.cx,
        cy = cameras.cy,
        width = torch.ones(n_camras) * width,
        height = torch.ones(n_camras) * height,
        camera_type=cameras.camera_type,
        time=cameras.time,
        zfar=cameras.zfar
    )

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_set: ImageSet,
            camera_device: torch.device | None = None,
            image_device: torch.device | None = None,
            val_min: float = 0,
            val_max: float|None = None,
            roi = (
                (200,200),  # top left
                (450,400),  # bottum right
            )
    ) -> None:
        super().__init__()
        self.image_set = image_set

        if camera_device is None:
            camera_device = torch.device("cpu")
        if image_device is None:
            image_device = torch.device("cpu")
        self.camera_device = camera_device
        self.image_device = image_device

        tiff_path = self.image_set.image_paths[0]
        for p in self.image_set.image_paths:
            assert p == tiff_path, "All image paths must be the same for tiff dataset"
        tiff_data = tiff.imread(self.image_set.image_paths[0])
        assert tiff_data.dtype == np.uint16, f"TIFF data must be uint16, got {tiff_data.dtype}"
        
        if roi is not None:
            ((x0, y0), (x1, y1)) = roi
            width = x1 - x0
            height = y1 - y0
            tiff_data = tiff_data[:, y0:y1, x0:x1]
            image_set.cameras = reset_cameras(image_set.cameras, width, height)
        
        self.image_cameras: list[Camera] = [i.to_device(camera_device) for i in image_set.cameras]  # store undistorted camera
        self.roi = roi
        
        # cut roi and then normalize to [0, 1]
        val_max_ = tiff_data.max() if val_max is None else val_max
        tiff_data = (tiff_data - val_min) / (val_max_ - val_min)
        self.val_min = val_min
        self.val_max = val_max_
        
        tiff_data = tiff_data.astype(np.float32)
        self.tiff_data = tiff_data

    def __len__(self):
        return len(self.image_set)

    def get_image(self, index) -> Tuple[str, torch.Tensor, Optional[torch.Tensor]]:
        image_index = self.image_set.image_names[index]
        image = torch.from_numpy(self.tiff_data[image_index]).to(self.image_device)  # [H, W]
        image = image.unsqueeze(0).repeat(3, 1, 1)
        return str(image_index), image, None

    def get_extra_data(self, index):
        if self.image_set.extra_data_processor is None or self.image_set.extra_data is None:
            return None
        return self.image_set.extra_data_processor(self.image_set.extra_data[index])

    def __getitem__(self, index) -> Tuple[Camera, Tuple, Any]:
        return self.image_cameras[index], self.get_image(index), self.get_extra_data(index)

class TiffDataModule(DataModule):
    def __init__(
        self,
        val_min: float = 0,
        val_max: float|None = None,
        roi = (
            (200, 150),  # top left
            (450, 400),  # bottom right
        ), *args, **kwargs
    ):
        self.val_min = val_min
        self.val_max = val_max
        self.roi = roi
        super().__init__(*args, **kwargs)
    
    @override
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        world_size = getattr(self.trainer, "world_size", 1)
        global_rank = getattr(self.trainer, "global_rank", 0)
        return CacheDataLoader(
            Dataset(
                self.dataparser_outputs.train_set,
                camera_device=self.camera_device,
                image_device=self.image_device,
                val_min=self.val_min,
                val_max=self.val_max,
                roi=self.roi
            ),
            max_cache_num=self.hparams["train_max_num_images_to_cache"],
            shuffle=True,
            seed=torch.initial_seed() + self.global_rank,  # seed with global rank
            num_workers=self.hparams["num_workers"],
            distributed=self.hparams["distributed"],
            world_size=world_size,
            global_rank=global_rank,
            async_caching=self.hparams["async_caching"],
        )

    @override
    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.hparams["val_on_train"] is True:
            image_set = self.dataparser_outputs.train_set
        else:
            image_set = self.dataparser_outputs.test_set
        return CacheDataLoader(
            Dataset(
                image_set,
                camera_device=self.camera_device,
                image_device=self.image_device,
                val_min=self.val_min,
                val_max=self.val_max,
                roi=self.roi
            ),
            max_cache_num=self.hparams["test_max_num_images_to_cache"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
        )

    @override
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.hparams["val_on_train"] is True:
            image_set = self.dataparser_outputs.train_set
        else:
            image_set = self.dataparser_outputs.val_set
        return CacheDataLoader(
            Dataset(
                image_set,
                camera_device=self.camera_device,
                image_device=self.image_device,
                val_min=self.val_min,
                val_max=self.val_max,
                roi=self.roi
            ),
            max_cache_num=self.hparams["val_max_num_images_to_cache"],
            shuffle=False,
            num_workers=self.hparams["num_workers"],
        )
