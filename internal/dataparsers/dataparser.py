from pathlib import Path
from typing import Tuple, Optional, Callable, Any
from dataclasses import dataclass
from torch import Tensor

import numpy as np
import torch

from ..cameras import Cameras
from ..instantiate_config import Instantiable


@dataclass
class ImageSet:
    image_names: list[str]

    image_paths: list[Path]
    """ Full path to the image file """

    cameras: Cameras
    """ Camera intrinscis and extrinsics """

    depth_paths: Optional[list[Path]] = None
    """ Full path to the depth file """

    mask_paths: Optional[list[Path]] = None
    """ Full path to the mask file """

    extra_data: Optional[list[Any]] = None

    extra_data_processor: Optional[Callable[[Any], Tensor]] = None

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        assert self.mask_paths is not None and self.extra_data is not None
        return self.image_names[index], self.image_paths[index], self.mask_paths[index], self.cameras[index], self.extra_data[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def _return_input(i):
        return i


@dataclass
class PointCloud:
    xyz: np.ndarray  # float

    rgb: np.ndarray  # uint8, in [0, 255]


@dataclass
class DataParserOutputs:
    train_set: ImageSet

    val_set: ImageSet

    test_set: ImageSet

    point_cloud: PointCloud

    camera_extent: Optional[float] = None

    def __post_init__(self):
        if self.camera_extent is None:
            camera_centers = self.train_set.cameras.camera_center
            average_camera_center = torch.mean(camera_centers, dim=0)
            camera_distance = torch.linalg.norm(camera_centers - average_camera_center, dim=-1)
            max_distance = torch.max(camera_distance)
            self.camera_extent = float(max_distance * 1.1)


class DataParser:
    def get_outputs(self) -> DataParserOutputs:
        """
        :return: [training set, validation set, point cloud]
        """

        raise NotImplementedError


@dataclass
class DataParserConfig(Instantiable):
    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        
        raise NotImplementedError
