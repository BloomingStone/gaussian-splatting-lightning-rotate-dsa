from dataclasses import dataclass
from typing import Generic

import numpy as np
import torch

from ..instantiate_config import Instantiable
from ..datasets.gs_dataset import GSImageDataset, GSImageDatasetConfig, DatasetCfgT


@dataclass
class PointCloud:
    xyz: np.ndarray  # float

    rgb: np.ndarray  # uint8, in [0, 255]


@dataclass
class DataParserOutputs:
    train_set: GSImageDataset

    val_set: GSImageDataset

    test_set: GSImageDataset

    point_cloud: PointCloud

    camera_extent: None|float = None

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
class DataParserConfig(Instantiable, Generic[DatasetCfgT]):
    dataset_config: DatasetCfgT
    
    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        
        raise NotImplementedError
