from dataclasses import dataclass
from typing import Protocol, NamedTuple, TypeVar, Literal, Generic
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..cameras import Camera, Cameras


MetaT = TypeVar("MetaT")

MetaT_co = TypeVar("MetaT_co", covariant=True)
MetaT_contra = TypeVar("MetaT_contra", contravariant=True)

class MetaLoader(Protocol[MetaT_co]):
    """Loader protocol for DataParser."""
    def load(self, data_dir: Path) -> MetaT_co:
        """Load data from the given path."""
        ...


class CamerasBuilder(Protocol[MetaT_contra]):
    """Protocol for building Cameras from Meta."""
    def build_cameras(self, meta: MetaT_contra) -> Cameras:
        """Build Cameras from Meta."""
        ...


@dataclass
class PointCloud:
    xyz: np.ndarray  # float
    feature: np.ndarray
    
    @property
    def rgb(self) -> np.ndarray:
        if self.feature.shape[1] == 3:
            return self.feature
        else:
            raise ValueError("Feature does not have 3 channels, cannot be interpreted as RGB.")


Stage = Literal["train", "val", "test"]


class CloudParser(Protocol[MetaT_contra]):
    num_points: int
    
    """Protocol for point cloud parsers."""
    def get_point_cloud(self, data_dir: Path, meta: MetaT_contra, splits: None|dict[Stage, list[int]]=None) -> PointCloud:
        """Parse the data directory and return a point cloud."""
        ...


class Spliter(Protocol[MetaT_contra]):
    """Protocol for dataset splitters."""
    def split(self, data_dir: Path, meta: MetaT_contra) -> dict[Stage, list[int]]:
        """Split the dataset into train/val/test indices."""
        ...


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
    return GSDataset.batch_one_collate_fn(batch)


class GSDataset(Dataset):
    data_dir: Path
    cameras: Cameras
    
    @staticmethod
    def batch_one_collate_fn(batch: list[ItemT]) -> BatchT:
        if len(batch) != 1:
            raise ValueError(f"GSDataset only supports batch size 1, got {len(batch)}")
        return batch[0]

    def __len__(self):
        raise NotImplementedError("GSDataset is an abstract class. Please implement __len__ method.")
    
    def __getitem__(self, index: int) -> ItemT:
        raise NotImplementedError("GSDataset is an abstract class. Please implement __getitem__ method.")


DatasetT = TypeVar("DatasetT", bound=GSDataset, covariant=True)

class DatasetBuilder(Protocol[MetaT_contra, DatasetT]):
    def build_dataset(
        self,
        data_dir: Path,
        cameras: Cameras,
        meta: MetaT_contra,
        indices: list[int],
        split: Stage,
    ) -> DatasetT:
        ...


@dataclass
class DataParserOutputs(Generic[MetaT]):
    meta: MetaT
    
    datasets: dict[Stage, GSDataset]

    point_cloud: PointCloud

    camera_extent: None|float = None

    def __post_init__(self):
        if self.camera_extent is None:
            camera_centers = self.datasets["train"].cameras.camera_center
            average_camera_center = torch.mean(camera_centers, dim=0)
            camera_distance = torch.linalg.norm(camera_centers - average_camera_center, dim=-1)
            max_distance = torch.max(camera_distance)
            self.camera_extent = float(max_distance * 1.1)

    @property
    def train_set(self) -> GSDataset:
        return self.datasets["train"]

    @property
    def val_set(self) -> GSDataset:
        return self.datasets["val"]

    @property
    def test_set(self) -> GSDataset:
        return self.datasets["test"]


def _get_visible_mask(points: np.ndarray, cameras: Cameras) -> np.ndarray:
    points_ = torch.from_numpy(points)
    device = torch.device("cpu")
    N = points.shape[0]
    cameras = cameras.to(device)
    full_projection = cameras.full_projection

    # (N,4)
    points_h = torch.cat([points_, torch.ones(N, 1, device=device)], dim=1).float()

    clip = torch.einsum("ni,mij->mnj", points_h, full_projection)  # (M,N,4)
    ndc = clip[..., :3] / (clip[..., 3:4] + 1e-8)   # (M,N,3), x/z, y/z, 1/z
    x = ndc[..., 0]
    y = ndc[..., 1]
    z = ndc[..., 2]
    in_frustum = (
        (x >= -1) & (x <= 1) &
        (y >= -1) & (y <= 1) &
        (z >= 0) & (z <= 1)
    )
    visible_any = in_frustum.any(dim=0)
    return visible_any.cpu().numpy()


@dataclass
class DataParser(Generic[MetaT, DatasetT]):
    meta_loader: MetaLoader[MetaT]
    cloud_parser: CloudParser[MetaT]
    spliter: Spliter[MetaT]
    cameras_builder: CamerasBuilder[MetaT]
    
    dataset_builder: DatasetBuilder[MetaT, DatasetT]
    
    filter_visible_points: bool = True

    def get_outputs(self, data_dir: Path) -> DataParserOutputs[MetaT]:
        """
        :return: [training set, validation set, point cloud]
        """
        meta = self.meta_loader.load(data_dir)
        splits = self.spliter.split(data_dir, meta)
        point_cloud = self.cloud_parser.get_point_cloud(data_dir, meta, splits)
        cameras = self.cameras_builder.build_cameras(meta)
        
        if self.filter_visible_points:
            visible_mask = _get_visible_mask(point_cloud.xyz, cameras)
            point_cloud = PointCloud(
                xyz=point_cloud.xyz[visible_mask], 
                feature=point_cloud.feature[visible_mask]
            )

        return DataParserOutputs(
            meta=meta,
            datasets = {
                stage: self.dataset_builder.build_dataset(data_dir, cameras, meta, indices, stage)
                for stage, indices in splits.items()
            },
            point_cloud=point_cloud,
        )
