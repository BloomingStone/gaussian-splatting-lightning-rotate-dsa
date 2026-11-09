from dataclasses import dataclass, field
from pathlib import Path

from ..dataparser import DataParser, DataParserBuilder, DataParserOutputs
from .meta import XRayMetaLoader
from .cloud_parsers import (
    XRayCloudParser,
    RandomCloudParser,
)
from .cameras_builder import RotateXRayCamerasBuilder
from .datasets import (
    ImagesDatasetBuilder,
    FrangiImagesDatasetBuilder,
    TiffDatasetBuilder,
    ImagesDatasetConfig,
)
from .splitters import ReconstructionSpliter, XRaySpliter
from .grid_val import GridValLoader, GridImagesDatasetBuilder
from ...visualizers.gamma_visulizer import GammaVisualizer


@dataclass
class XRayGridValDataParser(DataParser):
    """DataParser that uses the grid multi-view directory as val/test set."""

    grid_val_loader: GridValLoader = field(default_factory=GridValLoader)
    grid_dataset_config: ImagesDatasetConfig = field(
        default_factory=lambda: ImagesDatasetConfig(weight_source="none")
    )

    def get_outputs(self, data_dir: Path) -> DataParserOutputs:
        outputs = super().get_outputs(data_dir)

        grid_builder = GridImagesDatasetBuilder(
            grid_val_loader=self.grid_val_loader,
            dataset_config=self.grid_dataset_config,
        )
        grid_val = grid_builder.build_dataset(data_dir, None, self.meta, [], "val")
        outputs.datasets["val"] = grid_val
        outputs.datasets["test"] = grid_val
        return outputs


@dataclass
class XRayDataParserBuilder(DataParserBuilder):
    meta_loader: XRayMetaLoader =  field(default_factory=XRayMetaLoader)
    cloud_parser: XRayCloudParser = field(default_factory=RandomCloudParser)
    spliter: XRaySpliter =  field(default_factory=ReconstructionSpliter)
    cameras_builder: RotateXRayCamerasBuilder =  field(default_factory=RotateXRayCamerasBuilder)
    dataset_builder: ImagesDatasetBuilder|FrangiImagesDatasetBuilder|TiffDatasetBuilder = field(default_factory=ImagesDatasetBuilder)
    filter_visible_points: bool = True

    use_grid_val: bool = False
    """When True, the grid multi-view directory is used as the val/test set."""

    grid_val_loader: GridValLoader = field(default_factory=GridValLoader)
    grid_dataset_config: ImagesDatasetConfig = field(
        default_factory=lambda: ImagesDatasetConfig(weight_source="none")
    )

    def build(self) -> DataParser:
        visualizer = GammaVisualizer(gamma=0.1) if isinstance(self.dataset_builder, TiffDatasetBuilder) else None

        if self.use_grid_val:
            return XRayGridValDataParser(
                meta_loader=self.meta_loader,
                cloud_parser=self.cloud_parser,
                spliter=self.spliter,
                cameras_builder=self.cameras_builder,
                dataset_builder=self.dataset_builder,
                filter_visible_points=self.filter_visible_points,
                gt_image_visualizer=visualizer,
                grid_val_loader=self.grid_val_loader,
                grid_dataset_config=self.grid_dataset_config,
            )

        return DataParser(
            meta_loader=self.meta_loader,
            cloud_parser=self.cloud_parser,
            spliter=self.spliter,
            cameras_builder=self.cameras_builder,
            dataset_builder=self.dataset_builder,
            filter_visible_points=self.filter_visible_points,
            gt_image_visualizer=visualizer,
        )
