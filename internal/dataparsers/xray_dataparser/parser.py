from dataclasses import dataclass, field

from ..dataparser import DataParser, DataParserBuilder
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
)
from .splitters import ReconstructionSpliter, XRaySpliter


@dataclass
class XRayDataParserBuilder(DataParserBuilder):
    meta_loader: XRayMetaLoader =  field(default_factory=XRayMetaLoader)
    cloud_parser: XRayCloudParser = field(default_factory=RandomCloudParser)
    spliter: XRaySpliter =  field(default_factory=ReconstructionSpliter)
    cameras_builder: RotateXRayCamerasBuilder =  field(default_factory=RotateXRayCamerasBuilder)
    dataset_builder: ImagesDatasetBuilder|FrangiImagesDatasetBuilder|TiffDatasetBuilder = field(default_factory=ImagesDatasetBuilder)
    filter_visible_points: bool = True
    
    def build(self):
        return DataParser(
            meta_loader=self.meta_loader,
            cloud_parser=self.cloud_parser,
            spliter=self.spliter,
            cameras_builder=self.cameras_builder,
            dataset_builder=self.dataset_builder,
            filter_visible_points=self.filter_visible_points,
        )
