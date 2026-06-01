from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union, cast

import numpy as np
import nibabel as nib

from ..dataparser import DataParser, CloudParser, DataParserBuilder
from .meta import XRayMetaLoader, XRayMeta
from .cloud_parsers import (
    UniformCloudParser,
    RandomCloudParser,
    BallRandomCloudParser,
    LabelCloudParser,
    CentralLineCloudParser,
)
from .cameras_builder import RotateXRayCamerasBuilder
from .datasets import (
    ImagesDatasetBuilder,
    FrangiMaskImagesDatasetBuilder,
    TiffDatasetBuilder,
    ImagesDataset,
    TiffDataset,
)
from .splitters import ReconstructionSpliter, RenderNewViewsSpliter
from .cloud_parsers import UniformCloudParser, RandomCloudParser, BallRandomCloudParser, LabelCloudParser, CentralLineCloudParser, FdkCloudParser

XRayCloudParserType = Union[
    UniformCloudParser,
    RandomCloudParser,
    BallRandomCloudParser,
    LabelCloudParser,
    CentralLineCloudParser,
    FdkCloudParser,
]

@dataclass
class XRayDataParserBuilder(DataParserBuilder):
    meta_loader: XRayMetaLoader =  field(default_factory=XRayMetaLoader)
    cloud_parser: XRayCloudParserType = field(default_factory=RandomCloudParser)
    spliter: ReconstructionSpliter|RenderNewViewsSpliter =  field(default_factory=ReconstructionSpliter)
    cameras_builder: RotateXRayCamerasBuilder =  field(default_factory=RotateXRayCamerasBuilder)
    dataset_builder: ImagesDatasetBuilder|FrangiMaskImagesDatasetBuilder|TiffDatasetBuilder = field(default_factory=ImagesDatasetBuilder)
    filter_visible_points: bool = True
    label_3d_filename: str = "coronary_label.nii.gz"
    
    def build(self):
        return XRayDataParser(
            meta_loader=self.meta_loader,
            cloud_parser=self.cloud_parser,
            spliter=self.spliter,
            cameras_builder=self.cameras_builder,
            dataset_builder=self.dataset_builder,
            filter_visible_points=self.filter_visible_points,
            label_3d_filename=self.label_3d_filename,
        )


class XRayDataParser(DataParser[XRayMeta, ImagesDataset|TiffDataset]):
    label_3d: np.ndarray | None
    label_3d_affine: np.ndarray | None
    
    
    def __init__(
        self,
        meta_loader = XRayMetaLoader(),
        cloud_parser: CloudParser = RandomCloudParser(num_points=100_000),
        spliter: ReconstructionSpliter|RenderNewViewsSpliter = ReconstructionSpliter(),
        cameras_builder: RotateXRayCamerasBuilder = RotateXRayCamerasBuilder(),
        dataset_builder: ImagesDatasetBuilder|FrangiMaskImagesDatasetBuilder|TiffDatasetBuilder = ImagesDatasetBuilder(),
        filter_visible_points: bool = True,
        label_3d_filename: str = "coronary_label.nii.gz",
    ):
        if isinstance(cloud_parser, LabelCloudParser):
            cloud_parser.label_nii_filename = label_3d_filename
        
        super().__init__(
            meta_loader=meta_loader,
            cloud_parser=cloud_parser,
            spliter=spliter,
            cameras_builder=cameras_builder,
            dataset_builder=dataset_builder,
            filter_visible_points=filter_visible_points,
        )
        self.label_nii_filename = label_3d_filename
        
        label_nii_path = Path(label_3d_filename)
        if label_nii_path.is_file():
            nii_img = cast(nib.Nifti1Image, nib.load(label_nii_path))
            self.label_3d = nii_img.get_fdata().astype(np.uint8)
            self.label_3d_affine = nii_img.affine
        else:
            self.label_3d = None
            self.label_3d_affine = None
        

    def get_outputs(self, data_dir: Path):
        return super().get_outputs(data_dir)
