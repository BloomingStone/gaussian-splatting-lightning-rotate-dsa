from pathlib import Path
from typing import Literal, Union, cast

import numpy as np
import nibabel as nib

from ..dataparser import DataParser
from .meta import XRayMetaLoader
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
    TiffDatasetBuilder,
)
from .splitters import ReconstructionSpliter, RenderNewViewsSpliter


InitPointCloudMode = Literal["uniform", "random", "random-ball", "label", "central-line"]
DatasetType = Literal["images", "tiff"]
ParserMode = Literal["reconstruction", "render-new-views"]


class XRayDataParser(DataParser):
    label_3d: np.ndarray | None
    label_3d_affine: np.ndarray | None
    
    
    def __init__(
        self,
        meta_loader: XRayMetaLoader = XRayMetaLoader(),
        cloud_parser: Union[
            UniformCloudParser, RandomCloudParser, BallRandomCloudParser, LabelCloudParser, CentralLineCloudParser
        ] = RandomCloudParser(num_points=100_000),
        spliter: ReconstructionSpliter|RenderNewViewsSpliter = ReconstructionSpliter(),
        cameras_builder: RotateXRayCamerasBuilder = RotateXRayCamerasBuilder(),
        dataset_builder: ImagesDatasetBuilder|TiffDatasetBuilder = ImagesDatasetBuilder(),
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
