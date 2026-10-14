"""DataParser entry point for DICOM (rotational-DSA) datasets.

Assembles a :class:`internal.dataparsers.dataparser.DataParser` from DICOM-aware
components. The cloud parsers, splitters and camera math are reused from
``xray_dataparser`` because :class:`~.meta.DicomMeta` subclasses ``XRayMeta``.

Example ``yaml`` (jsonargparse ``class_path`` form)::

    data:
      parser:
        class_path: internal.dataparsers.dicom_dataparser.parser.DicomDataParserBuilder
        init_args:
          meta_loader:
            class_path: internal.dataparsers.dicom_dataparser.meta.DicomMetaLoader
            init_args:
              dcm_filename: RXA_chest.dcm
              orientation_type: AP
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...visualizers.gamma_visulizer import GammaVisualizer
from ..dataparser import DataParser, DataParserBuilder
from ..xray_dataparser.cloud_parsers import RandomCloudParser, XRayCloudParser
from ..xray_dataparser.splitters import ReconstructionSpliter, XRaySpliter
from .cameras_builder import DicomCamerasBuilder
from .datasets import DicomDatasetBuilder
from .meta import DicomMetaLoader


@dataclass
class DicomDataParserBuilder(DataParserBuilder):
    meta_loader: DicomMetaLoader = field(default_factory=DicomMetaLoader)
    cloud_parser: XRayCloudParser = field(default_factory=RandomCloudParser)
    spliter: XRaySpliter = field(default_factory=ReconstructionSpliter)
    cameras_builder: DicomCamerasBuilder = field(default_factory=DicomCamerasBuilder)
    dataset_builder: DicomDatasetBuilder = field(default_factory=DicomDatasetBuilder)
    filter_visible_points: bool = True

    visualization_gamma: float | None = None
    """When set, install a :class:`GammaVisualizer` so saved GT frames are
    brightened for viewing (the loss target itself is unaffected)."""

    def build(self) -> DataParser:
        visualizer = (
            GammaVisualizer(gamma=self.visualization_gamma)
            if self.visualization_gamma is not None
            else None
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
