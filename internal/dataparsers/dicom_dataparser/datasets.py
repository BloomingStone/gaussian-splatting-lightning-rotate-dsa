"""Dataset builder serving DICOM projection frames from memory.

Unlike :class:`internal.dataparsers.xray_dataparser.datasets.ImagesDataset` the
frames are not read from individual PNG files: the whole sequence is decoded once
into :attr:`internal.dataparsers.dicom_dataparser.meta.DicomMeta.pixel_data` and
served as grayscale ``(1, H, W)`` tensors, mirroring
:class:`internal.dataparsers.xray_dataparser.datasets.TiffDataset`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch

from ...cameras import Cameras
from ..dataparser import DatasetBuilder, GSDataset, ImageItemT, ItemT, Stage
from .meta import DicomMeta


@dataclass
class DicomDatasetConfig:
    """Options for :class:`DicomDataset`.

    Note: jsonargparse does not support empty dataclasses, so at least one field
    must be declared here.
    """

    channels: Literal[1, 3] = 3
    """Channels of the GT frames.

    ``3`` replicates the grayscale projection across RGB (like
    :class:`internal.dataparsers.xray_dataparser.datasets.ImagesDataset`). This
    is the default because the ``SaveImage`` callback concatenates the GT with
    the renderer's RGB outputs along the width axis, so the channel counts must
    match. Use ``1`` for a genuinely single-channel target (the metrics slice to
    one channel either way, but validation images will then fail to save)."""

    image_uint8: bool = False
    """Return ``uint8`` GT frames in ``[0, 255]`` instead of ``float32`` in
    ``[0, 1]``. Mirrors ``ImagesDatasetConfig``; the metrics/losses expect
    floats, so leave it ``False`` unless the model handles integer images."""


class DicomDataset(GSDataset):
    """Serve normalized DICOM frames as GT projections."""

    cfg: DicomDatasetConfig
    meta: DicomMeta
    source_indices: list[int]

    def __init__(
        self,
        cameras: Cameras,
        meta: DicomMeta,
        cfg: DicomDatasetConfig,
        source_indices: list[int],
    ):
        if meta.pixel_data is None:
            raise ValueError("DicomMeta.pixel_data is None; load the DICOM before building the dataset")

        self.cfg = cfg
        self.meta = meta
        self.source_indices = list(source_indices)

        # Keep cameras on CPU in dataset worker processes to avoid CUDA init.
        self.cameras = cameras.to(torch.device("cpu"))
        # (nf, H, W) float32 in [0, 1]; shared with the meta, not copied.
        frames = torch.from_numpy(cast(np.ndarray, meta.pixel_data)).float().unsqueeze(1)
        if cfg.channels == 3:
            # View, not a copy: stride-0 along the channel dimension.
            frames = frames.expand(-1, 3, -1, -1)
        if cfg.image_uint8:
            frames = (frames * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
        self.frames = frames

    def __len__(self) -> int:
        return len(self.source_indices)

    @property
    def image_names(self) -> list[str]:
        stem = self.meta.dcm_path.name if self.meta.dcm_path is not None else "dicom"
        return [f"{stem}#{idx}" for idx in self.source_indices]

    def get_image(self, index: int) -> torch.Tensor:
        """Return the ``(C, H, W)`` GT projection for dataset item ``index``."""
        return self.frames[self.source_indices[index]]

    def __getitem__(self, index: int) -> ItemT:
        return ItemT(
            camera=self.cameras[index],
            image=ImageItemT(
                image_name=self.image_names[index],
                gt_image=self.get_image(index),
                mask=None,
            ),
            extra_data=None,
        )


@dataclass
class DicomDatasetBuilder(DatasetBuilder[DicomMeta, DicomDataset]):
    dataset_config: DicomDatasetConfig = field(default_factory=DicomDatasetConfig)

    def build_dataset(
        self,
        data_dir: Path,
        cameras: Cameras,
        meta: DicomMeta,
        indices: list[int],
        stage: Stage,
    ) -> DicomDataset:
        del data_dir, stage
        return DicomDataset(
            cameras=cameras.get_from_indices(indices),
            meta=meta,
            cfg=self.dataset_config,
            source_indices=indices,
        )
