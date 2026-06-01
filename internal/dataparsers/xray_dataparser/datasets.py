from typing import Callable, cast, NamedTuple
from pathlib import Path
from dataclasses import dataclass, field

import torch
from PIL import Image
import numpy as np
import tifffile as tiff

from ...cameras import Cameras
from ..dataparser import GSDataset, DatasetBuilder, ItemT, ImageItemT, Stage
from .meta import XRayMeta


PixelPosition = NamedTuple("PixelPosition", [
    ("x", int),
    ("y", int),
])


ROI = NamedTuple("ROI", [
    ("top_left", PixelPosition),
    ("bottom_right", PixelPosition),
])


@dataclass
class ImagesDatasetConfig:
    # camera cached on CPU by default; do not move cameras to CUDA in dataset
    # if cache_image is True, cache images/masks as cpu tensors in dataset to avoid IO
    cache_image: bool = True
    image_uint8: bool = False


@dataclass
class TiffDatasetConfig:
    roi: ROI = ROI(
        top_left=PixelPosition(200, 200),
        bottom_right=PixelPosition(450, 400),
    )
    val_min: float | None = None
    val_max: float | None = None


def apply_roi(
    tiff_data: np.ndarray,
    cameras: Cameras,
    roi: ROI,
) -> tuple[np.ndarray, Cameras]:
    x0, y0 = roi.top_left
    x1, y1 = roi.bottom_right

    _, height, width = tiff_data.shape
    assert 0 <= x0 < x1 <= width, f"Invalid ROI x coordinates: {x0}, {x1}, image width: {width}"
    assert 0 <= y0 < y1 <= height, f"Invalid ROI y coordinates: {y0}, {y1}, image height: {height}"

    tiff_data = tiff_data[:, y0:y1, x0:x1]

    cropped_width = x1 - x0
    cropped_height = y1 - y0
    cropped_cameras = Cameras.build(
        idx=cameras.idx,
        R=cameras.R,
        T=cameras.T,
        fx=cameras.fx,
        fy=cameras.fy,
        cx=cameras.cx - x0,
        cy=cameras.cy - y0,
        width=torch.ones_like(cameras.width).to(cameras.width) * cropped_width,
        height=torch.ones_like(cameras.height).to(cameras.height) * cropped_height,
        time=cameras.time,
        phase=cameras.phase,
        zfar=cameras.zfar,
        znear=cameras.znear,
    )
    return tiff_data, cropped_cameras



class ImagesDataset(GSDataset):
    cfg: ImagesDatasetConfig

    image_paths: list[Path]
    masks_paths: list[Path] | None
    other_data_closure: Callable[[int], dict[str, torch.Tensor]] | None
    source_indices: list[int]

    cached_images: torch.Tensor | None
    cached_masks: torch.Tensor | None

    def __init__(
        self,
        cameras: Cameras,
        cfg: ImagesDatasetConfig,
        image_paths: list[Path],
        masks_paths: list[Path] | None = None,
        other_data_closure: Callable[[int], dict[str, torch.Tensor]] | None = None,
        source_indices: list[int] | None = None,
    ):
        self.cfg = cfg
        self.cameras = cameras
        self.image_paths = image_paths
        self.masks_paths = masks_paths
        self.other_data_closure = other_data_closure
        self.source_indices = list(range(len(image_paths))) if source_indices is None else source_indices

        # keep cameras on CPU in dataset worker processes to avoid CUDA init
        self.cameras = self.cameras.to(torch.device("cpu"))

        self.cached_images = None
        self.cached_masks = None

        def _try_cache_images():
            if not self.cfg.cache_image:
                return

            L = len(self)
            self.cached_images = torch.stack([self.get_image(i) for i in range(L)])

            if self.masks_paths is None:
                return

            masks = [self.get_mask(i) for i in range(L)]
            is_masks_valid = [mask is not None for mask in masks]
            assert all(is_masks_valid), "Some masks are invalid, cannot cache masks"

            self.cached_masks = torch.stack(cast(list[torch.Tensor], masks))

        _try_cache_images()

    def __len__(self):
        return len(self.image_paths)

    @property
    def image_names(self) -> list[str]:
        return [path.name for path in self.image_paths]

    def get_image(self, index: int) -> torch.Tensor:
        """ Return image tensor in uint8 format, shape (3, H, W) """
        if self.cached_images is not None:
            return self.cached_images[index]

        image = Image.open(self.image_paths[index])
        image = np.array(image, dtype=np.uint8)
        image = torch.from_numpy(image)

        match image.shape:
            case (_, _):
                image = image.unsqueeze(-1).expand(-1, -1, 3)
            case (_, _, 1):
                image = image.expand(-1, -1, 3)
            case (_, _, 3):
                pass
            case _:
                raise ValueError(f"Unsupported image shape: {image.shape}")

        image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        if self.cfg.image_uint8:
            return image
        else:
            return image.float() / 255.0

    def get_mask(self, index: int) -> torch.Tensor | None:
        """ Return mask tensor in bool format, (3, H, W) """
        if self.cached_masks is not None:
            return self.cached_masks[index]

        if self.masks_paths is None:
            return None

        mask = Image.open(self.masks_paths[index])
        mask = np.array(mask, dtype=np.uint8)
        mask = torch.from_numpy(mask).squeeze()

        match mask.shape:
            case (_, _):
                pass
            case (_, _, 3):
                mask = mask[:, :, 0]  # take the first channel as mask
            case _:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")

        mask = mask > 0  # convert to bool mask, True is valid pixel, False is masked pixel
        mask = mask.unsqueeze(0).expand(3, -1, -1)  # (H, W) -> (3, H, W)

        return mask.bool()

    def get_extra_data(self, index: int) -> None | dict[str, torch.Tensor]:
        if self.other_data_closure is None:
            return None
        res = self.other_data_closure(self.source_indices[index])
        return {k: v for k, v in res.items()}

    def __getitem__(self, index: int) -> ItemT:
        return ItemT(
            camera=self.cameras[index],
            image=ImageItemT(
                image_name=self.image_paths[index].name,
                gt_image=self.get_image(index),
                mask=self.get_mask(index),
            ),
            extra_data=self.get_extra_data(index),
        )

@dataclass
class ImagesDatasetBuilder(DatasetBuilder[XRayMeta, ImagesDataset]):
    image_dir_name: str = "rotate_dsa"
    mask_dir_name: str = "label"
    image_suffix: str = "*.png"
    use_depth_map: bool = True
    depth_map_filename: str = "depth_map.npz"
    dataset_config: ImagesDatasetConfig = field(default_factory=ImagesDatasetConfig)

    def build_dataset(
        self,
        data_dir: Path,
        cameras: Cameras,
        meta: XRayMeta,
        indices: list[int],
        split: Stage,
    ) -> ImagesDataset:
        del meta, split
        all_image_paths = sorted((data_dir / self.image_dir_name).glob(self.image_suffix))
        image_paths = [all_image_paths[i] for i in indices]

        masks_paths = None
        mask_dir = data_dir / self.mask_dir_name
        if mask_dir.exists():
            all_mask_paths = sorted(mask_dir.glob(self.image_suffix))
            if len(all_mask_paths) > 0:
                masks_paths = [all_mask_paths[i] for i in indices]

        other_data_closure = None
        if self.use_depth_map:
            depth_map_path = data_dir / self.depth_map_filename
            if depth_map_path.exists():
                depth_npy = np.load(depth_map_path)["arr_0"]

                def get_depth_closure(index: int) -> dict[str, torch.Tensor]:
                    return {"depth": torch.from_numpy(depth_npy[index]).float()}

                other_data_closure = get_depth_closure

        return ImagesDataset(
            cameras=cameras.get_from_indices(indices),
            cfg=self.dataset_config,
            image_paths=image_paths,
            masks_paths=masks_paths,
            other_data_closure=other_data_closure,
            source_indices=indices,
        )


class TiffDataset(GSDataset):
    cfg: TiffDatasetConfig

    tiff_path: Path
    tiff_data: torch.Tensor
    source_indices: list[int]

    def __init__(
        self,
        cameras: Cameras,
        cfg: TiffDatasetConfig,
        tiff_path: Path,
        source_indices: list[int],
    ):
        self.cfg = cfg
        self.cameras = cameras
        self.tiff_path = tiff_path
        self.source_indices = source_indices

        # keep cameras on CPU in dataset
        self.cameras = self.cameras.to(torch.device("cpu"))

        tiff_data = tiff.imread(self.tiff_path).astype(np.float32)
        tiff_data, self.cameras = apply_roi(tiff_data, self.cameras, cfg.roi)

        vol_max = tiff_data.max() if cfg.val_max is None else cfg.val_max
        vol_min = tiff_data.min() if cfg.val_min is None else cfg.val_min
        tiff_data = (tiff_data - vol_min) / (vol_max - vol_min + 1e-8)

        self.tiff_data = torch.from_numpy(tiff_data)

    def __len__(self):
        return len(self.source_indices)

    @property
    def image_names(self) -> list[str]:
        return [f"{self.tiff_path.name}_{idx}" for idx in self.source_indices]

    def get_image(self, index: int) -> torch.Tensor:
        image = self.tiff_data[self.source_indices[index]].unsqueeze(0)
        return image

    def __getitem__(self, index: int) -> ItemT:
        return ItemT(
            camera=self.cameras[index],
            image=ImageItemT(
                image_name=f"{self.tiff_path.name}_{self.source_indices[index]}",
                gt_image=self.get_image(index),
                mask=None,
            ),
            extra_data=None,
        )

@dataclass
class TiffDatasetBuilder(DatasetBuilder[XRayMeta, TiffDataset]):
    base_name: str = "rotate_dsa"
    dataset_config: TiffDatasetConfig = field(default_factory=TiffDatasetConfig)

    def _resolve_tiff_path(self, data_root: Path) -> Path:
        candidates = [
            data_root / f"{self.base_name}.tiff",
            data_root / f"{self.base_name}.tif",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"TIFF file not found. Tried: {', '.join(str(path) for path in candidates)}")

    def build_dataset(
        self,
        data_dir: Path,
        cameras: Cameras,
        meta: XRayMeta,
        indices: list[int],
        split: Stage,
    ) -> TiffDataset:
        del meta, split
        return TiffDataset(
            cameras=cameras.get_from_indices(indices),
            cfg=self.dataset_config,
            tiff_path=self._resolve_tiff_path(data_dir),
            source_indices=indices,
        )
