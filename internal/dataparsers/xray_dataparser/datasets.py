from typing import Callable, NamedTuple, Literal, cast
from pathlib import Path
from dataclasses import dataclass, field

import torch
from PIL import Image
import numpy as np
import tifffile as tiff

from ...cameras import Cameras
from ...utils.frangi import frangi_mask, frangi_vesselness
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
    image_uint8: bool = False
    
    # soft-mask weight:  w(x) = 1  inside GT mask,
    #                    w(x) = exp(-d²/2σ²)  outside GT mask
    soft_mask_sigma: float = 20.0


@dataclass
class FrangiImagesDatasetConfig(ImagesDatasetConfig):
    frangi_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0)
    frangi_beta: float = 0.5
    frangi_gamma: float = 15.0
    frangi_black_ridges: bool = True
    frangi_fusion: Literal["max", "soft"] = "soft"
    frangi_threshold: float = 0.2
    frangi_dilation_radius: int = 3
    frangi_closing_radius: int = 3
    frangi_eps: float = 1e-6


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

    def __len__(self):
        return len(self.image_paths)

    @property
    def image_names(self) -> list[str]:
        return [path.name for path in self.image_paths]

    def get_image(self, index: int) -> torch.Tensor:
        """ Return image tensor in uint8 format, shape (3, H, W) """
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
        extra: dict[str, torch.Tensor] = {}
        if self.other_data_closure is not None:
            res = self.other_data_closure(self.source_indices[index])
            extra.update(res)

        # soft mask weight based on ground-truth mask
        mask = self.get_mask(index)
        if mask is not None:
            extra["soft_mask_weight"] = self._compute_soft_mask_weight(
                mask[0], self.cfg.soft_mask_sigma,
            )

        return extra if extra else None

    @staticmethod
    def _compute_soft_mask_weight(mask: torch.Tensor, sigma: float) -> torch.Tensor:
        """mask: (H, W) bool. Returns (H, W) float: 1 inside, exp(-d²/2σ²) outside."""
        import numpy as np
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            # fallback: just return the mask itself as weight
            return mask.float()

        mask_np = mask.cpu().numpy().astype(bool)
        # distance from each pixel to nearest True (mask) pixel
        dist = cast(np.ndarray, distance_transform_edt(~mask_np))
        weight = np.where(mask_np, 1.0, np.exp(-dist ** 2 / (2.0 * sigma ** 2)))
        return torch.from_numpy(weight).float().to(mask.device)

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
        stage: Stage,
    ) -> ImagesDataset:
        del meta, stage
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


class FrangiImagesDataset(ImagesDataset):
    cfg: FrangiImagesDatasetConfig

    def __init__(
        self,
        cameras: Cameras,
        cfg: FrangiImagesDatasetConfig,
        image_paths: list[Path],
        masks_paths: list[Path] | None = None,
        other_data_closure: Callable[[int], dict[str, torch.Tensor]] | None = None,
        source_indices: list[int] | None = None,
    ):
        super().__init__(
            cameras=cameras,
            cfg=cfg,
            image_paths=image_paths,
            masks_paths=masks_paths,
            other_data_closure=other_data_closure,
            source_indices=source_indices,
        )

    @staticmethod
    def _normalize_01(t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        lo, hi = t.amin(), t.amax()
        if hi > lo:
            return (t - lo) / (hi - lo)
        return torch.zeros_like(t)

    def _compute_weight_map(self, gray_image: torch.Tensor) -> torch.Tensor:
        """Return raw Frangi vesselness (H, W) normalised to [0, 1]."""
        resp = frangi_vesselness(
            image=gray_image,
            sigmas=self.cfg.frangi_sigmas,
            beta=self.cfg.frangi_beta,
            gamma=self.cfg.frangi_gamma,
            black_ridges=self.cfg.frangi_black_ridges,
            fusion=self.cfg.frangi_fusion,
            eps=self.cfg.frangi_eps,
        )  # (1, H, W)
        return self._normalize_01(resp)[0]  # (H, W)

    def get_extra_data(self, index: int) -> dict[str, torch.Tensor] | None:
        extra = super().get_extra_data(index) or {}
        image = self._to_gray(self.get_image(index))
        if image.max() > 1.0:
            image = image / 255.0

        vesselness = frangi_vesselness(
            image=image,
            sigmas=self.cfg.frangi_sigmas,
            beta=self.cfg.frangi_beta,
            gamma=self.cfg.frangi_gamma,
            black_ridges=self.cfg.frangi_black_ridges,
            fusion=self.cfg.frangi_fusion,
            eps=self.cfg.frangi_eps,
        )  # (1, H, W)
        extra["weight_map"] = self._normalize_01(vesselness)[0]
        extra["weight_frangi"] = extra["weight_map"]  # alias for convenience

        # Frangi-derived binary mask (thresholded + morph)
        fmask = frangi_mask(
            vesselness=vesselness,
            threshold=self.cfg.frangi_threshold,
            dilation_radius=self.cfg.frangi_dilation_radius,
            closing_radius=self.cfg.frangi_closing_radius,
        )
        if fmask.dim() == 2:
            fmask = fmask.unsqueeze(0)
        fmask_bool = fmask.expand(3, -1, -1).bool()
        extra["frangi_mask"] = fmask_bool

        # soft weight based on frangi mask
        extra["frangi_soft_mask"] = self._compute_soft_mask_weight(
            fmask_bool[0], self.cfg.soft_mask_sigma,
        )
        
        mask = self.get_mask(index)
        if mask is not None:
            extra["soft_mask_weight"] = self._compute_soft_mask_weight(
                mask[0], self.cfg.soft_mask_sigma,
            )

        return extra

    @staticmethod
    def _to_gray(image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        if image.ndim == 2:
            return image.unsqueeze(0)
        if image.ndim != 3:
            raise ValueError(f"Unsupported image shape for Frangi mask: {image.shape}")
        if image.shape[0] == 1:
            return image
        return image.mean(dim=0, keepdim=True)


@dataclass
class FrangiImagesDatasetBuilder(ImagesDatasetBuilder):
    dataset_config: FrangiImagesDatasetConfig = field(default_factory=FrangiImagesDatasetConfig)

    def build_dataset(
        self,
        data_dir: Path,
        cameras: Cameras,
        meta: XRayMeta,
        indices: list[int],
        stage: Stage,
    ) -> FrangiImagesDataset:
        del meta, stage
        all_image_paths = sorted((data_dir / self.image_dir_name).glob(self.image_suffix))
        image_paths = [all_image_paths[i] for i in indices]

        other_data_closure = None
        if self.use_depth_map:
            depth_map_path = data_dir / self.depth_map_filename
            if depth_map_path.exists():
                depth_npy = np.load(depth_map_path)["arr_0"]

                def get_depth_closure(index: int) -> dict[str, torch.Tensor]:
                    return {"depth": torch.from_numpy(depth_npy[index]).float()}

                other_data_closure = get_depth_closure
        
        masks_paths = None
        mask_dir = data_dir / self.mask_dir_name
        if mask_dir.exists():
            all_mask_paths = sorted(mask_dir.glob(self.image_suffix))
            if len(all_mask_paths) > 0:
                masks_paths = [all_mask_paths[i] for i in indices]

        return FrangiImagesDataset(
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
        stage: Stage,
    ) -> TiffDataset:
        del meta, stage
        return TiffDataset(
            cameras=cameras.get_from_indices(indices),
            cfg=self.dataset_config,
            tiff_path=self._resolve_tiff_path(data_dir),
            source_indices=indices,
        )
