from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

import hashlib
import json

import numpy as np
import torch
import tifffile as tiff

from ..dataparsers.dataparser import DataParser, DataParserConfig, DataParserOutputs, ImageSet, PointCloud
from .rotated_xray_dataparser import (
    RotatedXRay,
    RotatedXRayDataParser,
    _filter_points_visible,
    _get_backgound_gaussian_from_xyz,
    _get_cameras,
    init_point_cloud_from_label,
)


def _resolve_tiff_path(data_root: Path, base_name: str) -> Path:
    candidates = [
        data_root / f"{base_name}.tiff",
        data_root / f"{base_name}.tif",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"TIFF file not found. Tried: {', '.join(str(path) for path in candidates)}")



@dataclass
class TiffDataParserConfig(RotatedXRay):
    
    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        return TiffDataParser(path=path, output_path=output_path, global_rank=global_rank, params=self)


class TiffDataParser(RotatedXRayDataParser):
    def __init__(
        self,
        path: str,
        output_path: str,
        global_rank: int,
        params: TiffDataParserConfig,
    ) -> None:
        super().__init__(
            path=path,
            output_path=output_path,
            global_rank=global_rank,
            params=params,
        )
        self.path = path
        self.output_path = output_path
        self.global_rank = global_rank
        self.params = params
        self._data: dict|None = None
        self._coronary_type: Literal["LCA", "RCA"]|None = params.coronary_type
        self._volume_shape: tuple[int, int, int]|None = None
        self._affine_dict: dict[Literal["LCA", "RCA", "volume"], np.ndarray]|None = None
        
    @override
    def get_outputs(self) -> DataParserOutputs:
        data_root = Path(self.path)
        json_path = data_root / f"{self.params.base_name}.json"
        tiff_path = _resolve_tiff_path(data_root, self.params.base_name)

        with open(json_path, "r") as f:
            data = json.load(f)

        self._data = data
        self._coronary_type = data["coronary_type"].upper()
        assert self._coronary_type in ["LCA", "RCA"], f"Unknown coronary type: {self._coronary_type}"
        self._volume_shape = tuple(data["volume_size"])
        self._affine_dict = {
            "LCA": np.array(data["lca_centering_affine"]),
            "RCA": np.array(data["rca_centering_affine"]),
            "volume": np.array(data["volume_affine"]),
        }
        self.roi = data.get("roi", None)

        n_images = len(data["frames"])
        image_names = [i for i in range(n_images)]
        image_paths = [tiff_path for _ in range(n_images)]

        if self.params.mode == "reconstruction":
            train_set = ImageSet(
                image_names=image_names,
                image_paths=image_paths,
                depth_paths=None,
                mask_paths=None,
                cameras=_get_cameras(data),
                extra_data=None,
                extra_data_processor=None,
            )
            valid_set = train_set
            test_set = train_set
        elif self.params.mode == "render-new-views":
            train_indices, valid_indices = self._random_split(n_images)
            train_set = ImageSet(
                image_names=[image_names[i] for i in train_indices],
                image_paths=[image_paths[i] for i in train_indices],
                depth_paths=None,
                mask_paths=None,
                cameras=_get_cameras(data, train_indices),
                extra_data=None,
                extra_data_processor=None,
            )
            valid_set = ImageSet(
                image_names=[image_names[i] for i in valid_indices],
                image_paths=[image_paths[i] for i in valid_indices],
                depth_paths=None,
                mask_paths=None,
                cameras=_get_cameras(data, valid_indices),
                extra_data=None,
                extra_data_processor=None,
            )
            test_set = valid_set
        else:
            raise ValueError(f"Unknown mode: {self.params.mode}")

        bounds = self._get_bounds()
        match self.params.init_point_cloud_mode:
            case "uniform":
                size = int(round(self.params.init_point_cloud_num ** (1 / 3)))
                axes = [np.linspace(-0.5 * b, 0.5 * b, size) for b in bounds]
                xyz = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
            case "random":
                xyz = np.random.rand(self.params.init_point_cloud_num, 3) * bounds - 0.5 * bounds
            case "random-ball":
                rng = np.random.default_rng(self.params.seed)
                phi = rng.uniform(0, 2 * np.pi, self.params.init_point_cloud_num)
                costheta = rng.uniform(-1, 1, self.params.init_point_cloud_num)
                u = rng.uniform(0, 1, self.params.init_point_cloud_num)
                theta = np.arccos(costheta)
                r = np.cbrt(u) * bounds.min() / 2
                xyz = np.array([
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta),
                ]).T
            case "FBP":
                raise NotImplementedError
            case "DL":
                raise NotImplementedError
            case "label":
                assert self.params.coronary_type is not None, "coronary_type must be specified when init_point_cloud_mode is label"
                assert self.params.coronary_type == self._coronary_type, f"coronary_type in params ({self.params.coronary_type}) must be the same as coronary_type in data ({self._coronary_type})"
                xyz = init_point_cloud_from_label(data_root / "label_3d.nii.gz", coronary_type=self.params.coronary_type)
                xyz_background = _get_backgound_gaussian_from_xyz(torch.from_numpy(xyz).float()).numpy()
                xyz = np.concatenate([xyz, xyz_background], axis=0)
            case "central-line":
                xyz = np.load(data_root / "central_line.npz")['arr_0']
                xyz_background = _get_backgound_gaussian_from_xyz(torch.from_numpy(xyz).float()).numpy()
                xyz = np.concatenate([xyz, xyz_background], axis=0)
            case _:
                raise ValueError(f"Unknown init_point_cloud_mod: {self.params.init_point_cloud_mode}")

        xyz = _filter_points_visible(xyz, train_set.cameras)
        rgb = np.ones(xyz.shape) * 127
        return DataParserOutputs(
            train_set=train_set,
            val_set=valid_set,
            test_set=test_set,
            point_cloud=PointCloud(xyz=xyz, rgb=rgb),
        )
