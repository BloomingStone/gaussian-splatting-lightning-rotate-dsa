from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast
from pathlib import Path
import numpy as np
import re

import torch

from ..cameras import Cameras
from .dataparser import (
    MetaLoader, Meta, CamerasBuilder, CloudParser, Stage, PointCloud, 
    Spliter, GSDataset, DatasetBuilder, ImageItemT, ItemT, 
    DataParser, DataParserBuilder
)
from .xray_dataparser.cloud_parsers import get_AABB_corners
from ..visualizers import FloatColormapVisualizer, ColorMapName, GammaVisualizer
from .xray_dataparser.meta import Label3DInfo, compute_aabb_mask


DEFAULT_NUM_POINTS = 10_000
MU_IDODINE = 0.25

@dataclass
class ConeBeamParams:
    affine: np.ndarray
    nVoxels: np.ndarray
    sVoxels: np.ndarray
    min_pt_world: np.ndarray
    max_pt_world: np.ndarray
    nh: int
    nw: int
    sh: float
    sw: float
    dde: float
    dso: float
    num_proj: int
    start_angle: float
    end_angle: float
    proj_range: float
    
    @staticmethod
    def from_dict(dic: dict) -> ConeBeamParams:
        return ConeBeamParams(
            affine=np.asanyarray(dic["affine"]),
            nVoxels=np.asanyarray(dic["nVoxels"]),
            sVoxels=np.asanyarray(dic["sVoxels"]),
            min_pt_world=np.asanyarray(dic["min_pt_world"]),
            max_pt_world=np.asanyarray(dic["max_pt_world"]),
            nh=dic["nh"],
            nw=dic["nw"],
            sh=dic["sh"],
            sw=dic["sw"],
            dde=dic["dde"],
            dso=dic["dso"],
            num_proj=dic["num_proj"],
            start_angle=dic["start_angle"],
            end_angle=dic["end_angle"],
            proj_range=dic["proj_range"],
        )

@dataclass
class ProjMeta:
    param: ConeBeamParams
    alphas: np.ndarray
    R: np.ndarray       # R rotation at odl (z axis around), (+Y front, +z up, +x right)
    T: np.ndarray       # source position in world coordinate, shape (num_proj, 3); also the T_c2w
    
    @staticmethod
    def load_from_dict(dic: dict) -> ProjMeta:
        meta = ProjMeta.__new__(ProjMeta)
        meta.param = ConeBeamParams.from_dict(dic["param"])
        meta.alphas = np.asanyarray(dic["alphas"])
        meta.R = np.asanyarray(dic["R"])
        meta.T = np.asanyarray(dic["T"])
        return meta

@dataclass
class ThreeDGRCarMeta(Meta):
    xca_projs: torch.Tensor
    ori_projs: torch.Tensor
    label_projs: torch.Tensor
    
    ori_projs_meta: ProjMeta
    label_projs_meta: ProjMeta
    default_use_proj: Literal["ori", "label", "xca", "xca-raw"] = "label"
    
    label_3d_info: Label3DInfo | None = None
    
    def __post_init__(self):
        assert self.ori_projs_meta.param.num_proj == self.label_projs_meta.param.num_proj,\
            "ori and label projs must have the same number of projections"
        assert self.ori_projs_meta.param.num_proj == self.ori_projs_meta.alphas.shape[0] == self.ori_projs_meta.R.shape[0] == self.ori_projs_meta.T.shape[0],\
            "ori projs meta num_proj does not match the length of alphas, R, or T"
        assert self.label_projs_meta.param.num_proj == self.label_projs_meta.alphas.shape[0] == self.label_projs_meta.R.shape[0] == self.label_projs_meta.T.shape[0],\
            "label projs meta num_proj does not match the length of alphas, R, or T"

    @property
    def projs_meta(self) -> ProjMeta:
        match self.default_use_proj:
            case "ori":
                return self.ori_projs_meta
            case "label":
                return self.label_projs_meta
            case "xca":
                return self.ori_projs_meta    # xca projs share the same meta with ori projs
            case "xca-raw":
                return self.ori_projs_meta    # xca-raw projs share the same meta with ori projs
            case _:
                raise ValueError(f"Invalid default_use_proj: {self.default_use_proj}")

    @property
    def projs(self) -> torch.Tensor:
        match self.default_use_proj:
            case "ori":
                return self.ori_projs
            case "label":
                return self.label_projs
            case "xca":
                return self.xca_projs
            case "xca-raw":
                return torch.exp( - (self.ori_projs + self.label_projs * MU_IDODINE) )
            case _:
                raise ValueError(f"Invalid default_use_proj: {self.default_use_proj}")

def load_label_3d(
    data_dir: Path,
    label_sub_dir_name: str,
    has_coronary_type: bool,
    case_name: str
) -> Label3DInfo|None:
    import warnings
    regex = r"^(\w+)_(lca|rca)$"
    match = re.match(regex, case_name)
    if match is None:
        warnings.warn(f"Filename {case_name} does not match expected pattern 'case_coronary.pt', cannot load label 3D info")
        return None
    
    case_name_base, coronary_type = match.groups()
    if not has_coronary_type:
        label_name = f"{case_name_base}.nii.gz"
        label_path = data_dir / label_sub_dir_name / label_name
    else:
        label_name = f"{case_name_base}_{coronary_type}.nii.gz"
        label_path = data_dir / label_sub_dir_name / case_name_base / label_name
    if not label_path.exists():
        warnings.warn(f"Label 3D file not found: {label_path}")
        return None
    
    import nibabel as nib
    nii_img = cast(nib.Nifti1Image, nib.load(label_path))
    if nii_img.affine is None:
        warnings.warn(f"NIfTI image must have affine, but {label_path} does not. Skipping loading label 3D info.")
        return None
    
    return Label3DInfo(
        data=nii_img.get_fdata().astype(np.uint8),
        affine=nii_img.affine,
        aabb=compute_aabb_mask(nii_img.get_fdata()),
        filename=label_path.name,
    )

@dataclass
class ThreeDGRCarMetaLoader(MetaLoader[ThreeDGRCarMeta]):
    case_name: str
    default_use_proj: Literal["ori", "label", "xca", "xca-raw"] = "label"
    label_sub_dir_name: str = "labels"
    has_coronary_type: bool = True
    
    
    def load(self, data_dir: Path) -> ThreeDGRCarMeta:
        projs_pt_file = data_dir / "projs" / f"{self.case_name}.pt"
        if not projs_pt_file.exists():
            raise FileNotFoundError(f"pt file not found: {projs_pt_file}")
        
        raw_pt_data = torch.load(projs_pt_file, weights_only=False)
        return ThreeDGRCarMeta(
            xca_projs=raw_pt_data["projs"],
            ori_projs=raw_pt_data["ori_projs"],
            label_projs=raw_pt_data["label_projs"],
            ori_projs_meta=ProjMeta.load_from_dict(raw_pt_data["ori_projs_meta"]),
            label_projs_meta=ProjMeta.load_from_dict(raw_pt_data["label_projs_meta"]),
            default_use_proj=self.default_use_proj,
            label_3d_info=load_label_3d(data_dir, self.label_sub_dir_name, self.has_coronary_type, self.case_name),
        )


@dataclass
class ThreeDGRCarCamerasBuilder(CamerasBuilder[ThreeDGRCarMeta]):
    def build_cameras(self, meta: ThreeDGRCarMeta) -> Cameras:
        """Build Cameras from ThreeDGRCarMeta."""
        
        projs_meta = meta.projs_meta
        
        # GS follows COLMAP orientation, where Z is forward direction of camera and Y is down.
        reorient_rot = torch.tensor([
            [1,  0,  0 ],
            [0,   0,  1 ],
            [0,   -1,  0 ]
        ], dtype=torch.float32)
        R_c2w = torch.from_numpy(projs_meta.R).to(torch.float32)
        cam_world = torch.from_numpy(projs_meta.T).to(torch.float32)
        
        R_c2w = R_c2w @ reorient_rot
        R = R_c2w.transpose(-1, -2) # R_w2c = R_c2w^T
        T = - torch.einsum("bmn,bn->bm", (R, cam_world))    # T_w2c = -R_w2c @ cam_world
        
        n_cameras = R.shape[0]
        nw = projs_meta.param.nw
        nh = projs_meta.param.nh
        sw = projs_meta.param.sw
        sh = projs_meta.param.sh
        cx = nw / 2
        cy = nh / 2
        dde = projs_meta.param.dde
        dso = projs_meta.param.dso
        
        delx = sw / nw
        dely = sh / nh
        
        sdd = dso + dde
        fx = sdd / delx
        fy = sdd / dely
        
        return Cameras.build(
            idx = torch.arange(n_cameras),
            R = R,
            T = T,
            fx = fx,
            fy = fy,
            cx = cx,
            cy = cy,
            width=nw,
            height=nh,
            znear = 0.01,
            zfar = 1e5
        )


@dataclass
class UniformCloudParser(CloudParser[ThreeDGRCarMeta]):
    num_points: int = DEFAULT_NUM_POINTS

    def get_point_cloud(self, data_dir: Path, meta: ThreeDGRCarMeta, splits: None|dict[Stage, list[int]]=None) -> PointCloud:
        del data_dir, splits
        
        size = int(round(self.num_points ** (1/3)))
        conebeam_param = meta.projs_meta.param
        bounds = get_AABB_corners(conebeam_param.nVoxels, conebeam_param.affine)
        x0, y0, z0 = bounds.min(axis=0)
        x1, y1, z1 = bounds.max(axis=0)
        axes = [np.linspace(x0, x1, size), np.linspace(y0, y1, size), np.linspace(z0, z1, size)]
        xyz = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        rgb = np.ones(xyz.shape) * 127
        return PointCloud(xyz=xyz, feature=rgb)


@dataclass
class RandomCloudParser(CloudParser[ThreeDGRCarMeta]):
    num_points: int = DEFAULT_NUM_POINTS
    seed: int = 0

    def get_point_cloud(self, data_dir: Path, meta: ThreeDGRCarMeta, splits: None|dict[Stage, list[int]]=None) -> PointCloud:
        del data_dir, splits
        
        rng = np.random.default_rng(self.seed)
        conebeam_param = meta.projs_meta.param
        bounds = get_AABB_corners(conebeam_param.nVoxels, conebeam_param.affine)
        xyz = rng.random((self.num_points, 3)) * (bounds.max(axis=0) - bounds.min(axis=0)) + bounds.min(axis=0)
        rgb = np.ones(xyz.shape) * 127
        return PointCloud(xyz=xyz, feature=rgb)


@dataclass
class SparseViewPickSpliter(Spliter[ThreeDGRCarMeta]):
    num_views: int = 32
    val_on_selected_views: bool = True
    
    def split(self, data_dir: Path, meta: ThreeDGRCarMeta) -> dict[Stage, list[int]]:
        del data_dir
        
        n_views = meta.projs_meta.param.num_proj
        if self.num_views > n_views:
            raise ValueError(f"num_views {self.num_views} is greater than the total number of views {n_views}")
        
        step = n_views // self.num_views
        selected_indices = list(range(0, n_views, step))[:self.num_views]
        if self.val_on_selected_views:
            val_indices = selected_indices
        else:
            val_indices = []
            for idx in range(n_views):
                if idx not in selected_indices:
                    val_indices.append(idx)
        return {
            "train": selected_indices,
            "val": val_indices,
            "test": val_indices,
        }


class ThreeDGRCarDataset(GSDataset):
    meta: ThreeDGRCarMeta
    cameras: Cameras
    indices: list[int]
    
    def __init__(self, meta: ThreeDGRCarMeta, cameras: Cameras, indices: list[int]):
        self.meta = meta
        self.cameras = cameras
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> ItemT:
        idx = self.indices[idx]     # map from dataset index to camera/image index
        
        camera = self.cameras[idx]
        image = self.meta.projs[idx].clone().float()
        image = torch.rot90(image, k=1, dims=(-2, -1))   # rotate to (H, W)
        image = image[None]   # (C, H, W)
        image_item = ImageItemT(
            image_name=f"proj_{idx:03d}",
            gt_image=image,
            mask=None,
        )
        return ItemT(
            camera=camera,
            image=image_item,
            extra_data=None,
        )
    
    @property
    def image_names(self) -> list[str]:
        return [f"proj_{idx:03d}" for idx in self.indices]

@dataclass
class ThreeDGRCarDatasetBuilder(DatasetBuilder[ThreeDGRCarMeta, ThreeDGRCarDataset]):
    def build_dataset(
        self,
        data_dir: Path,
        cameras: Cameras,
        meta: ThreeDGRCarMeta,
        indices: list[int],
        stage: Stage,
    ) -> ThreeDGRCarDataset:
        del data_dir, stage
        return ThreeDGRCarDataset(meta=meta, cameras=cameras, indices=indices)


@dataclass
class ThreeDGRCarDataParserBuilder(DataParserBuilder):
    meta_loader: ThreeDGRCarMetaLoader
    spliter: SparseViewPickSpliter
    cloud_parser: UniformCloudParser|RandomCloudParser = field(default_factory=UniformCloudParser)
    cameras_builder: ThreeDGRCarCamerasBuilder = field(default_factory=ThreeDGRCarCamerasBuilder)
    dataset_builder: ThreeDGRCarDatasetBuilder = field(default_factory=ThreeDGRCarDatasetBuilder)
    filter_visible_points: bool = False
    
    def build(self):
        return DataParser(
            meta_loader=self.meta_loader,
            cloud_parser=self.cloud_parser,
            spliter=self.spliter,
            cameras_builder=self.cameras_builder,
            dataset_builder=self.dataset_builder,
            filter_visible_points=self.filter_visible_points,
            gt_image_visualizer=FloatColormapVisualizer(ColorMapName.GRAY) \
                if self.meta_loader.default_use_proj != "xca-raw" \
                else GammaVisualizer(gamma=0.1, colormap=ColorMapName.GRAY)
        )