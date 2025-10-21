from turtle import width
from typing import Literal
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sympy import cxxcode
import torch

from .dataparser import DataParserConfig, DataParser, DataParserOutputs, ImageSet, PointCloud
from ..cameras import Cameras

# ref: DiffDRR at diffdrr/pose.py
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


# ref: DiffDRR at diffdrr/pose.py
def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


@dataclass
class RotatedXRay(DataParserConfig):
    """
    rotated x ray parser
    mode:
        reconstruction-3d: will use all images as training set to reconstruction 3d model, the valid set and test set will be the same as training set (for GS part)
        render-new-view: will use 4/5 images as training set, and the rest as valid set and test set to render new view. 
    init_point_cloud_mod:
        uniform: will use uniform sampling in given range to init point cloud
        random: will use random sampling in given range to init point cloud
        FBP: use FBP to construct volume from images firstly, then sample points by volume value.
        DL: use deep learning to init point cloud.
    """
    base_name: str = "rotate_dsa"
    mode: Literal["reconstruction-3d", "render-new-view"] = "reconstruction-3d"
    init_point_cloud_mod: Literal["uniform", "random", "FBP", "DL"] = "uniform"
    init_point_cloud_num: int = 100_000
    
    use_angles: bool = False    # if use angles to init cameras, will use angles to init cameras, otherwise will use R_w2c and T_w2c to init cameras
    
    seed: int = 0
    
    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        return RotatedXRayDataParser(path=path, output_path=output_path, global_rank=global_rank, params=self)


def _get_frames_param(json_data: dict, key: str, indices: list[int] | None = None) -> torch.Tensor:
    if indices is None:
        return torch.tensor([d[key] for d in json_data["frames"]])
    else:
        return torch.tensor([d[key] for i, d in enumerate(json_data["frames"]) if i in indices])


def _get_cameras(json_data: dict, indices: list[int] | None = None, use_angles: bool = False) -> Cameras:
    """
    ! ATTENTION !
    The geometry stored in x-ray json file using mm, but Gaussian Splatting uses meter. Here we must do some conversion.
    """
    if indices is None:
        n_camras = len(json_data["frames"])
    else:
        n_camras = len(indices)

    if use_angles:
        sod = json_data["c_arm_geometry"]["sod"]
        convention = json_data["rotate_parameters"]["convention"]
        reorient = torch.tensor(json_data["additional_config"]["reorient"])
        angles = _get_frames_param(json_data, "angle", indices) / 180 * torch.pi
        R_c2w = euler_angles_to_matrix(angles, convention)
        M_c2w = torch.eye(4)[None].repeat(n_camras, 1, 1)
        M_c2w[:, :3, :3] = R_c2w
        T_c2w = torch.eye(4)
        T_c2w[:3, 3] = torch.tensor([0, sod, 0])
        M_c2w = M_c2w @ T_c2w @ reorient
        
        R_w2c = M_c2w[:, :3, :3].transpose(-1, -2)
        T_w2c = torch.einsum("nij, nj -> ni", -R_w2c, M_c2w[:, :3, 3]) / 1000   # mm to meter
    else:
        R_w2c = torch.tensor([d["R_w2c"] for d in json_data["frames"]])
        T_w2c = torch.tensor([d["T_w2c"] for d in json_data["frames"]]) / 1000  # mm to meter
    
    
    sdd = json_data["c_arm_geometry"]["sdd"]        # mm
    dx = json_data["c_arm_geometry"]["delx"]        # mm / pixel
    dy = json_data["c_arm_geometry"]["dely"]        # mm / pixel
    fx = sdd / dx                                   # pixel
    fy = sdd / dy                                   # pixel
    width = json_data["c_arm_geometry"]["width"]    # pixel
    height = json_data["c_arm_geometry"]["height"]  # pixel
    cx = width / 2
    cy = height / 2
    
    return Cameras(
        R = R_w2c,
        T = T_w2c,
        fx = torch.ones(n_camras) * fx,
        fy = torch.ones(n_camras) * fy,
        cx = torch.ones(n_camras) * cx,
        cy = torch.ones(n_camras) * cy,
        width = torch.ones(n_camras) * width,
        height = torch.ones(n_camras) * height,
        appearance_id=torch.zeros(n_camras),
        normalized_appearance_id=torch.zeros(n_camras),
        distortion_params=None,
        camera_type=torch.zeros(n_camras),
        time=_get_frames_param(json_data, "time_s", indices),
    )


def _get_bounds(json_data: dict) -> np.ndarray:
    shape = np.array(json_data["origin_image_size"])
    affine = np.array(json_data["origin_image_affine"])
    return affine[:3, :3] @ shape / 1000     # mm to meter


class RotatedXRayDataParser(DataParser):
    def __init__(
        self,
        path: str,
        output_path: str,
        global_rank: int,
        params: RotatedXRay,
    ) -> None:
        super().__init__()
        self.path = path
        self.output_path = output_path
        self.global_rank = global_rank
        self.params = params
    
    def _random_split(self, n_images: int) -> tuple[list[int], list[int]]:
        indices = np.arange(n_images)
        np.random.seed(self.params.seed)
        np.random.shuffle(indices)
        len_train = int(n_images * 0.8)
        train_indices = indices[:len_train].tolist()
        valid_indices = indices[len_train:].tolist()
        return train_indices, valid_indices

    def get_outputs(self) -> DataParserOutputs:
        """
        ! ATTENTION !
        The geometry stored in x-ray json file using mm, but Gaussian Splatting uses meter. Here we must do some conversion.
        """
        images_dir = Path(self.path) / self.params.base_name
        json_path = Path(self.path) / f"{self.params.base_name}.json"
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        image_paths = [f.absolute() for f in images_dir.glob("*.png")]
        image_names = [f.name for f in image_paths]
        
        if self.params.mode == "reconstruction-3d":
            train_set = ImageSet(
                image_names=image_names,
                image_paths=image_paths,
                cameras=_get_cameras(data),
                extra_data=_get_frames_param(data, "phase").tolist()    # cardiac phase
            )
            valid_set = train_set
            test_set = train_set
        elif self.params.mode == "render-new-view":
            train_indices, valid_indices = self._random_split(len(image_paths))
            train_set = ImageSet(
                image_names=[image_names[i] for i in train_indices],
                image_paths=[image_paths[i] for i in train_indices],
                cameras=_get_cameras(data, train_indices),
                extra_data=_get_frames_param(data, "phase", train_indices).tolist()  # cardiac phase
            )
            valid_set = ImageSet(
                image_names=[image_names[i] for i in valid_indices],
                image_paths=[image_paths[i] for i in valid_indices],
                cameras=_get_cameras(data, valid_indices),
                extra_data=_get_frames_param(data, "phase", valid_indices).tolist()  # cardiac phase
            )
            test_set = valid_set
        else:
            raise ValueError(f"Unknown mode: {self.params.mode}")
        
        bounds = _get_bounds(data)
        if self.params.init_point_cloud_mod == "uniform":
            size = int(round(self.params.init_point_cloud_num ** (1/3)))
            axes = [np.linspace(-0.5 * b, 0.5 * b, size) for b in bounds]
            xyz = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        elif self.params.init_point_cloud_mod == "random":
            xyz = np.random.rand(self.params.init_point_cloud_num, 3) * bounds - 0.5 * bounds
        elif self.params.init_point_cloud_mod == "FBP":
            raise NotImplementedError   # TODO
        elif self.params.init_point_cloud_mod == "DL":
            raise NotImplementedError   # TODO
        else:
            raise ValueError(f"Unknown init_point_cloud_mod: {self.params.init_point_cloud_mod}")
        
        rgb = np.ones(xyz.shape) * 127
        return DataParserOutputs(
            train_set=train_set,
            val_set=valid_set,
            test_set=test_set,
            point_cloud=PointCloud(xyz=xyz, rgb=rgb)
        )


