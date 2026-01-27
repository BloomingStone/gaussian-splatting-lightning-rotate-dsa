from typing import Literal
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
        reconstruction: will use all images as training set to reconstruction 3d model, the valid set and test set will be the same as training set (for GS part)
        render-new-views: will use 4/5 images as training set, and the rest as valid set and test set to render new views. 
    init_point_cloud_mod:
        uniform: will use uniform sampling in given range to init point cloud
        random: will use random sampling in given range to init point cloud
        FBP: use FBP to construct volume from images firstly, then sample points by volume value.
        DL: use deep learning to init point cloud.
    """
    base_name: str = "rotate_dsa"
    mode: Literal["reconstruction", "render-new-views"] = "reconstruction"
    init_point_cloud_mode: Literal["uniform", "random", "random-ball", "FBP", "DL", "label", "central-line"] = "uniform"
    init_point_cloud_num: int = 100_000
    coronary_type: Literal["LCA", "RCA"] = "LCA"
    
    # use_angles: bool = True    # if use angles to init cameras, will use angles to init cameras, otherwise will use R_w2c and T_w2c to init cameras
    
    seed: int = 0
    
    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        return RotatedXRayDataParser(path=path, output_path=output_path, global_rank=global_rank, params=self)


def _get_frames_param(json_data: dict, key: str, indices: list[int] | None = None) -> torch.Tensor:
    if indices is None:
        return torch.tensor([d[key] for d in json_data["frames"]])
    else:
        return torch.tensor([d[key] for i, d in enumerate(json_data["frames"]) if i in indices])


def _get_cameras(json_data: dict, indices: list[int] | None = None) -> Cameras:
    if indices is None:
        n_camras = len(json_data["frames"])
    else:
        n_camras = len(indices)

    sod = json_data["c_arm_geometry"]["sod"]
    
    # GS follows COLMAP orientation, where Z is forward direction of camera and Y is down
    R_colmap_orient = euler_angles_to_matrix(torch.tensor((torch.pi/2, torch.pi, 0.)), "XZY")
    M_colmap_orient = torch.eye(4)
    M_colmap_orient[:3, :3] = R_colmap_orient
    
    # DRR defaultly use RAS coordiant system, where right side of patient is x axis. RAS -> XYZ
    # In DSA the primary angle (alpha) is RAO (right anterior oblique), i.e. from A to R, witch is negative rotation around z axis.
    # And similar to secondary angle (beta), so here use negative angles
    M_rotation = torch.eye(4)[None].repeat(n_camras, 1, 1)
    convention = json_data["rotate_parameters"]["convention"]
    alpha = _get_frames_param(json_data, "alpha_degree", indices) / 180 * torch.pi # (N, )
    beta = _get_frames_param(json_data, "beta_degree", indices) / 180 * torch.pi # (N, )
    angles = torch.stack(( - alpha, - beta, torch.zeros(n_camras)), dim=-1)
    M_rotation[:, :3, :3] =  euler_angles_to_matrix(angles, convention)
    
    # In RAS system the default position of source/camera is in front of patient, i.e. in A(Y)
    # The souce first translate then rotation
    M_translation = torch.eye(4)
    M_translation[:3, 3] = torch.tensor([0, sod, 0])
    M_c2w = M_rotation @ M_translation @ M_colmap_orient
    
    M_w2c = torch.linalg.inv(M_c2w).float()
    R_w2c = M_w2c[:, :3, :3]
    T_w2c = M_w2c[:, :3, 3]

    sdd = json_data["c_arm_geometry"]["sdd"]
    dx = json_data["c_arm_geometry"]["delx"]        # mm / pixel
    dy = json_data["c_arm_geometry"]["dely"]        # mm / pixel
    fx = sdd / dx                                  # pixel
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
        time=_get_frames_param(json_data, "phase", indices),
        zfar=1e5
    )


def _get_bounds(json_data: dict) -> np.ndarray:
    shape = np.array(json_data["volume_size"])
    affine = np.array(json_data["volume_affine"])
    return np.abs(affine[:3, :3]) @ shape


def _filter_points_visible(
    points: np.ndarray,
    cameras: Cameras,
) -> np.ndarray:
    """
    Filter points that are visible in at least min_visible_ratio of the cameras.
    """
    points_ = torch.from_numpy(points)
    device = torch.device('cpu')
    N = points.shape[0]
    full_projection = cameras.full_projection
    M = full_projection.shape[0]

    # (N,4)
    points_h = torch.cat([points_, torch.ones(N,1, device=device)], dim=1).float()

    # full_projection: (M,4,4)
    # full_projection is row major, so need to transpose it to column major
    # i.e. points_ndc = points_h @ proj
    clip = torch.einsum('ni,mij->mnj', points_h, full_projection)  # (M,N,4)

    # NDC
    ndc = clip[..., :3] / (clip[..., 3:4] + 1e-8)   # (x, y, z, w) -> (x/w, y/w, z/w, 1)

    x = ndc[..., 0]
    y = ndc[..., 1]
    z = ndc[..., 2]

    in_frustum = (
        (x >= -1) & (x <= 1) &
        (y >= -1) & (y <= 1) &
        (z >=  0) & (z <= 1)
    )  # (M, N)

    # 只要进过任意一个相机
    visible_any = in_frustum.any(dim=0)  # (N,)
    return points_[visible_any].cpu().numpy()
    


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
        data_root = Path(self.path)
        images_dir = data_root / self.params.base_name
        label_dir = data_root / "label"
        json_path = data_root / f"{self.params.base_name}.json"
        depth_npy = np.load(data_root / "depth_map.npy")
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        image_paths = [f.absolute() for f in sorted(images_dir.glob("*.png"))]
        label_paths = [f.absolute() for f in sorted(label_dir.glob("*.png"))]
        image_names = [f.name for f in image_paths]
        
        if self.params.mode == "reconstruction":
            train_set = ImageSet(
                image_names=image_names,
                image_paths=image_paths,
                mask_paths=label_paths,
                cameras=_get_cameras(data),
                extra_data=depth_npy,
                extra_data_processor=torch.from_numpy
            )
            valid_set = train_set
            test_set = train_set
        elif self.params.mode == "render-new-views":
            train_indices, valid_indices = self._random_split(len(image_paths))
            train_set = ImageSet(
                image_names=[image_names[i] for i in train_indices],
                image_paths=[image_paths[i] for i in train_indices],
                mask_paths=[label_paths[i] for i in train_indices],
                cameras=_get_cameras(data, train_indices),
                extra_data=depth_npy[train_indices],
                extra_data_processor=torch.from_numpy
            )
            valid_set = ImageSet(
                image_names=[image_names[i] for i in valid_indices],
                image_paths=[image_paths[i] for i in valid_indices],
                mask_paths=[label_paths[i] for i in valid_indices],
                cameras=_get_cameras(data, valid_indices),
                extra_data=depth_npy[valid_indices],
                extra_data_processor=torch.from_numpy
            )
            test_set = valid_set
        else:
            raise ValueError(f"Unknown mode: {self.params.mode}")
        
        bounds = _get_bounds(data)
        match self.params.init_point_cloud_mode:
            case "uniform":
                size = int(round(self.params.init_point_cloud_num ** (1/3)))
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
                r = np.cbrt(u) * bounds.min() / 2  # cube root to ensure uniform distribution
                xyz = np.array([
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta)
                ]).T
            case "FBP":
                raise NotImplementedError   # TODO
            case "DL":
                raise NotImplementedError   # TODO
            case "label":
                xyz = init_point_cloud_from_label(data_root / "label_3d.nii.gz", coronary_type=self.params.coronary_type)
            case "central-line":
                xyz = np.load(data_root / "central_line.npy")
            case _:
                raise ValueError(f"Unknown init_point_cloud_mod: {self.params.init_point_cloud_mode}")
        
        xyz = _filter_points_visible(xyz, train_set.cameras)
        rgb = np.ones(xyz.shape) * 127
        return DataParserOutputs(
            train_set=train_set,
            val_set=valid_set,
            test_set=test_set,
            point_cloud=PointCloud(xyz=xyz, rgb=rgb)
        )


def init_point_cloud_from_label(
    nii_file: Path, 
    max_points: int | None = None, 
    label_value: int| None = None, 
    coronary_type: Literal["LCA", "RCA"] = "LCA", 
    dtype: type = np.float64
):
    """
    从 label.nii.gz 提取点云并把重心移到世界坐标中心。

    Args:
        path (str | Path): 工程/数据路径，期望文件 path/"label.nii.gz" 存在。
        max_points (int | None): 若不为 None，随机采样至该点数以控制规模。
        label_value (int | None): 若为 None，则把所有 != 0 的体素视作前景；否则只选择等于该值的体素。
        dtype: 输出坐标数据类型（默认 np.float64）。
    Returns:
        xyz_centered: (M,3) numpy array，世界坐标系下的点云（重心已移到原点）。
        centroid: (3,) numpy array，原始世界坐标系下点云的重心（未平移前）。
    """
    import nibabel as nib
    from scipy.ndimage import center_of_mass
    
    if not nii_file.exists():
        raise FileNotFoundError(f"Label file not found: {nii_file}")

    nii_image = nib.loadsave.load(nii_file)
    assert isinstance(nii_image, nib.nifti1.Nifti1Image)
    data = nii_image.get_fdata().astype(np.uint8)
    lca, rca = separate_coronary(data)
    data = lca if coronary_type == "LCA" else rca
    assert nii_image.affine is not None
    affine: np.ndarray = nii_image.affine  # shape (4,4)
    
    label_center: tuple[int, int, int] = center_of_mass(data) # type: ignore
    W, H, D = data.shape[-3:]
    image_center = (W/2, H/2, D/2)
    label_center_voxel = (
        int( (image_center[0] + label_center[0]) / 2 ), # left and right, set as the mean of image_center and label_center
        int( (image_center[1] + label_center[1]) / 2 ), # antero-posterior, same as above
        int( image_center[2] )                          # up and down, set as the center of image
    )
    
    B = affine[:3, :3]
    new_t = (B @ np.array(label_center_voxel)) * -1
    affine_new = np.eye(4)
    affine_new[:3, :3] = B
    affine_new[:3, 3] = new_t
    
    # 二值掩码：根据 label_value 或 非零
    if label_value is None:
        mask = data != 0
    else:
        mask = data == label_value

    # 获取体素索引 (i,j,k) —— 注意 nibabel 的索引顺序是 (i,j,k) 对应 data array 的 axes
    idxs = np.argwhere(mask)
    if idxs.size == 0:
        raise ValueError("No label voxels found with the given label_value/mask.")

    # 将体素索引转换为齐次坐标再乘以 affine -> world coords
    # idxs are (N,3) in (i,j,k) order; append ones for homogeneous coords
    ones = np.ones((idxs.shape[0], 1), dtype=np.float64)
    idxs_h = np.concatenate([idxs.astype(np.float64), ones], axis=1)  # (N,4)

    idxs_h = idxs_h @ affine_new.T
    
    world_xyz = idxs_h[:, :3].astype(dtype)  # (N,3)

    # 可选随机采样以控制点数
    N = world_xyz.shape[0]
    if (max_points is not None) and (N > max_points):
        rng = np.random.default_rng()
        chosen = rng.choice(N, size=max_points, replace=False)
        world_xyz = world_xyz[chosen, :]
        N = max_points

    return world_xyz


def separate_coronary(coronary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """return lca and rca"""
    from scipy.ndimage import center_of_mass, label
    coronary = coronary.squeeze()
    assert coronary.ndim == 3, "Coronary tensor after squeeze must be in shape (D, H, W)"

    labeled_array, num_features = label(coronary.astype(np.int8))  # type: ignore
    
    if num_features <= 1:
        raise ValueError("Coronary segmentation must have at least 2 components")
    
    assert labeled_array is not None
    component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)
    
    largest_indices = np.argsort(component_sizes)[-2:][::-1] + 1
    
    region_0 = (labeled_array == largest_indices[0]).astype(np.bool_)
    region_1 = (labeled_array == largest_indices[1]).astype(np.bool_)
    
    center_0 = center_of_mass(region_0)
    center_1 = center_of_mass(region_1)
    
    if center_0[0] > center_1[0]:
        return region_0, region_1
    else:
        return region_1, region_0