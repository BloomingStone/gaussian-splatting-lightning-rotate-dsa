from typing import Literal, Any
import json
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math

import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float32
from PIL import Image

from ..dataparsers.dataparser import DataParserConfig, DataParser, DataParserOutputs, ImageSet, PointCloud
from ..cameras import Cameras
from .conebeam import ConeBeamParams, PngOdlTransform, ConeBeamProjector

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
        FBP: reconstruct a volume with ODL FBP or adjoint, then sample points from the normalized voxel probability.
        DL: use deep learning to init point cloud.
    init_point_cloud_num: number of points to sample for point cloud initialization.
    init_point_cloud_fbp_use_filter: use filtered back-projection when init_point_cloud_mode is FBP. If False, use the adjoint instead.
    init_point_cloud_fbp_phase_max: upper bound of phase used for FBP initialization. Only frames with phase in [0, phase_max] are used.
    coronary_type: only used when init_point_cloud_mode is label, specify which coronary to reconstruct
    train_ratio: ratio of images to use for training, only used when mode is render-new-views
    seed: random seed for data splitting and point cloud initialization
    """
    base_name: str = "rotate_dsa"
    mode: Literal["reconstruction", "render-new-views"] = "reconstruction"
    init_point_cloud_mode: Literal["uniform", "random", "random-ball", "FBP", "DL", "label", "central-line"] = "uniform"
    init_point_cloud_num: int = 100_000
    
    init_point_cloud_fbp_use_filter: bool = True
    init_point_cloud_fbp_phase_min: float = 0.0
    init_point_cloud_fbp_phase_max: float = 0.5
    
    coronary_type: Literal["LCA", "RCA"]|None = None
    train_ratio: float = 0.8
    seed: int = 0
    random_loader_mode: Literal["random-shuffle", "random-start", "no-random"] = "random-shuffle"
    
    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        return RotatedXRayDataParser(path=path, output_path=output_path, global_rank=global_rank, params=self)

def _get_frames_list(json_data: dict, key: str, indices: list[int] | None = None) -> list[Any]:
    if indices is None:
        return [d[key] for d in json_data["frames"]]
    else:
        return [json_data["frames"][i][key] for i in indices]

def _get_frames_param(json_data: dict, key: str, indices: list[int] | None = None) -> torch.Tensor:
    return torch.tensor(_get_frames_list(json_data, key, indices))


def _get_cameras(json_data: dict, indices: list[int] | None = None) -> Cameras:
    if indices is None:
        n_camras = len(json_data["frames"])
    else:
        n_camras = len(indices)

    sod = json_data["c_arm_geometry"]["sod"]
    
    # GS follows COLMAP orientation, where Z is forward direction of camera and Y is down
    # 注意欧拉角使用内旋，绕旋转后的坐标轴继续旋转，旋转矩阵作用顺序与世界坐标旋转相反，R_colmap_orient = Rx(90) @ Rz(180)
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
        camera_type=torch.zeros(n_camras),
        time=_get_frames_param(json_data, "phase", indices),
        zfar=1e5
    )


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



def _preprocess_indices_alphas(
    indices: list[int],
    alphas: list[float],
    phases: list[float],
    phase_min: float,
    phase_max: float,
) -> tuple[list[int], list[float]]:
    assert phase_min >= 0.0 and phase_max <= 0.5 and phase_min <= phase_max, \
        f"Invalid phase range: [{phase_min}, {phase_max}]"
    
    # DSA 中 alpha 角为从前向右转。在 RAS 坐标系中即旋转方向从 前（A +Y）转到右（R +X），即绕 Z 轴负向旋转。
    # degree 转 radian
    alphas = [-math.radians(a) for a in alphas]
    
    data = zip(indices, alphas, phases)
    
    # 收缩期和舒张期大致对称，因此也可以考虑使用对称的 phase 范围 [1-phase_max, 1] 来增加可用的视角数量
    y0 = 1 - phase_min
    y1 = 1 - phase_max
    
    phase_min_sym = min(y0, y1)
    phase_max_sym = max(y0, y1)
    
    data_selected = [
        (i, a, phi) for i, a, phi in data
        if (
            (phi <= phase_max + 1e-8 and phi >= phase_min - 1e-8) or
            (phi <= phase_max_sym + 1e-8 and phi >= phase_min_sym - 1e-8)
        )
    ]

    if len(data_selected) == 0:
        raise ValueError(f"No frames found in phase range [0, {phase_max}]")

    # ODL 仅接受单调递增的视角列表，因此根据 alpha 角对选定的帧进行排序
    data_sorted = sorted(data_selected, key=lambda d: float(d[1]))
    
    indices, alphas, phases = map(list, zip(*data_sorted))

    return indices, alphas


def _sample_points_from_volume(
    volume: np.ndarray,
    affine: np.ndarray,
    num_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    
    volume = volume.astype(np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

    if volume.size == 0:
        raise ValueError("Cannot sample points from an empty volume")

    volume = volume - volume.min()
    
    threshold = np.percentile(volume, 75)
    mask = volume > threshold
    indices = np.argwhere(mask)
    weights = volume[mask]
    weights /= weights.sum()
    
    chosen = rng.choice(len(indices), size=num_points, p=weights)
    coords = indices[chosen]

    coords_h = np.concatenate([coords, np.ones((coords.shape[0], 1), dtype=np.float64)], axis=1)
    xyz = (coords_h @ np.asarray(affine, dtype=np.float64).T)[:, :3]
    return xyz
    

class RotatedXRayDataParser(DataParser):
    path: str       # path to data root, which contains the json file, image folder and label folder
    output_path: str    # path to save the outputs, including the validation results, checkpoint, log, etc.
    global_rank: int
    params: RotatedXRay
    
    data_root: Path #  Path(self.path)
    data: dict[str, Any]
    coronary_type: Literal["LCA", "RCA"]
    volume_shape: tuple[int, int, int]
    _affine_dict: dict[Literal["LCA", "RCA", "volume"], np.ndarray]
    
    image_paths: list[Path]
    label_paths: list[Path]
    image_names: list[str]
    json_path: Path
    
    train_indices: list[int]
    valid_indices: list[int]
    
    depth_npy: np.ndarray
    
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
        
        self.data_root = Path(self.path)
        images_dir = self.data_root / self.params.base_name
        label_dir = self.data_root / "label"
        
        self.json_path = self.data_root / f"{self.params.base_name}.json"
        
        self.depth_npy = np.load(self.data_root / "depth_map.npz")["arr_0"]
        
        with open(self.json_path, "r") as f:
            data = json.load(f)
        
        self.data = data
        self.coronary_type = data["coronary_type"].upper()
        assert self.coronary_type in ["LCA", "RCA"], f"Unknown coronary type: {self.coronary_type}"
        self.volume_shape = tuple(data["volume_size"])
        self._affine_dict = {
            "LCA": np.array(data["lca_centering_affine"]),
            "RCA": np.array(data["rca_centering_affine"]),
            "volume": np.array(data["volume_affine"]),
        }
        
        self.image_paths = [f.absolute() for f in sorted(images_dir.glob("*.png"))]
        self.label_paths = [f.absolute() for f in sorted(label_dir.glob("*.png"))]
        self.image_names = [f.name for f in self.image_paths]
        all_indices = list(range(len(self.image_paths)))
        
        if self.params.mode == "reconstruction":
            self.train_indices = all_indices
            self.valid_indices = all_indices
        elif self.params.mode == "render-new-views":
            self.train_indices, self.valid_indices = self._random_split(len(self.image_paths))
        else:
            raise ValueError(f"Unknown mode: {self.params.mode}")
    
    def _random_split(self, n_images: int) -> tuple[list[int], list[int]]:
        indices = np.arange(n_images)
        len_train = int(n_images * self.params.train_ratio)
        
        seed = int.from_bytes(hashlib.sha256(str(self.path).encode()).digest(), byteorder='big') % 2**32 + self.params.seed
        np.random.seed(seed)
        
        match self.params.random_loader_mode:
            case "random-shuffle":
                np.random.shuffle(indices)
            case "random-start":
                start = np.random.randint(0, n_images-len_train)
                indices = np.roll(indices, -start)
            case "no-random":
                pass
            case _:
                raise ValueError(f"Unknown random_loader_mode: {self.params.random_loader_mode}")
        
        train_indices = indices[:len_train].tolist()
        valid_indices = indices[len_train:].tolist()
        return train_indices, valid_indices

    def get_outputs(self) -> DataParserOutputs:        
        def _get_set(indices: list[int]) -> ImageSet:
            return ImageSet(
                image_names=[self.image_names[i] for i in indices],
                image_paths=[self.image_paths[i] for i in indices],
                mask_paths=[self.label_paths[i] for i in indices],
                cameras=_get_cameras(self.data, indices),
                extra_data=self.depth_npy[indices].tolist(),
                extra_data_processor=torch.from_numpy
            )

        train_set = _get_set(self.train_indices)
        valid_set = _get_set(self.valid_indices)
        test_set = _get_set(self.valid_indices)  # use the same set for validation and testing
        
        bounds = self._get_bounds()
        match self.params.init_point_cloud_mode:
            case "uniform":
                size = int(round(self.params.init_point_cloud_num ** (1/3)))
                axes = [np.linspace(-0.5 * b, 0.5 * b, size) for b in bounds]
                xyz = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
            case "random":
                xyz = np.random.rand(self.params.init_point_cloud_num, 3) * bounds - bounds * 0.5
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
                xyz, volume = self._init_point_cloud_from_fbp()
            case "DL":
                raise NotImplementedError   # TODO
            case "label":
                assert self.params.coronary_type is not None, "coronary_type must be specified when init_point_cloud_mode is label"
                assert self.params.coronary_type == self.coronary_type, f"coronary_type in params ({self.params.coronary_type}) must be the same as coronary_type in data ({self.coronary_type})"
                xyz = init_point_cloud_from_label(self.data_root / "label_3d.nii.gz", coronary_type=self.params.coronary_type)
                xyz_background = _get_backgound_gaussian_from_xyz(torch.from_numpy(xyz).float()).numpy()
                xyz = np.concatenate([xyz, xyz_background], axis=0)
            case "central-line":
                xyz = np.load(self.data_root / "central_line.npz")["arr_0"]
                xyz_background = _get_backgound_gaussian_from_xyz(torch.from_numpy(xyz).float()).numpy()
                xyz = np.concatenate([xyz, xyz_background], axis=0)
            case _:
                raise ValueError(f"Unknown init_point_cloud_mod: {self.params.init_point_cloud_mode}")
        
        xyz = _filter_points_visible(xyz, train_set.cameras)
        rgb = np.ones(xyz.shape) * 127
        
        assert xyz.shape[0] > 0, "No points are visible in any camera; check the volume bounds and camera parameters"
        assert len(train_set.cameras) > 0, "No cameras found; check the input data and camera parsing"
        
        return DataParserOutputs(
            train_set=train_set,
            val_set=valid_set,
            test_set=test_set,
            point_cloud=PointCloud(xyz=xyz, rgb=rgb)
        )
    
    @property
    def volume_affine(self) -> np.ndarray:
        assert self._affine_dict is not None, "affine_dict is not set yet"
        return self._affine_dict["volume"]
    
    @property
    def coronary_affine(self) -> np.ndarray:
        """Get the centering affine for the specified coronary type (LCA or RCA)."""
        assert self._affine_dict is not None, "affine_dict is not set yet"
        return self._affine_dict[self.coronary_type]


    def _get_bounds(self) -> np.ndarray:
        shape = np.array(self.volume_shape)
        affine = self.coronary_affine
        return np.abs(affine[:3, :3]) @ shape
    
    
    def _init_point_cloud_from_fbp(self) -> tuple[np.ndarray, np.ndarray]:
        data = self.data
        
        # FBP 初始化仅使用训练集的图像，防止可能的数据泄露造成的 valid/test 评估不公平。
        # 但需要注意如果 mode 是 reconstruction，则训练集已包含所有图像。
        indices, alphas = _preprocess_indices_alphas(
            indices     =   self.train_indices,
            alphas      =   _get_frames_list(data, "alpha_degree", self.train_indices),
            phases      =   _get_frames_list(data, "phase", self.train_indices),
            phase_min   =   self.params.init_point_cloud_fbp_phase_min,
            phase_max   =   self.params.init_point_cloud_fbp_phase_max
        )

        projections = np.stack([
            np.asarray(Image.open(self.image_paths[i]).convert("L"))
            for i in indices
        ], axis=0)
        
        geom = data["c_arm_geometry"]
        affine = self.coronary_affine   # 注意这里使用 centering affine 来保证 FBP 初始化的点云与后续训练使用的坐标系一致
        projector = ConeBeamProjector(
            param = ConeBeamParams.init_from(
                shape       =   self.volume_shape,
                affine      =   affine,    
                alphas      =   np.asarray(alphas),
                proj_size   =   projections.shape[1:],
                dh          =   geom["dely"],
                dw          =   geom["delx"],
                dde         =   geom["sdd"] - geom["sod"],
                dso         =   geom["sod"],
            ), 
            img_transform = PngOdlTransform()
        )
        
        volume = projector.backward_proj(projections, use_filter=self.params.init_point_cloud_fbp_use_filter)
        xyz = _sample_points_from_volume(
            volume=volume,
            affine=affine,
            num_points=self.params.init_point_cloud_num,
            seed=self.params.seed,
        )
        
        return xyz, volume


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

    nii_image = nib.load(nii_file)
    assert isinstance(nii_image, nib.Nifti1Image)
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


def _get_backgound_gaussian_from_xyz(xyz: Float32[Tensor, "n_gaussians 3"]) -> Float32[Tensor, "n_gaussians 3"]:
    """
    sample points from sphere around coronary's xyz.
    """
    center = xyz.mean(dim=0, keepdim=True)
    d = torch.norm(xyz - center, dim=1)
    radius = d.max() * 1.2
    N = xyz.shape[0]
    device = xyz.device
    dirs = torch.randn(N, 3, device=device)
    dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)

    # p ∝ V ∝ r**3 therefor r ∝ p**{1/3}
    p = torch.rand(N, 1, device=device)
    r = radius * (p ** (1/3))
    
    return center + dirs * r