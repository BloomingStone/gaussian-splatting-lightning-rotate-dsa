from typing import Literal, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..dataparser import MetaLoader, Meta


MM = float | int
Pixel = int
MMPerPixel = float | int
Degree = float | int
DegreePerSec = float | int
Radian = float | int
CorType = Literal["LCA", "RCA"]


@dataclass
class CArmGeometry:
    sdd: MM             # Source to detector distance
    sod: MM             # Source to object distance
    height: Pixel       # Height of image
    width: Pixel        # Width of image, default to height
    delx: MMPerPixel    # Pixel size in x direction
    dely: MMPerPixel    # Pixel size in y direction, default to delx
    x0: MM = 0.0        # detector principal point x-offset
    y0: MM = 0.0        # detector principal point y-offset


@dataclass
class RotatedParameters:
    """
    Parameters for rotated DRR.
    By default, the coordianate system is RAS, which means X axis is to the right of patient, Y axis is to the 
    anterior of patient, and Z axis is to the superior of patient.
    The rotation type is "euler_angles" and the order is ZXY, alpha is primary rotation angle, beta is secondary 
    rotation angle, so the rotation first rotate around Z axis(SI axis) by alpha, then around X axis (RL axis) by 
    beta, and finally around Y axis(AP axis).
    For now, only alpha will change, with alpha_f = alpha_start - d_alpha, d_alpha = frame * angular_velocity / fps
    """
    total_frame: int                    # Total frame of DSA;
    alpha_start: Degree                 # Primary rotation angle
    beta_start: Degree                  # Secondary rotation angle
    angular_velocity: DegreePerSec      # Angular velocity of alpha
    fps: float                          # Frame per second of DSA;
    coordinate_system: str = "RAS"          # X is R, Y is A, Z is S    # TODO use it when build cameras
    parameterization: str = "euler_angles"  # representation of rotation
    convention: str = "ZXY"                 # Camera rotation axis sequence, internal rotation

@dataclass
class FrameInfo:
    frame: int
    time_s: float   # second
    phase: float    # [0, 1], where 0 and 1 are both end-diastole, 0.5 is end-systole
    alpha_degree: Degree    # primary rotation angle, rotation around SI axis
    beta_degree: Degree     # secondary rotation angle, rotation around RL axis

 
@dataclass
class XRayMeta(Meta):
    """MetaData for XRay dataset."""
    raw_data: dict[str, Any]  # raw data loaded from json, can be used to construct other fields in Meta
    coronary_type: CorType
    num_frames: int
    volume_size: np.ndarray  # (3,) array of float, in voxel unit
    centering_affine_dict: dict[CorType, np.ndarray]  # dict of {frame_idx: (4, 4) array of float)}
    c_arm_geometry: CArmGeometry
    rotated_parameters: RotatedParameters
    frames: list[FrameInfo]
    
    @property
    def centering_affine(self) -> np.ndarray:
        """Get the centering affine for the coronary type specified in self.coronary_type."""
        return self.centering_affine_dict[self.coronary_type]
    
    @property
    def alphas_radians(self) -> np.ndarray:
        """Get the primary rotation angles in radians for all frames."""
        return np.array([frame.alpha_degree for frame in self.frames]) / 180 * np.pi
    
    @property
    def betas_radians(self) -> np.ndarray:
        """Get the secondary rotation angles in radians for all frames."""
        return np.array([frame.beta_degree for frame in self.frames]) / 180 * np.pi
    
    @property
    def phase_array(self) -> np.ndarray:
        """Get the cardiac phase for all frames."""
        return np.array([frame.phase for frame in self.frames])
    
    @property
    def time_array(self) -> np.ndarray:
        """Get the time in seconds for all frames."""
        return np.array([frame.time_s for frame in self.frames])
    
    @property
    def frame_indices(self) -> np.ndarray:
        """Get the frame indices for all frames."""
        return np.array([frame.frame for frame in self.frames])
    
    @property
    def volume_origin(self) -> np.ndarray:
        """Get the volume origin in mm, which is the translation part of centering affine."""
        return self.centering_affine[:3, 3]

@dataclass
class XRayMetaLoader(MetaLoader[XRayMeta]):
    meta_json_name: str = "rotate_dsa.json"
    
    def load(self, data_dir: Path) -> XRayMeta:
        """Load XRayMeta from the given path."""
        meta_path = data_dir / self.meta_json_name
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        
        import json
        with open(meta_path, "r") as f:
            meta_dict = json.load(f)

        rotate_parameters = dict(meta_dict.get("rotated_parameters", meta_dict.get("rotate_parameters", {})))
        if "total_frame" not in rotate_parameters and "total_frames" in rotate_parameters:
            rotate_parameters["total_frame"] = rotate_parameters["total_frames"]
        if "total_frames" in rotate_parameters:
            rotate_parameters.pop("total_frames")

        frames = [FrameInfo(**frame_info) for frame_info in meta_dict["frames"]]

        centering_affine_dict = meta_dict.get("centering_affine_dict")
        if centering_affine_dict is None:
            centering_affine_dict: dict[CorType, np.ndarray] = {
                "LCA": np.array(meta_dict["lca_centering_affine"], dtype=np.float64),
                "RCA": np.array(meta_dict["rca_centering_affine"], dtype=np.float64),
            }
        else:
            centering_affine_dict = {
                key: np.array(value, dtype=np.float64)
                for key, value in centering_affine_dict.items()
            }

        return XRayMeta(
            raw_data=meta_dict,
            coronary_type=meta_dict["coronary_type"],
            num_frames=meta_dict.get("num_frames", len(frames)),
            volume_size=np.array(meta_dict["volume_size"], dtype=np.float64),
            centering_affine_dict=centering_affine_dict,
            c_arm_geometry=CArmGeometry(**meta_dict["c_arm_geometry"]),
            rotated_parameters=RotatedParameters(**rotate_parameters),
            frames=frames,
        )