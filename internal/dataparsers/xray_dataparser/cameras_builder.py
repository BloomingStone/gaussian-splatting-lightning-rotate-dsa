import torch

from ...cameras import Cameras
from ..dataparser import CamerasBuilder
from .meta import XRayMeta

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




class RotateXRayCamerasBuilder(CamerasBuilder):
    def build_cameras(self, meta: XRayMeta) -> Cameras:
        """Build Cameras from XRayMeta."""
        n_cameras = meta.num_frames
        sod = meta.c_arm_geometry.sod

        # GS follows COLMAP orientation, where Z is forward direction of camera and Y is down.
        #
        # R_colmap_orient makes the camera top is +Z(patient's superior) and look towards -Y(from patient's 
        # anterior to posterior). 
        #
        # Note that the Euler Angle uses internal rotation and continues to rotate around the rotated 
        # coordinate axis. The sequence of action of the rotation matrix is opposite to that of the world 
        # coordinate rotation. R_colmap_orient = Rx(90) @ Rz (180) 
        R_colmap_orient = euler_angles_to_matrix(torch.tensor((torch.pi/2, torch.pi, 0.)), "XZY")
        M_colmap_orient = torch.eye(4)
        M_colmap_orient[:3, :3] = R_colmap_orient

        # Here we use RAS coordiant system, where right side of patient is x axis. RAS -> XYZ
        # In DSA the primary angle (alpha) is RAO (right anterior oblique), i.e. from A to R, witch is negative 
        # rotation around z axis.
        # And similar to secondary angle (beta), so here use negative angles
        #
        # convention usally is "ZXY", which means the rotation first rotate around Z axis(SI axis) by alpha, 
        # then around X axis (RL axis) by beta
        convention = meta.rotated_parameters.convention
        M_rotation = torch.eye(4)[None].repeat(n_cameras, 1, 1)
        alpha = torch.from_numpy(-meta.alphas_radians).float()
        beta = torch.from_numpy(-meta.betas_radians).float()
        angles = torch.stack([alpha, beta, torch.zeros_like(alpha)], dim=-1)
        M_rotation[:, :3, :3] =  euler_angles_to_matrix(angles, convention)
        
        # In RAS system the default position of source/camera is in front of patient. 
        # The souce first translate then rotation
        M_translation = torch.eye(4)
        M_translation[:3, 3] = torch.tensor([0, sod, 0])
        M_c2w = M_rotation @ M_translation @ M_colmap_orient
        
        M_w2c = torch.linalg.inv(M_c2w)
        R_w2c = M_w2c[:, :3, :3]
        T_w2c = M_w2c[:, :3, 3]
        
        geom = meta.c_arm_geometry
        t = torch.from_numpy(meta.time_array).float()
        t -= t.min()  # set the first frame time to 0
        t /= t.max()  # normalize time to [0, 1]
        
        return Cameras.build(
            idx =   torch.from_numpy(meta.frame_indices),  # use time as camera idx, shape (n_cameras,)
            R   =   R_w2c,  # shape (n_cameras, 3, 3)
            T   =   T_w2c,  # shape (n_cameras, 3)
            cx  =   geom.x0 + geom.width / 2,
            cy  =   geom.y0 + geom.height / 2,
            fx  =   geom.sdd / geom.delx,
            fy  =   geom.sdd / geom.dely,
            height  =   geom.height,
            width   =   geom.width,
            time    =   t,
            phase   =   torch.from_numpy(meta.phase_array).float(),
            znear   =   0.01,
            zfar    =   1e5,
        )