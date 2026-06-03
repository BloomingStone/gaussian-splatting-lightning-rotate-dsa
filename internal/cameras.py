from typing import Optional, Union
from dataclasses import dataclass, field

import torch
from torch import Tensor


class CameraType:
    PERSPECTIVE: int = 0
    FISHEYE: int = 1


@dataclass
class Camera:
    idx: Tensor
    R: Tensor  # [3, 3]
    T: Tensor  # [3]
    fx: Tensor
    fy: Tensor
    fov_x: Tensor
    fov_y: Tensor
    cx: Tensor
    cy: Tensor
    width: Tensor
    height: Tensor
    time: Tensor
    phase: Tensor

    world_to_camera: Tensor
    projection: Tensor
    full_projection: Tensor
    camera_center: Tensor
    znear: Tensor
    zfar: Tensor

    def to_device(self, device):
        for field in Camera.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))

        return self

    def get_K(self):
        K = torch.eye(4, dtype=torch.float, device=self.device)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy

        return K

    def get_full_perspective_projection(self):
        K = self.get_K()

        # full.transpose() = (K[R T]).transpose() = [R T].transpose() K.transpose()

        return self.world_to_camera @ K.T

    @property
    def device(self):
        return self.R.device



    
    
@dataclass
class Cameras:
    """
    Y down, Z forward
    world-to-camera
    """

    R: Tensor  # [n_cameras, 3, 3]
    T: Tensor  # [n_cameras, 3]
    fx: Tensor  # [n_cameras]
    fy: Tensor  # [n_cameras]
    fov_x: Tensor  # [n_cameras]
    fov_y: Tensor # [n_cameras]
    cx: Tensor  # [n_cameras]
    cy: Tensor  # [n_cameras]
    width: Tensor  # [n_cameras]
    height: Tensor  # [n_cameras]

    world_to_camera: Tensor # [n_cameras, 4, 4], transposed
    projection: Tensor 
    full_projection: Tensor
    camera_center: Tensor

    time: Tensor  # [n_cameras]
    phase: Tensor # [n_cameras]

    idx: Tensor  # [N_cameras]
    znear: Tensor  # [n_cameras]
    zfar: Tensor  # [n_cameras]
    
    
    @staticmethod
    def build(
        idx: Tensor,  # [n_cameras]
        R: Tensor,  # [n_cameras, 3, 3]
        T: Tensor,  # [n_cameras, 3]
        fx: Tensor|float,  # [n_cameras]
        fy: Tensor|float,  # [n_cameras]
        cx: Tensor|float,  # [n_cameras]
        cy: Tensor|float,  # [n_cameras]
        width: Tensor|float,  # [n_cameras]
        height: Tensor|float,  # [n_cameras]
        zfar: Tensor|float,  # [n_cameras]
        znear: Tensor|float,  # [n_cameras]
        time: Tensor|None = None,  # [n_cameras]
        phase: Tensor|None = None,  # [n_cameras]
    ) -> "Cameras":
        # Camera builder: keep the explanatory comments and formulas here
        # (these comments document the NDC/projection math and are useful
        # for future maintenance and verification).
        n_cameras = R.shape[0]
        
        fx = fx if isinstance(fx, torch.Tensor) else torch.full((n_cameras,), fx)
        fy = fy if isinstance(fy, torch.Tensor) else torch.full((n_cameras,), fy)
        cx = cx if isinstance(cx, torch.Tensor) else torch.full((n_cameras,), cx)
        cy = cy if isinstance(cy, torch.Tensor) else torch.full((n_cameras,), cy)
        width = width if isinstance(width, torch.Tensor) else torch.full((n_cameras,), width)
        height = height if isinstance(height, torch.Tensor) else torch.full((n_cameras,), height)
        zfar = zfar if isinstance(zfar, torch.Tensor) else torch.full((n_cameras,), zfar)
        znear = znear if isinstance(znear, torch.Tensor) else torch.full((n_cameras,), znear)
        time = time if time is not None else torch.zeros((n_cameras,))
        phase = phase if phase is not None else torch.zeros((n_cameras,))
        
        world_to_camera = torch.zeros((n_cameras, 4, 4), device=R.device)
        world_to_camera[:, :3, :3] = R
        world_to_camera[:, :3, 3] = T
        world_to_camera[:, 3, 3] = 1.
        world_to_camera = torch.transpose(world_to_camera, 1, 2)
        
        camera_center = torch.linalg.inv(world_to_camera)[:, 3, :3]
        
        fov_x = 2 * torch.atan((width / 2) / fx)
        fov_y = 2 * torch.atan((height / 2) / fy)
        
        """
        calculate ndc projection matrix
        http://www.songho.ca/opengl/gl_projectionmatrix.html
        """
        tanHalfFovY = torch.tan((fov_y / 2))
        tanHalfFovX = torch.tan((fov_x / 2))

        top = tanHalfFovY * 0.01
        bottom = -top
        right = tanHalfFovX * 0.01
        left = -right

        P = torch.zeros(fov_y.shape[0], 4, 4, device=R.device)

        z_sign = 1.0

        P[:, 0, 0] = 2.0 * 0.01 / (right - left)  # = 1 / tanHalfFovX = 2 * fx / width
        P[:, 1, 1] = 2.0 * 0.01 / (top - bottom)  # = 2 * fy / height
        P[:, 0, 2] = (right + left) / (right - left)  # = 0, right + left = 0
        P[:, 1, 2] = (top + bottom) / (top - bottom)  # = 0, top + bottom = 0
        P[:, 3, 2] = z_sign
        P[:, 2, 2] = z_sign * zfar / (zfar - znear)
        P[:, 2, 3] = -(zfar * znear) / (zfar - znear)
        projection = torch.transpose(P, 1, 2)
        
        full_projection = world_to_camera.bmm(projection)
        
        time = torch.zeros(n_cameras) if time is None else time
        phase = torch.zeros(n_cameras) if phase is None else phase
        
        return Cameras(
            R=R,
            T=T,
            fx=fx,
            fy=fy,
            fov_x=fov_x,
            fov_y=fov_y,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            time=time,
            phase=phase,
            world_to_camera=world_to_camera,
            projection=projection,
            full_projection=full_projection,
            camera_center=camera_center,
            idx=idx,
            znear=znear,
            zfar=zfar,
        )

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, index) -> Camera:
        assert self.idx is not None and self.time is not None and self.phase is not None
        return Camera(
            idx=self.idx[index],
            R=self.R[index],
            T=self.T[index],
            fx=self.fx[index],
            fy=self.fy[index],
            fov_x=self.fov_x[index],
            fov_y=self.fov_y[index],
            cx=self.cx[index],
            cy=self.cy[index],
            width=self.width[index],
            height=self.height[index],
            time=self.time[index],
            phase=self.phase[index],
            world_to_camera=self.world_to_camera[index],
            projection=self.projection[index],
            full_projection=self.full_projection[index],
            camera_center=self.camera_center[index],
            znear=self.znear[index],
            zfar=self.zfar[index],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def to(self, device: torch.device):
        for field in Cameras.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))
                

        return self
    
    @staticmethod
    def build_from_camera_list(
        camera_list: list[Camera]
    ) -> "Cameras":
        return Cameras(
            R=torch.stack([camera.R for camera in camera_list]),
            T=torch.stack([camera.T for camera in camera_list]),
            fx=torch.stack([camera.fx for camera in camera_list]),
            fy=torch.stack([camera.fy for camera in camera_list]),
            fov_x=torch.stack([camera.fov_x for camera in camera_list]),
            fov_y=torch.stack([camera.fov_y for camera in camera_list]),
            cx=torch.stack([camera.cx for camera in camera_list]),
            cy=torch.stack([camera.cy for camera in camera_list]),
            width=torch.stack([camera.width for camera in camera_list]),
            height=torch.stack([camera.height for camera in camera_list]),
            time=torch.stack([camera.time for camera in camera_list]),
            phase=torch.stack([camera.phase for camera in camera_list]),
            world_to_camera=torch.stack([camera.world_to_camera for camera in camera_list]),
            projection=torch.stack([camera.projection for camera in camera_list]),
            full_projection=torch.stack([camera.full_projection for camera in camera_list]),
            camera_center=torch.stack([camera.camera_center for camera in camera_list]),
            idx=torch.stack([camera.idx for camera in camera_list]),
            znear=torch.stack([camera.znear for camera in camera_list]),
            zfar=torch.stack([camera.zfar for camera in camera_list]),
        )
    
    def get_from_indices(self, indices: list[int]) -> "Cameras":
        return Cameras(
            R=self.R[indices],
            T=self.T[indices],
            fx=self.fx[indices],
            fy=self.fy[indices],
            fov_x=self.fov_x[indices],
            fov_y=self.fov_y[indices],
            cx=self.cx[indices],
            cy=self.cy[indices],
            width=self.width[indices],
            height=self.height[indices],
            time=self.time[indices],
            phase=self.phase[indices],
            world_to_camera=self.world_to_camera[indices],
            projection=self.projection[indices],
            full_projection=self.full_projection[indices],
            camera_center=self.camera_center[indices],
            idx=self.idx[indices],
            znear=self.znear[indices],
            zfar=self.zfar[indices],
        )
