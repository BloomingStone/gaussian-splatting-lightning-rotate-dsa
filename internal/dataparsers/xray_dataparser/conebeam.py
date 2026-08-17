from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from matplotlib import pyplot as plt

import numpy as np
from numpy.typing import DTypeLike

try:
    import odl
    from odl.applications import tomo
    from odl.applications.tomo.operators.ray_trafo import (
        RAY_TRAFO_IMPLS,
    )
except ImportError as exc:
    raise ImportError(
        "FBP initialization requires odl and its cone-beam "
        "dependencies; install them before using "
        "init_point_cloud_mode='FBP'."
    ) from exc

IMPL = "astra_cuda"
available_impls = set(RAY_TRAFO_IMPLS.keys())
if IMPL not in available_impls:
    raise RuntimeError(f"Requested impl '{IMPL}' is not available. Available ODL impls: {sorted(available_impls)}")


@dataclass
class ConeBeamParams:
    affine: np.ndarray
    shape: np.ndarray
    min_pt_world: np.ndarray
    max_pt_world: np.ndarray
    nh: int
    nw: int
    sh: float
    sw: float
    dde: float
    dso: float
    
    alphas: np.ndarray

    @staticmethod
    def init_from(
        shape: tuple[int, ...],
        affine: np.ndarray,
        alphas: np.ndarray,
        proj_size: tuple[int, int],
        dh: float,
        dw: float,
        dde: float,     # distance from origin to detector center (O -> D)
        dso: float,     # distance from origin to source (O -> S)
    ) -> "ConeBeamParams":
        affine = np.asarray(affine, dtype=np.float64)
        A = affine[:3, :3]
        spacing = np.linalg.norm(A, axis=0)
        if np.any(spacing <= 0):
            raise ValueError("Invalid volume affine: zero spacing detected")

        D = A / spacing
        perm = np.argmax(np.abs(D), axis=0)
        if len(np.unique(perm)) != 3:
            raise ValueError("Volume affine includes oblique rotation/shear; ODL cone-beam geometry requires axis-aligned volume")

        aligned_score = np.abs(D[perm, np.arange(3)])
        if not np.allclose(aligned_score, 1.0, atol=1e-3):
            raise ValueError("Volume affine includes arbitrary rotation; ODL cone-beam geometry requires axis-aligned volume")

        if not np.allclose(perm, np.arange(3)):
            raise NotImplementedError("Volume affine includes axis permutation; this parser currently assumes identity axis order")

        origin_world = affine[:3, 3]
        shape_world = A @ np.array(shape, dtype=np.float64) + origin_world
        min_pt_world = np.minimum(origin_world, shape_world).astype(np.float32)
        max_pt_world = np.maximum(origin_world, shape_world).astype(np.float32)

        nh = proj_size[0]
        nw = proj_size[1]
        sh = nh * dh
        sw = nw * dw

        params = ConeBeamParams(
            affine=affine,
            shape=np.asanyarray(shape),
            min_pt_world=min_pt_world,
            max_pt_world=max_pt_world,
            nh=nh,
            nw=nw,
            sh=sh,
            sw=sw,
            dde=dde,
            dso=dso,
            alphas=alphas,
        )
        return params


    def build_conebeam_geometry(self) -> tuple[odl.DiscretizedSpace, tomo.ConeBeamGeometry, tomo.RayTransform, odl.Operator]:
        reco_space = odl.uniform_discr(
            min_pt=[float(self.min_pt_world[0]), float(self.min_pt_world[1]), float(self.min_pt_world[2])],
            max_pt=[float(self.max_pt_world[0]), float(self.max_pt_world[1]), float(self.max_pt_world[2])],
            shape=[int(self.shape[0]), int(self.shape[1]), int(self.shape[2])],
            dtype="float32",
        )

        angle_partition = odl.nonuniform_partition(
            self.alphas.astype(np.float32))
        
        detector_partition = odl.uniform_partition(
            min_pt=[-(self.sh / 2.0), -(self.sw / 2.0)],
            max_pt=[(self.sh / 2.0), (self.sw / 2.0)],
            shape=[self.nh, self.nw],
        )
        geometry = tomo.ConeBeamGeometry(
            apart=angle_partition,
            dpart=detector_partition,
            src_radius=self.dso,
            det_radius=self.dde,
            axis=[0, 0, 1],
        )
        ray_trafo = tomo.RayTransform(vol_space=reco_space, geometry=geometry, impl=IMPL)
        fbp_op = tomo.fbp_op(ray_trafo=ray_trafo, filter_type="Ram-Lak", frequency_scaling=1.0)
        return reco_space, geometry, ray_trafo, fbp_op


class ConeBeamProjector:
    """Cone-beam projector backed by ODL / ASTRA.

    Coordinates & conventions
    -------------------------
    * ODL's ``ConeBeamGeometry`` expects monotonically increasing alpha angles
      (in radians) that describe the **source** rotation around the Z axis
      (``axis=[0, 0, 1]``).
    * Our DSA cameras are defined with the source rotating in the *opposite*
      direction, and the source–detector assignment is 180° rotated relative
      to ODL's convention.  **Therefore the caller must pass**

          alphas = -original_alpha_radians + π

      to correctly align forward / backward projections with the NIfTI
      ``centering_affine``.  See ``FdkCloudParser._preprocess_indices_alphas``
      for the reference implementation.

    * ``centering_affine`` has negative X/Y spacing (e.g. -0.4375 mm/vox),
      whereas ODL's ``uniform_discr`` always has positive spacing (min_pt →
      max_pt).  When ``align_ras=True`` (the default), ``backward_proj``
      automatically flips X and Y axes so that the returned volume is in the
      same orientation as the NIfTI label (RAS-aligned).

    Parameters
    ----------
    param : ConeBeamParams
        Cone-beam geometry parameters.
    img_transform : ImageOdlTransform
        Transform between image domain and ODL projection domain.
    align_ras : bool
        When True (default), ``backward_proj`` flips X and Y axes to match
        the RAS / NIfTI convention (ODL positive spacing → NIfTI negative
        spacing).  Set to False only when you need the raw ODL volume.
    """

    def __init__(
        self, 
        param: ConeBeamParams,
        img_transform: ImageOdlTransform,
        align_ras: bool = True,
    ):
        super().__init__()
        self.param = param
        self.reco_space, self.geometry, self.ray_trafo, self.fbp_op = self.param.build_conebeam_geometry()
        self.adj_trafo = self.ray_trafo.adjoint
        
        self.img_transform = img_transform
        self.align_ras = align_ras

    def forward_proj(self, x: np.ndarray, to_dtype: DTypeLike = np.float32) -> np.ndarray:
        # If backward_proj applied XY flip (align_ras=True), undo it before
        # raytracing — ODL's ray_trafo expects the volume in the original
        # ODL coordinate system (positive X/Y spacing).
        if self.align_ras:
            x = x[::-1, ::-1, :]
        proj = self.ray_trafo(x).asarray()
        proj -= proj.min()
        if proj.max() <= 0:
            raise ValueError("Projection is empty; cannot be normalized to [0, 1]")
        proj /= proj.max()
        
        return self.img_transform.odl_to_img(proj, to_dtype)
    
    def backward_proj(self, y: np.ndarray, use_filter: bool) -> np.ndarray:
        """Reconstruct a 3-D volume from projections.

        Parameters
        ----------
        y : (N_angles, H, W) float32
            Projection images (will be converted to ODL domain via
            ``img_transform.img_to_odl``).
        use_filter : bool
            If True, apply Ram-Lak filtered back-projection (FDK).
            If False, use the unfiltered adjoint.

        Returns
        -------
        volume : (D, H, W) float32
            Reconstructed volume.  When ``self.align_ras`` is True the X and Y
            axes are flipped so the result aligns with the NIfTI
            ``centering_affine`` (ODL positive spacing → NIfTI negative spacing).
        """
        y = self.img_transform.img_to_odl(y)
        y = add_hanning_window_at_edge(y, edge_width=10)
        if use_filter:
            volume = self.fbp_op(y).asarray()
        else:
            volume = self.adj_trafo(y).asarray()

        # ODL uses positive X/Y spacing (min_pt → max_pt), whereas our
        # centering_affine has negative X/Y spacing.  Flip to align.
        if self.align_ras:
            volume = volume[::-1, ::-1, :]
        return volume
    
def add_hanning_window_at_edge(input_proj: np.ndarray, edge_width: int) -> np.ndarray:
    if edge_width <= 0:
        return input_proj
    
    spatial_dim = input_proj.shape[-2:]
    if edge_width * 2 > min(spatial_dim):
        raise ValueError(f"edge_width {edge_width} is too large for projection spatial dimensions {spatial_dim}")
    
    hanning_h = np.hanning(2 * edge_width)
    hanning_w = np.hanning(2 * edge_width)
    
    window_h = np.ones(spatial_dim[0], dtype=np.float32)
    window_w = np.ones(spatial_dim[1], dtype=np.float32)
    
    window_h[:edge_width] = hanning_h[:edge_width]
    window_h[-edge_width:] = hanning_h[edge_width:]
    
    window_w[:edge_width] = hanning_w[:edge_width]
    window_w[-edge_width:] = hanning_w[edge_width:]
    
    window_2d = np.outer(window_h, window_w)
    return input_proj[..., :, :] * window_2d


class ImageOdlTransform(Protocol):
    def img_to_odl(self, arr: np.ndarray) -> np.ndarray:
        ...

    def odl_to_img(self, arr: np.ndarray, to_dtype: DTypeLike) -> np.ndarray:
        ...


class IdentityOdlTransform:
    def img_to_odl(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def odl_to_img(self, arr: np.ndarray, to_dtype: DTypeLike) -> np.ndarray:
        return arr


class PngOdlTransform:
    def __init__(
        self, 
        reverse_gray: bool = True,
    ):
        super().__init__()
        self.reverse_gray = reverse_gray

    
    def img_to_odl(self, arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype == np.float32 or arr.dtype == np.float64:
            assert arr.min() >= 0 and arr.max() <= 1, "Input float image should be in [0, 1]"
        
        out = arr
        out = np.rot90(out, axes=(-1, -2))
        if self.reverse_gray:
            out = 1.0 - out
        
        return out.astype(np.float32)

    def odl_to_img(self, arr: np.ndarray, to_dtype: DTypeLike) -> np.ndarray:
        # ODL 的投影结果xy轴方向与 PNG 相反，且默认原点在左下角，因此需要逆时针旋转90度
        out = arr
        out = np.rot90(out,axes=(-2, -1))
        if self.reverse_gray:
            assert arr.min() >= 0 and arr.max() <= 1, "Negation currently assumes input is in [0, 1]"
            out = 1.0 - out
        
        match to_dtype:
            case np.uint8:
                out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            case np.float32 | np.float64:
                out = out.astype(to_dtype)
            case np.uint16:
                out = np.clip(out * 65535.0, 0, 65535).astype(np.uint16)
            case _:
                raise ValueError(f"Unsupported to_dtype {to_dtype} in odl_to_img")
        return out