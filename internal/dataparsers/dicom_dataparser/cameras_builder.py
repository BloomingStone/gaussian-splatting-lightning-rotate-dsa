"""Camera construction for DICOM sequences.

Two geometries are supported:

* **Ideal-circle model** (default): delegates to
  :class:`internal.dataparsers.xray_dataparser.cameras_builder.RotateXRayCamerasBuilder`,
  which places the source on a circle of radius ``sod`` using the per-frame
  ``alpha`` / ``beta`` positioner angles.
* **Exact per-frame projection matrices**: ports ``camera_from_matrix`` from the
  Rust ``brush-dataset`` DICOM loader. When the vendor stored 3x4 projection
  matrices in a private tag, every camera is decoded from its own matrix, which
  captures real C-arm mechanical deviations the ideal-circle model cannot
  express. Off by default because the ideal-circle model is the one the
  existing xray camera tests validate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from ...cameras import Cameras
from ..xray_dataparser.cameras_builder import RotateXRayCamerasBuilder
from .meta import DicomMeta


@dataclass
class DicomCamerasBuilder(RotateXRayCamerasBuilder):
    """Build cameras for a DICOM sequence."""

    use_projection_matrix: bool = False
    """Use the vendor per-frame 3x4 projection matrices when available instead
    of the ideal-circle ``alpha / beta + SOD / SDD`` model."""

    def build_cameras(self, meta: DicomMeta) -> Cameras:
        matrices = meta.projection_matrices
        if self.use_projection_matrix:
            if matrices is not None and len(matrices) == meta.num_frames:
                print(f"[dicom] exact per-frame projection matrices: {len(matrices)} frames")
                return _cameras_from_projection_matrices(meta)
            print(
                "[dicom] WARNING: use_projection_matrix=True but no usable per-frame "
                "projection matrices → falling back to the ideal-circle model"
            )
        elif matrices is not None:
            print(
                f"[dicom] {len(matrices)} per-frame projection matrices present but "
                "use_projection_matrix=False → using the ideal-circle model"
            )
        return super().build_cameras(meta)


def _cameras_from_projection_matrices(meta: DicomMeta) -> Cameras:
    """Build one camera per frame from the vendor projection matrices."""
    matrices = cast(np.ndarray, meta.projection_matrices)
    geometry = meta.c_arm_geometry
    full_height, full_width = meta.full_detector_size
    roi_x0, roi_y0 = meta.roi_offset

    n = meta.num_frames
    positions = np.zeros((n, 3), dtype=np.float64)
    rotations = np.zeros((n, 3, 3), dtype=np.float64)
    fx = np.zeros(n, dtype=np.float64)
    fy = np.zeros(n, dtype=np.float64)
    cx = np.zeros(n, dtype=np.float64)
    cy = np.zeros(n, dtype=np.float64)

    for i in range(n):
        position, r_wc, f_x, f_y, c_x, c_y = _decompose_projection_matrix(
            matrices[i], float(full_width), float(full_height)
        )
        positions[i] = position
        rotations[i] = r_wc
        fx[i] = f_x
        fy[i] = f_y
        # Principal point in cropped (detector) coordinates.
        cx[i] = c_x - roi_x0
        cy[i] = c_y - roi_y0

    translations = -np.einsum("nij,nj->ni", rotations, positions)

    time = torch.from_numpy(meta.time_array).float()
    if time.numel() > 1 and float(time.max()) > 0.0:
        time = time - time.min()
        time = time / time.max()

    return Cameras.build(
        idx=torch.from_numpy(meta.frame_indices),
        R=torch.from_numpy(rotations).float(),
        T=torch.from_numpy(translations).float(),
        fx=torch.from_numpy(fx).float(),
        fy=torch.from_numpy(fy).float(),
        cx=torch.from_numpy(cx).float(),
        cy=torch.from_numpy(cy).float(),
        width=float(geometry.width),
        height=float(geometry.height),
        time=time,
        phase=torch.from_numpy(meta.phase_array).float(),
        znear=0.01,
        zfar=1e5,
    )


def _decompose_projection_matrix(
    matrix: np.ndarray, full_width: float, full_height: float
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Decode a vendor 3x4 projection matrix ``[A | b]`` (row-major, 12 values).

    The vendor world frame is ``(Anterior, Left, Superior)`` while this pipeline
    uses the RAS frame ``(Right, Anterior, Superior)`` = ``(-Left, Anterior,
    Superior)``, so the matrix is rotated into that frame first. The COLMAP/GS
    image convention mirrors the DICOM ``u`` axis about the image centre, so
    matrix row 0 is mirrored as well.

    ``A = K . R_wc`` is then split by RQ decomposition into the intrinsics
    ``K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`` and the world-to-camera
    rotation ``R_wc``; the X-ray source sits at ``S = -A^-1 b``.
    """
    # Vendor (A, L, S) -> repo RAS (R, A, S): R = -L.
    # P_repo = Mz . P_vendor  =>  M_repo = M_vendor . diag(Mz^T, 1), i.e. the
    # columns get mixed:  Mr[:, 0] = -Mv[:, 1], Mr[:, 1] = Mv[:, 0], Mr[:, 2] = Mv[:, 2].
    mv = matrix
    mr = np.empty(12, dtype=np.float64)
    for r in range(3):
        mr[r * 4 + 0] = -mv[r * 4 + 1]
        mr[r * 4 + 1] = mv[r * 4 + 0]
        mr[r * 4 + 2] = mv[r * 4 + 2]
        mr[r * 4 + 3] = mv[r * 4 + 3]

    # Mirror the u axis about the image centre: u' = W - u  =>  row0' = -row0 + W . row2.
    m = np.empty(12, dtype=np.float64)
    for c in range(4):
        m[c] = -mr[c] + full_width * mr[8 + c]
    m[4:12] = mr[4:12]

    a = np.array(
        [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]], dtype=np.float64
    )
    b = np.array([m[3], m[7], m[11]], dtype=np.float64)
    position = -np.linalg.solve(a, b)

    # RQ decomposition (cross-product form): rows r1 / r2 / r3 of R_wc.
    m1 = np.array([m[0], m[1], m[2]], dtype=np.float64)
    m2 = np.array([m[4], m[5], m[6]], dtype=np.float64)
    m3 = np.array([m[8], m[9], m[10]], dtype=np.float64)
    lam = float(np.linalg.norm(m3))
    r3 = m3 / lam
    k1 = np.cross(m1, m3)
    k2 = np.cross(m2, m3)
    r1 = k2 / np.linalg.norm(k2)
    r2 = np.cross(r3, r1)
    fx = float(np.linalg.norm(k1) / (lam * lam))
    fy = float(np.linalg.norm(k2) / (lam * lam))
    cx = float(m1 @ r3 / lam)
    cy = float(m2 @ r3 / lam)

    r_wc = np.stack([r1, r2, r3], axis=0)
    return position, r_wc, fx, fy, cx, cy
