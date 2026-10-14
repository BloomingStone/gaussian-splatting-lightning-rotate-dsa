"""Meta loader for DICOM (rotational-DSA) datasets.

:class:`DicomMeta` subclasses
:class:`internal.dataparsers.xray_dataparser.meta.XRayMeta` so that the existing
cloud parsers, splitters and the ``RotateXRayCamerasBuilder`` can be reused
unchanged: the DICOM header is mapped onto the same ``c_arm_geometry`` /
``frames`` / ``rotated_parameters`` shape the xray JSON meta exposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..dataparser import MetaLoader
from ..xray_dataparser.meta import (
    CArmGeometry,
    FrameInfo,
    RotatedParameters,
    XRayMeta,
)
from .dicom_reader import (
    DicomHeader,
    extract_pixels,
    normalize_01,
    normalize_01_percentile,
    parse_dicom_header,
)

Normalization = Literal["minmax", "percentile"]


@dataclass
class DicomMeta(XRayMeta):
    """Meta for a single multi-frame DICOM sequence.

    In addition to the inherited :class:`XRayMeta` interface this carries the
    decoded, normalized pixel volume and the optional vendor projection
    matrices.
    """

    dcm_path: Path | None = None
    """:class:`Path` to the source ``.dcm`` file."""

    pixel_data: np.ndarray | None = None
    """Normalized projection frames, ``float32`` of shape ``(nf, H, W)`` in ``[0, 1]``."""

    projection_matrices: np.ndarray | None = None
    """Vendor per-frame 3x4 projection matrices ``(nf, 12)``, or ``None``."""

    normalization: Normalization = "minmax"
    """Normalization applied to ``pixel_data``."""

    gamma: float | None = None
    """Gamma exponent applied to ``pixel_data`` (``None`` = identity)."""

    raw_pixel_min: int = 0
    """Minimum raw (pre-normalization) pixel value."""

    raw_pixel_max: int = 0
    """Maximum raw (pre-normalization) pixel value."""

    full_detector_size: tuple[int, int] = (0, 0)
    """Full detector size ``(height, width)`` *before* any ROI crop."""

    roi_offset: tuple[int, int] = (0, 0)
    """ROI crop origin ``(x0, y0)`` in full-detector pixels (``(0, 0)`` when no crop)."""

    def __post_init__(self) -> None:
        if self.full_detector_size == (0, 0):
            self.full_detector_size = (self.c_arm_geometry.height, self.c_arm_geometry.width)


@dataclass
class DicomMetaLoader(MetaLoader[DicomMeta]):
    """Build a :class:`DicomMeta` from a ``.dcm`` file inside ``data_dir``."""

    dcm_filename: str = "RXA_chest.dcm"
    """File name (or glob) of the DICOM sequence relative to ``data_dir``."""

    orientation_type: Literal["AP", "PA"] = "AP"
    """C-arm orientation. ``AP`` puts the source in front of the patient (+Y),
    ``PA`` behind (-Y). Not stored in the DICOM header, hence configurable."""

    normalization: Normalization = "minmax"
    """``minmax`` maps the global pixel range to ``[0, 1]``; ``percentile`` clips
    to ``percentile`` first, which is more robust to bright outliers."""

    percentile: tuple[float, float] = (0.01, 0.99)
    """``(lo, hi)`` percentiles used when ``normalization == "percentile"``."""

    gamma: float | None = None
    """Explicit gamma exponent. Wins over :attr:`gamma_target` when set."""

    gamma_target: float | None = None
    """When :attr:`gamma` is ``None`` and this is set, auto-compute gamma so the
    global intensity median maps to this target (e.g. ``0.5``)."""

    roi: tuple[int, int, int, int] | None = None
    """Optional ROI crop ``(x0, y0, w, h)`` in full-detector pixels. Both the
    pixels and the camera intrinsics are adjusted so the projection stays exact."""

    flip_u: bool = True
    """Mirror the detector ``u`` (column) axis about the image centre. The DICOM
    pixel array uses the display convention while the projection model uses the
    physical one; the two differ by a left/right mirror (see the Rust
    ``brush-dicom`` reference)."""

    volume_size: tuple[float, float, float] | None = None
    """Reconstruction volume size in mm used by the point-cloud parsers. When
    ``None`` it is derived from the iso-centre field of view as
    ``(fov_x, fov_x, fov_y)``."""

    def _resolve_dcm_path(self, data_dir: Path) -> Path:
        candidate = data_dir / self.dcm_filename
        if candidate.is_file():
            return candidate
        matches = sorted(data_dir.glob(self.dcm_filename))
        if not matches:
            matches = sorted(data_dir.glob("*.dcm"))
        if not matches:
            raise FileNotFoundError(f"No DICOM file found in {data_dir} (pattern: {self.dcm_filename})")
        return matches[0]

    def load(self, data_dir: Path) -> DicomMeta:
        dcm_path = self._resolve_dcm_path(Path(data_dir))
        header = parse_dicom_header(dcm_path)
        raw_pixels = extract_pixels(dcm_path)
        if raw_pixels.shape[0] != header.num_frames:
            # Trust the pixel array when the header multiplicity disagrees.
            header.num_frames = int(raw_pixels.shape[0])

        raw_min = int(raw_pixels.min()) if raw_pixels.size else 0
        raw_max = int(raw_pixels.max()) if raw_pixels.size else 0

        normalized = self._normalize(raw_pixels)
        gamma = self._resolve_gamma(normalized)
        if gamma is not None:
            normalized = np.power(normalized, gamma, dtype=np.float32)

        if self.flip_u:
            normalized = np.ascontiguousarray(normalized[:, :, ::-1])

        geometry, roi_offset = self._build_geometry(header)
        full_height, full_width = header.height, header.width
        if self.roi is not None:
            x0, y0, w, h = self.roi
            normalized = np.ascontiguousarray(normalized[:, y0 : y0 + h, x0 : x0 + w])

        frames = [
            FrameInfo(
                frame=i,
                time_s=float(header.times_s[i]),
                phase=float(header.phases[i]),
                alpha_degree=float(header.alphas_degree[i]),
                beta_degree=float(header.betas_degree[i]),
            )
            for i in range(header.num_frames)
        ]

        volume_size = self._resolve_volume_size(header, geometry)
        centering_affine = np.eye(4, dtype=np.float64)
        centering_affine[:3, 3] = -volume_size / 2.0

        return DicomMeta(
            raw_data=self._raw_data_summary(dcm_path, header, gamma),
            coronary_type="LCA",
            num_frames=header.num_frames,
            volume_size=volume_size,
            centering_affine_dict={"LCA": centering_affine, "RCA": centering_affine},
            c_arm_geometry=geometry,
            rotated_parameters=self._build_rotated_parameters(header),
            frames=frames,
            postprocess_meta={
                "normalization": self.normalization,
                "percentile": list(self.percentile),
                "gamma": gamma,
                "flip_u": self.flip_u,
                "raw_pixel_min": raw_min,
                "raw_pixel_max": raw_max,
            },
            label_3d_info=None,
            dcm_path=dcm_path,
            pixel_data=normalized,
            projection_matrices=header.projection_matrices,
            normalization=self.normalization,
            gamma=gamma,
            raw_pixel_min=raw_min,
            raw_pixel_max=raw_max,
            full_detector_size=(full_height, full_width),
            roi_offset=roi_offset,
        )

    # -- helpers ----------------------------------------------------------

    def _normalize(self, raw_pixels: np.ndarray) -> np.ndarray:
        if self.normalization == "percentile":
            lo, hi = self.percentile
            return normalize_01_percentile(raw_pixels, float(lo), float(hi))
        return normalize_01(raw_pixels)

    def _resolve_gamma(self, normalized: np.ndarray) -> float | None:
        if self.gamma is not None:
            return float(self.gamma)
        if self.gamma_target is None:
            return None
        median = float(np.median(normalized[np.isfinite(normalized)])) if normalized.size else 0.0
        if median > 1e-4 and self.gamma_target > 0.0:
            return float(np.clip(np.log(self.gamma_target) / np.log(median), 0.05, 1.0))
        return 1.0

    def _build_geometry(self, header: DicomHeader) -> tuple[CArmGeometry, tuple[int, int]]:
        """Assemble the C-arm geometry, folding an optional ROI crop into
        ``x0 / y0 / width / height`` so that ``cx = x0 + width / 2`` stays exact."""
        height, width = header.height, header.width
        x0, y0 = 0.0, 0.0
        roi_offset = (0, 0)

        if self.roi is not None:
            rx0, ry0, rw, rh = (int(v) for v in self.roi)
            if rw <= 0 or rh <= 0:
                raise ValueError(f"Invalid ROI size in {self.roi}")
            if not (0 <= rx0 < rx0 + rw <= width and 0 <= ry0 < ry0 + rh <= height):
                raise ValueError(f"ROI {self.roi} out of bounds for detector {width}x{height}")
            height, width = rh, rw
            x0 = (header.width / 2.0 - rx0) - rw / 2.0
            y0 = (header.height / 2.0 - ry0) - rh / 2.0
            roi_offset = (rx0, ry0)

        geometry = CArmGeometry(
            sdd=header.sdd,
            sod=header.sod,
            height=height,
            width=width,
            delx=header.delx,
            dely=header.dely,
            x0=x0,
            y0=y0,
            orientation_type=self.orientation_type,
        )
        return geometry, roi_offset

    @staticmethod
    def _build_rotated_parameters(header: DicomHeader) -> RotatedParameters:
        alphas = header.alphas_degree
        betas = header.betas_degree
        frame_count = max(len(alphas), 1)
        if frame_count > 1:
            angular_velocity = float((alphas[-1] - alphas[0]) / (frame_count - 1) * header.fps)
        else:
            angular_velocity = 0.0
        return RotatedParameters(
            total_frame=frame_count,
            alpha_start=float(alphas[0]) if len(alphas) else 0.0,
            beta_start=float(betas[0]) if len(betas) else 0.0,
            angular_velocity=angular_velocity,
            fps=header.fps,
            coordinate_system="RAS",
            parameterization="euler_angles",
            convention="ZXY",
        )

    def _resolve_volume_size(
        self, header: DicomHeader, geometry: CArmGeometry
    ) -> np.ndarray:
        if self.volume_size is not None:
            return np.asarray(self.volume_size, dtype=np.float64)

        # Object-plane (iso-centre) extent of the detector: physical detector
        # size scaled by the magnification sod / sdd.
        magnification = geometry.sod / geometry.sdd
        fov_x = geometry.width * geometry.delx * magnification
        fov_z = geometry.height * geometry.dely * magnification
        # Primary rotation is about the SI axis, so the transverse plane is
        # roughly rotationally symmetric: use the horizontal FOV for both x and y.
        return np.array([fov_x, fov_x, fov_z], dtype=np.float64)

    @staticmethod
    def _raw_data_summary(dcm_path: Path, header: DicomHeader, gamma: float | None) -> dict[str, Any]:
        return {
            "source": str(dcm_path),
            "num_frames": header.num_frames,
            "detector_size": [header.height, header.width],
            "delx": header.delx,
            "dely": header.dely,
            "sdd": header.sdd,
            "sod": header.sod,
            "fps": header.fps,
            "gamma": gamma,
        }
