"""Low-level reading of multi-frame rotational-DSA (XA) DICOM files.

Python port of the Rust ``brush-dicom`` crate
(``crates/brush-dicom/src/{parser,pixel,meta}.rs``). It extracts everything the
camera / dataset builders need from a single ``.dcm`` file:

* C-arm geometry (SDD / SOD / detector size / pixel spacing) — tag mapping that
  mirrors :class:`internal.dataparsers.xray_dataparser.meta.CArmGeometry`.
* Per-frame primary / secondary positioner angles, acquisition time and
  cardiac phase.
* Optional vendor per-frame 3x4 projection matrices (private tags).
* The multi-frame ``uint16`` pixel data.

The files targeted here are uncompressed 16-bit ``MONOCHROME2`` with
``SamplesPerPixel == 1``, so ``pydicom`` can decode ``PixelData`` directly
without a codec.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.tag import Tag

# ---------------------------------------------------------------------------
# DICOM tags (mirrors `brush-dicom::parser::tags`)
# ---------------------------------------------------------------------------

TAG_DISTANCE_SOURCE_TO_DETECTOR = Tag(0x0018, 0x1110)
TAG_DISTANCE_SOURCE_TO_PATIENT = Tag(0x0018, 0x1111)
TAG_IMAGER_PIXEL_SPACING = Tag(0x0018, 0x1164)
TAG_PIXEL_SPACING = Tag(0x0028, 0x0030)

TAG_CINE_RATE = Tag(0x0018, 0x0040)
TAG_RECOMMENDED_DISPLAY_FRAME_RATE = Tag(0x0008, 0x2144)
TAG_FRAME_TIME = Tag(0x0018, 0x1063)
TAG_FRAME_TIME_VECTOR = Tag(0x0018, 0x1065)

TAG_POSITIONER_PRIMARY_ANGLE = Tag(0x0018, 0x1510)
TAG_POSITIONER_SECONDARY_ANGLE = Tag(0x0018, 0x1511)
TAG_POSITIONER_PRIMARY_ANGLE_INCREMENT = Tag(0x0018, 0x1520)
TAG_POSITIONER_SECONDARY_ANGLE_INCREMENT = Tag(0x0018, 0x1521)

TAG_ROWS = Tag(0x0028, 0x0010)
TAG_COLUMNS = Tag(0x0028, 0x0011)
TAG_NUMBER_OF_FRAMES = Tag(0x0028, 0x0008)
TAG_SAMPLES_PER_PIXEL = Tag(0x0028, 0x0002)
TAG_BITS_ALLOCATED = Tag(0x0028, 0x0100)
TAG_PIXEL_DATA = Tag(0x7FE0, 0x0010)

#: Private cardiac-phase array. The private creator is reserved at
#: ``(0071,0010)`` and the data element (``VR = FL``, one ``f32`` per frame)
#: lives at ``(0071,1010)``.
TAG_PHASE_ARRAY = Tag(0x0071, 0x1010)

#: Vendor per-frame 3x4 projection matrices (NeuAngio), ``VR = DS``, 12 decimal
#: strings per frame, row-major. Present in ``RXA_chest*.dcm``.
TAG_PROJECTION_MATRIX_DS = Tag(0x0019, 0x1003)
#: Same matrices for vendors that store them as ``VR = OF`` (float32, 12 values
#: per frame, row-major). Present in ``RXA_pig_with_phase.dcm`` / ``RXA_brain.dcm``.
TAG_PROJECTION_MATRIX_OF = Tag(0x0021, 0x1005)

#: Fallback frame rate used when the header carries no rate information.
DEFAULT_FPS = 30.0


class DicomReadError(RuntimeError):
    """Raised when a required DICOM element is missing or malformed."""


@dataclass
class DicomHeader:
    """Everything the meta / camera / dataset builders need from a header."""

    num_frames: int
    sdd: float
    sod: float
    height: int
    width: int
    delx: float
    dely: float
    fps: float
    #: Per-frame primary rotation angle around the SI axis, in degrees.
    alphas_degree: np.ndarray
    #: Per-frame secondary rotation angle around the RL axis, in degrees.
    betas_degree: np.ndarray
    #: Cumulative acquisition time in seconds.
    times_s: np.ndarray
    #: Cardiac phase in ``[0, 1]`` (0 when the private tag is absent).
    phases: np.ndarray
    #: Vendor per-frame 3x4 projection matrices, row-major, shape ``(nf, 12)``.
    #: ``None`` when absent or malformed.
    projection_matrices: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        """Detector shape as ``(frames, height, width)``."""
        return (self.num_frames, self.height, self.width)


# ---------------------------------------------------------------------------
# Dataset loading + element readers
# ---------------------------------------------------------------------------


def read_dataset(source: str | Path | bytes | bytearray) -> Dataset:
    """Read a DICOM dataset from a path or an in-memory byte buffer."""
    if isinstance(source, (bytes, bytearray)):
        return pydicom.dcmread(io.BytesIO(bytes(source)))
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")
    return pydicom.dcmread(path)


def _element(ds: Dataset, tag: Tag):
    try:
        return ds[tag]
    except KeyError:
        return None


def _to_float_list(value: Any) -> list[float] | None:
    """Coerce a DICOM value to a flat ``list[float]`` (handles bytes / multi)."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        # Raw little-endian float32 payload (VR = OF / FL).
        if len(value) % 4 == 0:
            return np.frombuffer(bytes(value), dtype="<f4").astype(np.float64).tolist()
        return None
    if isinstance(value, str):
        parts = [p for p in value.strip().split("\\") if p != ""]
        if not parts:
            return None
        try:
            return [float(p) for p in parts]
        except ValueError:
            return None
    try:
        return [float(x) for x in value]
    except TypeError:
        try:
            return [float(value)]
        except (TypeError, ValueError):
            return None


def _read_float(ds: Dataset, tag: Tag) -> float | None:
    values = _to_float_list(getattr(_element(ds, tag) or None, "value", None))
    if not values:
        return None
    return values[0]


def _read_int(ds: Dataset, tag: Tag) -> int | None:
    value = getattr(_element(ds, tag) or None, "value", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_float_multi(ds: Dataset, tag: Tag) -> list[float] | None:
    return _to_float_list(getattr(_element(ds, tag) or None, "value", None))


# ---------------------------------------------------------------------------
# Header parsing (mirrors `brush-dicom::parser`)
# ---------------------------------------------------------------------------


def parse_dicom_header(source: str | Path | bytes | bytearray) -> DicomHeader:
    """Parse the C-arm geometry, per-frame angles / timing / phase and the
    optional vendor projection matrices from a DICOM header."""
    ds = read_dataset(source)

    height = _read_int(ds, TAG_ROWS)
    width = _read_int(ds, TAG_COLUMNS)
    if height is None or width is None:
        raise DicomReadError("DICOM header is missing Rows / Columns")

    num_frames = _read_int(ds, TAG_NUMBER_OF_FRAMES) or 1

    sdd = _read_float(ds, TAG_DISTANCE_SOURCE_TO_DETECTOR)
    sod = _read_float(ds, TAG_DISTANCE_SOURCE_TO_PATIENT)
    if sdd is None:
        raise DicomReadError("DICOM header is missing DistanceSourceToDetector (SDD)")
    if sod is None:
        raise DicomReadError("DICOM header is missing DistanceSourceToPatient (SOD)")

    delx, dely = _read_pixel_spacing(ds)
    fps = _read_fps(ds)

    alphas, betas = _read_positioner_angles(ds, num_frames)
    times_s = _read_times(ds, num_frames, fps)
    phases = _read_phase_array(ds, num_frames)
    projection_matrices = _read_projection_matrices(ds, num_frames)

    return DicomHeader(
        num_frames=num_frames,
        sdd=float(sdd),
        sod=float(sod),
        height=height,
        width=width,
        delx=delx,
        dely=dely,
        fps=fps,
        alphas_degree=alphas,
        betas_degree=betas,
        times_s=times_s,
        phases=phases,
        projection_matrices=projection_matrices,
    )


def _read_pixel_spacing(ds: Dataset) -> tuple[float, float]:
    """Return ``(delx, dely)`` in mm.

    ``ImagerPixelSpacing`` is measured at the detector plane and matches the
    ``delx / dely`` used for ``fx = sdd / delx``; ``PixelSpacing`` is the
    physical (isocenter) spacing and is only a fallback.
    """
    spacing = _read_float_multi(ds, TAG_IMAGER_PIXEL_SPACING)
    if spacing is None or len(spacing) != 2:
        spacing = _read_float_multi(ds, TAG_PIXEL_SPACING)
    if spacing is None or len(spacing) != 2:
        raise DicomReadError(
            "DICOM header is missing a usable ImagerPixelSpacing / PixelSpacing (2 values)"
        )
    return float(spacing[0]), float(spacing[1])


def _read_fps(ds: Dataset) -> float:
    """Nominal frame rate: CineRate, then RecommendedDisplayFrameRate, then the
    inverse of ``FrameTime``, finally :data:`DEFAULT_FPS`."""
    for tag in (TAG_CINE_RATE, TAG_RECOMMENDED_DISPLAY_FRAME_RATE):
        value = _read_float(ds, tag)
        if value is not None and value > 0:
            return float(value)
    frame_time_ms = _read_float(ds, TAG_FRAME_TIME)
    if frame_time_ms is not None and frame_time_ms > 0:
        return 1000.0 / float(frame_time_ms)
    return DEFAULT_FPS


def _read_positioner_angles(ds: Dataset, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame primary / secondary angles in degrees.

    Prefers the multi-value ``...AngleIncrement`` arrays, falling back to the
    (constant) starting angles, then to zero.
    """
    primary_inc = _read_float_multi(ds, TAG_POSITIONER_PRIMARY_ANGLE_INCREMENT)
    secondary_inc = _read_float_multi(ds, TAG_POSITIONER_SECONDARY_ANGLE_INCREMENT)
    primary_start = _read_float(ds, TAG_POSITIONER_PRIMARY_ANGLE)
    secondary_start = _read_float(ds, TAG_POSITIONER_SECONDARY_ANGLE)

    def angle_at(values: list[float] | None, start: float | None, i: int) -> float:
        if values is not None and i < len(values):
            return float(values[i])
        if start is not None:
            return float(start)
        return 0.0

    alphas = np.array(
        [angle_at(primary_inc, primary_start, i) for i in range(num_frames)], dtype=np.float64
    )
    betas = np.array(
        [angle_at(secondary_inc, secondary_start, i) for i in range(num_frames)], dtype=np.float64
    )
    return alphas, betas


def _read_times(ds: Dataset, num_frames: int, fps: float) -> np.ndarray:
    """Cumulative acquisition time (seconds) for each frame.

    ``FrameTimeVector`` holds per-frame intervals in ms; frame ``i`` occurs at
    the cumulative sum of intervals ``[0, i)``. When the vector is absent or
    degenerate (too few entries / all zero — e.g. a single stored ``0``), fall
    back to a uniform ``i / fps`` grid.
    """
    frame_times_ms = _read_float_multi(ds, TAG_FRAME_TIME_VECTOR)
    valid = (
        frame_times_ms is not None
        and len(frame_times_ms) >= max(num_frames - 1, 1)
        and any(abs(v) > 1e-9 for v in frame_times_ms)
    )
    if valid:
        assert frame_times_ms is not None
        times = np.zeros(num_frames, dtype=np.float64)
        acc = 0.0
        for i in range(num_frames):
            times[i] = acc
            if i < len(frame_times_ms):
                acc += frame_times_ms[i] / 1000.0
        return times

    if fps <= 0:
        fps = DEFAULT_FPS
    return np.arange(num_frames, dtype=np.float64) / fps


def _read_phase_array(ds: Dataset, num_frames: int) -> np.ndarray:
    """Per-frame cardiac phase, zero-padded / truncated to ``num_frames``."""
    phases = _read_float_multi(ds, TAG_PHASE_ARRAY) or []
    out = np.zeros(num_frames, dtype=np.float64)
    count = min(len(phases), num_frames)
    if count:
        out[:count] = np.asarray(phases[:count], dtype=np.float64)
    return out


def _read_projection_matrices(ds: Dataset, num_frames: int) -> np.ndarray | None:
    """Vendor per-frame 3x4 projection matrices, row-major, shape ``(nf, 12)``.

    Returns ``None`` (so callers fall back to the ideal-circle angle model) when
    the tags are missing, have the wrong multiplicity, or hold non-finite
    values.
    """
    want = num_frames * 12

    for tag in (TAG_PROJECTION_MATRIX_DS, TAG_PROJECTION_MATRIX_OF):
        values = _read_float_multi(ds, tag)
        if values is None or len(values) != want:
            continue
        matrix = np.asarray(values, dtype=np.float64).reshape(num_frames, 12)
        if np.all(np.isfinite(matrix)):
            return matrix

    return None


# ---------------------------------------------------------------------------
# Pixel extraction + normalization (mirrors `brush-dicom::pixel`)
# ---------------------------------------------------------------------------


def extract_pixels(source: str | Path | bytes | bytearray) -> np.ndarray:
    """Extract multi-frame grayscale pixels as a ``uint16`` array ``(nf, H, W)``."""
    ds = read_dataset(source)

    samples_per_pixel = _read_int(ds, TAG_SAMPLES_PER_PIXEL)
    if samples_per_pixel is not None and samples_per_pixel != 1:
        raise DicomReadError(
            f"expected SamplesPerPixel == 1 (MONOCHROME), got {samples_per_pixel}"
        )
    bits_allocated = _read_int(ds, TAG_BITS_ALLOCATED)
    if bits_allocated is not None and bits_allocated != 16:
        raise DicomReadError(f"expected 16-bit pixels, got BitsAllocated == {bits_allocated}")
    if TAG_PIXEL_DATA not in ds:
        raise DicomReadError("DICOM header is missing PixelData")

    array = np.asarray(ds.pixel_array)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise DicomReadError(f"unsupported pixel array shape {array.shape}, expected (nf, H, W)")
    return np.ascontiguousarray(array, dtype=np.uint16)


def normalize_01(values: np.ndarray) -> np.ndarray:
    """Min-max normalize raw pixel values to ``[0, 1]``.

    The rendered X-ray projection is mapped to the same range, so the two sides
    differ only by an unknown linear transform, which min-max normalization
    discards. A constant image maps to all zeros.
    """
    flat = np.asarray(values, dtype=np.float32)
    lo = float(flat.min()) if flat.size else 0.0
    hi = float(flat.max()) if flat.size else 0.0
    rng = hi - lo
    if rng <= 0.0:
        return np.zeros_like(flat)
    return (flat - lo) / rng


def normalize_01_percentile(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip to the ``[lo, hi]`` percentiles then min-max scale to ``[0, 1]``.

    A single bright outlier (metal / bone spike) inflates the global max and
    crushes everything else under plain min-max; percentile scaling keeps the
    clinically relevant soft-tissue / iodine range visible and gives the
    Beer-Lambert loss a better-conditioned target.
    """
    flat = np.asarray(values, dtype=np.float32)
    n = flat.size
    if n == 0:
        return flat

    sorted_values = np.sort(flat.ravel())

    def index(p: float) -> int:
        return min(int(round(p * (n - 1))), n - 1)

    lo_v = float(sorted_values[index(lo)])
    hi_v = float(sorted_values[index(hi)])
    rng = hi_v - lo_v
    if rng <= 0.0:
        return np.zeros_like(flat)
    scaled = (flat - lo_v) / rng
    return np.clip(scaled, 0.0, 1.0)
