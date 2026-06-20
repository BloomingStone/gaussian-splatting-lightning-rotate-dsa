from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast
import hashlib
import json

import numpy as np

from ..dataparser import Spliter, Stage
from .meta import XRayMeta

@dataclass
class XRaySpliter(Spliter[XRayMeta]):
    """
    Base class for XRay dataset splitters.  lightning jsonargparser's typing does not support Protocols, so we use a base class instead of a Protocol for splitters.
    """
    pass


@dataclass
class ReconstructionSpliter(XRaySpliter):
    """
    ReconstructionSpliter does not split the dataset, all frames are used for training, validation and testing. 
    This is used for reconstruction task, where we want to reconstruct the whole volume from all frames.
    """
    
    def split(self, data_dir: Path, meta: XRayMeta) -> dict[Stage, list[int]]:
        indices = list(range(meta.num_frames))
        return {
            "train": indices,
            "val": indices,
            "test": indices,
        }


@dataclass
class RenderNewViewsSpliter(XRaySpliter):
    train_ratio: float = 0.8
    seed: int = 0
    random_loader_mode: Literal["random-shuffle", "random-start", "no-random"] = "random-shuffle"
    """
    Random-shuffle mode may cause bad 3D metrics due to no phase-0 frames in training set.
    Random-start mode has less view range in training set, which may also cause bad metrics, 
        it should not be use unless for ablation purpose to show the influence of view range.
    It's recommended to use PhaseStratifiedSpliter instead.
    """
    

    def split(self, data_dir: Path, meta: XRayMeta) -> dict[Stage, list[int]]:
        n_images = meta.num_frames
        indices = np.arange(n_images)
        train_len = int(n_images * self.train_ratio)

        seed = int.from_bytes(hashlib.sha256(str(data_dir).encode()).digest(), byteorder="big") % (2**32)
        seed += self.seed
        rng = np.random.default_rng(seed)

        if self.random_loader_mode == "random-shuffle":
            rng.shuffle(indices)
        elif self.random_loader_mode == "random-start":
            start = int(rng.integers(0, max(n_images - train_len, 1)))
            indices = np.roll(indices, -start)
        elif self.random_loader_mode == "no-random":
            pass
        else:
            raise ValueError(f"Unknown random_loader_mode: {self.random_loader_mode}")

        train_indices = indices[:train_len].tolist()
        valid_indices = indices[train_len:].tolist()
        return {
            "train": train_indices,
            "val": valid_indices,
            "test": valid_indices,
        }


@dataclass
class PhaseStratifiedSpliter(XRaySpliter):
    """
    Stratified split by cardiac phase bins.

    For each bin: floor-allocate ``int(count * train_ratio)`` frames to training,
    then randomly pick the remaining slots from all unallocated frames.
    This guarantees ``floor(n_frames * train_ratio)`` total training frames.
    """

    train_ratio: float = 0.8
    seed: int = 0
    n_bins: int = 50

    def split(self, data_dir: Path, meta: XRayMeta) -> dict[Stage, list[int]]:
        phases: np.ndarray = cast(np.ndarray, meta.phase_array)
        n_frames = len(phases)
        target_train = int(n_frames * self.train_ratio)

        # ── discretise phase into bins ───────────────────────────────────────
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_ids = np.clip(np.digitize(phases, bin_edges) - 1, 0, self.n_bins - 1)

        # ── collect & shuffle members per bin ────────────────
        rng = np.random.default_rng(self.seed)
        bin_members: list[list[int]] = [[] for _ in range(self.n_bins)]
        for i, b in enumerate(bin_ids):
            bin_members[b].append(i)
        for m in bin_members:
            rng.shuffle(m)

        # ── floor allocation per bin ─────────────────────────
        train_mask = np.zeros(n_frames, dtype=bool)
        for b in range(self.n_bins):
            cnt = len(bin_members[b])
            n = int(cnt * self.train_ratio)
            for idx in bin_members[b][:n]:
                train_mask[idx] = True

        # ── randomly fill remaining slots from all bins ──────
        unallocated = np.where(~train_mask)[0].tolist()
        rng.shuffle(unallocated)
        need = target_train - int(train_mask.sum())
        for idx in unallocated[:need]:
            train_mask[idx] = True

        train_idx = np.where(train_mask)[0].tolist()
        val_idx = np.where(~train_mask)[0].tolist()

        return {"train": train_idx, "val": val_idx, "test": val_idx}


@dataclass
class FileSpliter(XRaySpliter):
    """
    Load train / val / test split indices from a JSON file.

    The file must contain a dict with keys ``"train"``, ``"val"``, ``"test"``,
    each mapping to a list of integer frame indices.
    """

    
    split_file_path: str = "splits.json"
    """Path to JSON file containing splits.  Can be absolute or relative to data directory."""


    def split(self, data_dir: Path, meta: XRayMeta) -> dict[Stage, list[int]]:
        path = Path(self.split_file_path)
        if not path.is_absolute():
            path = data_dir / path
        with open(path) as f:
            return cast(dict[Literal["train", "val", "test"], list[int]], json.load(f))


@dataclass
class UniformIntervalSpliter(XRaySpliter):
    """
    Uniformly sample validation frames from the full sequence.

    The first and last frames are always kept as training (they are typically
    phase-0 and also define the view‑angle span for reconstruction).
    Remaining validation slots are evenly spread across the interior frames.

    .. note::

       Because 0 and ``n-1`` are reserved for training, the actual validation
       count may be 1-2 frames **less** than ``n - floor(n * train_ratio)``
       when the target validation count is very high.  The training count will
       be correspondingly larger by the same amount.
    """

    train_ratio: float = 0.8

    def split(self, data_dir: Path, meta: XRayMeta) -> dict[Stage, list[int]]:
        n = meta.num_frames
        target_val = n - int(n * self.train_ratio)

        if target_val <= 0 or n <= 2:
            return {"train": list(range(n)), "val": [], "test": []}

        # reserve first & last for training, then spread val uniformly inside
        interior = n - 2                     # number of frames in (0, n-1)
        val_len = min(target_val, interior)  # at most interior

        if val_len == 1:
            val_set = {n // 2}
        else:
            step = (interior - 1) / (val_len - 1)
            val_set = {int(round(1 + i * step)) for i in range(val_len)}

        # ensure indices are within interior bounds
        val_set = {max(1, min(n - 2, v)) for v in val_set}

        # backfill if rounding produced fewer unique positions than needed
        remaining = sorted(set(range(1, n - 1)) - val_set)
        while len(val_set) < val_len and remaining:
            best = max(remaining, key=lambda x: min(abs(x - v) for v in val_set))
            val_set.add(best)
            remaining.remove(best)

        val_idx = sorted(val_set)[:val_len]
        train_idx = [i for i in range(n) if i not in val_idx]

        return {"train": train_idx, "val": val_idx, "test": val_idx}

    
@dataclass 
class Phase0InTrainSpliter(XRaySpliter):
    """
    Ensure all phase-0 frames are included in all splits, which significantly improves 3D metrics. 
    This is a wrapper spliter that can be used with any other spliter, e.g. ReconstructionSpliter or RenderNewViewsSpliter.
    """
    other_spliter: XRaySpliter = field(default_factory=RenderNewViewsSpliter)
    remove_phase0_from_other_spliter: bool = True
    """If True, the other_spliter will be applied only on non-phase-0 frames"""
    
    def split(self, data_dir: Path, meta: XRayMeta) -> dict[Stage, list[int]]:
        splits = self.other_spliter.split(data_dir, meta)
        phase0_frames = [i for i, p in enumerate(meta.phase_array) if p == 0]

        for idx in phase0_frames:
            if idx not in splits["train"]:
                splits["train"].append(idx)
        
        if self.remove_phase0_from_other_spliter:
            for stage in ("val", "test"):
                splits[stage] = [idx for idx in splits[stage] if idx not in phase0_frames]
        
        return splits
        

@dataclass
class AlphaRangedSpliter(XRaySpliter):
    """
    Split the dataset by a maximum angular range centered on the middle frame.

    The middle frame (``n_frames // 2``) is treated as the fully contrast-filled frame.
    Training frames are those whose alpha angle falls within
    ``[center_angle - max_angle_degree / 2, center_angle + max_angle_degree / 2]``;
    all remaining frames are used for validation / testing.

    .. note::

       This splitter assumes that the alpha angle increases monotonically from the
       first frame onward (as is typical for rotational X-ray acquisitions).
    """

    max_angle_degree: float = 180.0
    """Full angular span (in degrees) centered on the middle frame for the training set."""

    def split(self, data_dir: Path, meta: XRayMeta) -> dict[Stage, list[int]]:
        alphas = np.array([frame.alpha_degree for frame in meta.frames])
        n = len(alphas)
        center_angle = alphas[n // 2]
        half_span = self.max_angle_degree / 2.0

        train_mask = (alphas >= center_angle - half_span) & (alphas <= center_angle + half_span)

        train_idx = np.where(train_mask)[0].tolist()
        val_idx = np.where(~train_mask)[0].tolist()

        return {"train": train_idx, "val": val_idx, "test": val_idx}
