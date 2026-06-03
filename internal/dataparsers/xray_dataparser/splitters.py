from dataclasses import dataclass
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

    def split(self, data_dir: Path, meta: XRayMeta) -> dict[Stage, list[int]]:
        n_images = meta.num_frames
        indices = np.arange(n_images)
        train_len = int(n_images * self.train_ratio)

        seed = int.from_bytes(hashlib.sha256(str(data_dir).encode()).digest(), byteorder="big") % (2**32)
        seed += self.seed
        rng = np.random.default_rng(seed)

        if self.random_loader_mode == "random-shuffle":
            import warnings
            warnings.warn(
                "random-shuffle mode may bad 3D metrics due to no phase‑0 frames in training set; \
                consider using random-start mode or PhaseStratifiedSpliter instead"
            )
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

    For each phase bin (except phase 0), allocate ``max(1, floor(count × train_ratio))``
    frames to training.  Phase‑0 frames are **excluded** from training by default —
    they are only admitted when phase 0 has strictly more frames than *every other*
    bin (i.e. it is the dominant phase).  This preserves phase‑0 for validation /
    testing where 3D metrics are evaluated.
    """
    
    train_ratio: float = 0.8
    seed: int = 0
    n_bins: int = 50

    def split(self, data_dir: Path, meta: XRayMeta) -> dict[Stage, list[int]]:
        phases: np.ndarray = cast(np.ndarray, meta.phase_array)
        n_frames = len(phases)

        # ── discretise phase into bins ───────────────────────────────────────
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_ids = np.clip(np.digitize(phases, bin_edges) - 1, 0, self.n_bins - 1)

        # ── compute which bin contains phase = 0 ─────────────────────
        # Bin 0 covers [0, 1/n_bins); phase=0 falls into bin 0.
        phase0_bin = 0       # by construction above

        # ── count frames per bin ────────────────────────────
        unique, counts = np.unique(bin_ids, return_counts=True)
        bin_counts = dict(zip(unique, counts))

        # ── decide whether phase‑0 frames are allowed in training ──────────
        use_phase0 = True
        if phase0_bin in bin_counts:
            n0 = bin_counts[phase0_bin]
            use_phase0 = all(n0 > bin_counts.get(b, 0) for b in range(self.n_bins) if b != phase0_bin)

        # ── allocate frames per bin ────────────────────────────
        rng = np.random.default_rng(self.seed)
        train_idx: list[int] = []
        val_idx: list[int] = []

        for b in range(self.n_bins):
            members = np.where(bin_ids == b)[0].tolist()
            if not members:
                continue
            rng.shuffle(members)

            n_members = len(members)
            n_train = max(1, int(n_members * self.train_ratio))

            # phase‑0 bin: set n_train = 0 unless condition is met
            if b == phase0_bin and not use_phase0:
                n_train = 0

            # ensure at least 1 frame remains for validation when possible
            if n_train >= n_members and n_members > 1:
                n_train = n_members - 1

            train_idx.extend(members[:n_train])
            val_idx.extend(members[n_train:])

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
    
    
     
        
        
        
