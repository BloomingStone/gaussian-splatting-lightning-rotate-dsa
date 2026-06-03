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
    
    
     
        
        
        
