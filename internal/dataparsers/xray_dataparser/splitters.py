from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import hashlib

import numpy as np

from ..dataparser import Spliter
from .meta import XRayMeta


@dataclass
class ReconstructionSpliter(Spliter[XRayMeta]):
    """
    ReconstructionSpliter does not split the dataset, all frames are used for training, validation and testing. 
    This is used for reconstruction task, where we want to reconstruct the whole volume from all frames.
    """
    
    def split(self, data_dir: Path, meta: XRayMeta) -> dict[Literal["train", "val", "test"], list[int]]:
        indices = list(range(meta.num_frames))
        return {
            "train": indices,
            "val": indices,
            "test": indices,
        }


@dataclass
class RenderNewViewsSpliter(Spliter[XRayMeta]):
    train_ratio: float = 0.8
    seed: int = 42
    random_loader_mode: Literal["random-shuffle", "random-start", "no-random"] = "random-shuffle"

    def split(self, data_dir: Path, meta: XRayMeta) -> dict[Literal["train", "val", "test"], list[int]]:
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
