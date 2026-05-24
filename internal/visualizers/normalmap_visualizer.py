from __future__ import annotations

from dataclasses import dataclass

import torch

from .visualizer import Visualizer

@dataclass
class NormalMapVisualizer(Visualizer):
    def process(self, image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(image, dim=0) * 0.5 + 0.5