from __future__ import annotations

from dataclasses import dataclass
import torch

from .visualizer import Visualizer
from .float_colormap_visualizer import ColorMapName, float_colormap

@dataclass
class DepthMapVisualizer(Visualizer):
    max_depth: float|None = None
    colormap: ColorMapName = ColorMapName.TURBO

    def process(self, image: torch.Tensor) -> torch.Tensor:
        max_depth = 0.
        if self.max_depth is not None:
            max_depth = self.max_depth
        if max_depth <= 0:
            max_depth = image.max()

        depth_map = image - torch.minimum(image.min(), torch.tensor(0., dtype=torch.float, device=image.device))
        depth_map = (depth_map / (max_depth + 1e-8)).clamp(max=1.)
        
        return float_colormap(depth_map, self.colormap)