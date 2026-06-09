from __future__ import annotations

from dataclasses import dataclass
import torch

from .visualizer import Visualizer
from .float_colormap_visualizer import FloatColormapVisualizer, ColorMapName

@dataclass
class GammaVisualizer(Visualizer):
    gamma: float = 1.0
    normalize: bool = True
    colormap: ColorMapName = ColorMapName.GRAY

    def process(self, image: torch.Tensor) -> torch.Tensor:
        image = image ** self.gamma
        
        return FloatColormapVisualizer(self.colormap, self.normalize).process(image)