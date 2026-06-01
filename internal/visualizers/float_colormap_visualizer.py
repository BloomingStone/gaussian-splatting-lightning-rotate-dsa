from __future__ import annotations

from enum import StrEnum
from dataclasses import dataclass

from torch import Tensor
import torch
import matplotlib

from .visualizer import Visualizer


class ColorMapName(StrEnum):
    TURBO = "turbo"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    INFERNO = "inferno"
    CIVIDIS = "cividis"
    GRAY = "gray"


def normalization_preprocessor(
    image: Tensor, 
    min_clamp: float|None,
    max_clamp: float|None
) -> Tensor:
    if min_clamp is not None:
        image = torch.clamp_min(image, min=min_clamp)
        
    if max_clamp is not None:
        image = torch.clamp_max(image, max=max_clamp)

    max_value = image.max()
    min_value = image.min()
    max_diff = max_value - min_value

    image = image - min_value
    if max_diff > 0:
        image = image / max_diff
    
    return image



def float_colormap(image: Tensor, colormap: ColorMapName = ColorMapName.TURBO) -> Tensor:
    """Copied from NeRFStudio: https://github.com/nerfstudio-project/nerfstudio/blob/f97eb2e5f0c754e1ab0873374c8dcea5d18e169c/nerfstudio/utils/colormaps.py#L93-L114. Please follow their license.

    Convert single channel to a color image.

    Args:
        image: Single channel image.
        colormap: Colormap for image.

    Returns:
        Tensor: Colored image with colors in [0, 1]
    """
    image = torch.nan_to_num(image, 0)
    if colormap == ColorMapName.GRAY:
        return image.repeat(3, 1, 1)
    
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    colormap_colors = matplotlib.colormaps[colormap].colors     # type: ignore
    return torch.tensor(colormap_colors, device=image.device)[image_long[0, ...]].permute(2, 0, 1)

@dataclass
class FloatColormapVisualizer(Visualizer):
    colormap: ColorMapName = ColorMapName.TURBO
    normalize: bool = True
    normalization_clamp: tuple[float|None, float|None] = (None, None)

    def process(self, image: Tensor) -> Tensor:
        if self.normalize:
            image = normalization_preprocessor(
                image,
                min_clamp=self.normalization_clamp[0],
                max_clamp=self.normalization_clamp[1],
            )
        return float_colormap(image, colormap=self.colormap)