from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from internal.utils.gaussian_utils import GaussianTransformUtils
from internal.instantiate_config import Instantiable

@dataclass
class DefromModelConfig(Instantiable):
    
    def instantiate(self, *args, **kwargs) -> Any:
        return DeformModel(self)

class DeformModel(nn.Module):
    def __init__(
            self,
            cfg: DefromModelConfig
    ):
        super().__init__()
        self.cfg = cfg

    
    def forward(
        self, 
        x: torch.Tensor,
        phase: torch.Tensor
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    @staticmethod
    def deform(
        xyz: torch.Tensor,
        rotation: torch.Tensor,
        scaling: torch.Tensor,
        d_xyz: torch.Tensor,
        d_rotation: torch.Tensor,
        d_scaling: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz = xyz + d_xyz
        scaling = scaling * ( 1 + d_scaling )
        rotation = GaussianTransformUtils.quat_multiply(rotation, d_rotation)
        
        return xyz, rotation, scaling
    