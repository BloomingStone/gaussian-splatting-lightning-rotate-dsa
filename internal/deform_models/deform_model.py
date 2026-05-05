from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from internal.utils.gaussian_utils import GaussianTransformUtils
from internal.instantiate_config import Instantiable

@dataclass
class Deforms:
    d_xyz: torch.Tensor
    d_rotation: torch.Tensor
    d_scaling: torch.Tensor
    
    cat_together_ch = 3 + 3 + 1   # [d_xyz, d_scale, d_rotation(quat_angle)]
        
    def cat_together(self) -> torch.Tensor:
        """ Concatenate the deforms into a single tensor for easier processing. The order is [d_xyz, d_scale, d_rotation(quat_angle), d_density].

        Returns:
            torch.Tensor: The concatenated tensor. shape: (n_gaussians, 3 + 3 + 1)
        """
        d_xyz, d_rotation, d_scale = self.d_xyz, self.d_rotation, self.d_scaling
        n_d_xyz, n_d_scale, n_d_rot = d_xyz.shape[0], d_scale.shape[0], d_rotation.shape[0]
        assert n_d_xyz == n_d_scale == n_d_rot
        
        d_rotation_norm = torch.nn.functional.normalize(d_rotation, dim=-1)
        d_rotation_norm = d_rotation_norm.clamp(-1 + 1e-6, 1 - 1e-6)
        d_angle = 2 * torch.acos(d_rotation_norm[:, 0]).unsqueeze(-1)
        deform = torch.cat((d_xyz, d_scale, d_angle), dim=-1)
        assert deform.shape[-1] == self.cat_together_ch
        return deform


class DeformsMARecoder(nn.Module):
    deforms_mean: torch.Tensor|None
    deforms_2_mean: torch.Tensor|None
    
    def __init__(self, deforms_type: type[Deforms]=Deforms, ema_lambda: float = 0.95):
        super().__init__()
        self.deforms_ch = deforms_type.cat_together_ch
        self.ema_lambda = ema_lambda
    
    def setup(self, n_gaussians: int, device: torch.device):
        self.register_buffer("deforms_mean", torch.zeros(n_gaussians, self.deforms_ch, device=device, requires_grad=False))
        self.register_buffer("deforms_2_mean", torch.zeros(n_gaussians, self.deforms_ch, device=device, requires_grad=False))
    
    def update(self, new_deforms: Deforms):
        assert self.deforms_mean is not None and self.deforms_2_mean is not None, "DeformsMARecoder not setup yet. Call setup() before update()."
        d = new_deforms.cat_together()   # shape = (n_gaussians, 3+3+1+1)
        
        d_2 = torch.square(d)
        k = self.ema_lambda
        deforms_mean = k * self.deforms_mean + (1-k) * d
        deforms_2_mean = k * self.deforms_2_mean + (1-k) * d_2
        
        self.deforms_mean = deforms_mean.detach()
        self.deforms_2_mean = deforms_2_mean.detach()

        deforms_var = deforms_2_mean - torch.square(deforms_mean)
        
        return deforms_mean, deforms_var
    
    def forward(self, deforms: Deforms):
        return self.update(deforms)
    
    def get_deforms_var(self) -> torch.Tensor:
        assert self.deforms_mean is not None and self.deforms_2_mean is not None, "DeformsMARecoder not setup yet. Call setup() before update()."
        return self.deforms_2_mean - torch.square(self.deforms_mean)
    
    def get_deforms_mean(self) -> torch.Tensor:
        assert self.deforms_mean is not None and self.deforms_2_mean is not None, "DeformsMARecoder not setup yet. Call setup() before update()."
        return self.deforms_mean
    
    def clone_deforms_by_mask(self, mask: torch.Tensor, repeats: int):
        assert self.deforms_mean is not None and self.deforms_2_mean is not None, "DeformsMARecoder not setup yet. Call setup() before update()."
        new_deforms_mean = self.deforms_mean[mask].repeat(repeats, 1)
        new_deforms_2_mean = self.deforms_2_mean[mask].repeat(repeats, 1)
        
        self.deforms_mean = torch.cat(
            (self.deforms_mean, new_deforms_mean), 
            dim=0
        ).detach()
        self.deforms_2_mean = torch.cat(
            (self.deforms_2_mean, new_deforms_2_mean),
            dim = 0
        ).detach()
        
        assert self.deforms_mean.shape[0] == self.deforms_2_mean.shape[0]
    
    def filter_deforms_by_mask(self, valid_mask: torch.Tensor):
        assert self.deforms_mean is not None and self.deforms_2_mean is not None, "DeformsMARecoder not setup yet. Call setup() before update()."
        self.deforms_mean = self.deforms_mean[valid_mask].detach()
        self.deforms_2_mean = self.deforms_2_mean[valid_mask].detach()
        
        assert self.deforms_mean.shape[0] == self.deforms_2_mean.shape[0]
        

@dataclass
class GSParam:
    xyz: torch.Tensor
    rotation: torch.Tensor
    scaling: torch.Tensor
    density: torch.Tensor

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
        xyz: torch.Tensor,
        t: torch.Tensor,
        phase: torch.Tensor|None = None,
    )-> Deforms:
        raise NotImplementedError
    
    @staticmethod
    def deform(
        source: GSParam,
        deforms: Deforms
    ) -> GSParam:
        xyz = source.xyz + deforms.d_xyz
        scaling = source.scaling * ( 1 + deforms.d_scaling )
        rotation = GaussianTransformUtils.quat_multiply(source.rotation, deforms.d_rotation)

        return GSParam(xyz=xyz, rotation=rotation, scaling=scaling, density=source.density)