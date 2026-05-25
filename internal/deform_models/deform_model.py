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
    
    @classmethod
    def cat_together_ch(cls) -> int:
        return 3 + 3 + 1   # [d_xyz, d_scale, d_rotation(quat_angle)]
    
    @classmethod
    def ch_schema(cls) -> dict[str, tuple[int, int]]:
        return {
            "d_xyz": (0, 3),
            "d_rotation": (3, 6),
            "d_scaling": (6, 7),
        }
        
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
        assert deform.shape[-1] == self.cat_together_ch()
        return deform


class DeformsMARecoder(nn.Module):
    deforms_mean: torch.Tensor|None
    deforms_2_mean: torch.Tensor|None
    
    def __init__(self, deforms_type: type[Deforms]=Deforms, ema_lambda: float = 0.95):
        super().__init__()
        self.deforms_ch = deforms_type.cat_together_ch()
        self.schema = deforms_type.ch_schema()
        self.ema_lambda = ema_lambda
    
    def setup(self, n_gaussians: int, device: torch.device):
        """ Setup the buffers for recording the mean and variance of the deforms. This should be called before training starts."""
        self.register_buffer("deforms_mean", torch.zeros(n_gaussians, self.deforms_ch, device=device, requires_grad=False))
        self.register_buffer("deforms_2_mean", torch.zeros(n_gaussians, self.deforms_ch, device=device, requires_grad=False))
    
    def update(self, new_deforms: Deforms) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Update the mean and variance MA of the deforms.
        Args:
            new_deforms (Deforms): The new deforms to update the MA with. 
            The deforms will be concatenated into a single tensor using the `cat_together()` method
        Returns:
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: A tuple containing two dictionaries, each with
            the mean and variance of each deform component. The keys are the same as the keys in the `self.schema`, 
            and the values are tensors of shape (n_gaussians, ch) where ch is the number of channels for that deform component. 
            
            The mean and variance are calculated using an exponential moving average with the `ema_lambda` parameter.
            
            The the old means and means of squares are detached, but the new deforms and results are not, allows the 
            gradients to flow through the input deforms and therefore the deform model, but not through the MA recording itself.
        """
        assert self.deforms_mean is not None and self.deforms_2_mean is not None, "DeformsMARecoder not setup yet. Call setup() before update()."
        d = new_deforms.cat_together()   # shape = (n_gaussians, 3+3+1+1)
        
        d_2 = torch.square(d)
        k = self.ema_lambda
        deforms_mean = k * self.deforms_mean + (1-k) * d
        deforms_2_mean = k * self.deforms_2_mean + (1-k) * d_2
        
        self.deforms_mean = deforms_mean.detach()
        self.deforms_2_mean = deforms_2_mean.detach()

        deforms_var = deforms_2_mean - torch.square(deforms_mean)
        
        deforms_mean_dict = {
            key: deforms_mean[:, slice(*idx_range)] for key, idx_range in self.schema.items()
        }
        deforms_var_dict = {
            key: deforms_var[:, slice(*idx_range)] for key, idx_range in self.schema.items()
        }
        
        return deforms_mean_dict, deforms_var_dict
    
    def forward(self, deforms: Deforms) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return self.update(deforms)
    
    def get_deforms_var(self) -> dict[str, torch.Tensor]:
        """ 
        Get the variance of the deforms. The variance is calculated as E[X^2] - (E[X])^2.
        Returns:
            dict[str, torch.Tensor]: A dictionary containing the variance of each deform component. The keys
            are the same as the keys in the `self.schema`, and the values are tensors of shape (n_gaussians, ch) 
            where ch is the number of channels for that deform component.
            The result always returns a detached tensor, which is not tracked by autograd.
        """
        
        assert self.deforms_mean is not None and self.deforms_2_mean is not None, "DeformsMARecoder not setup yet. Call setup() before update()."
        var = self.deforms_2_mean - torch.square(self.deforms_mean)
        var = var.detach()
        return {
            key: var[:, slice(*idx_range)] for key, idx_range in self.schema.items()
        }
    
    def get_deforms_mean(self) -> dict[str, torch.Tensor]:
        """
        Get the mean of the deforms.
        Returns:
            dict[str, torch.Tensor]: A dictionary containing the mean of each deform component. The keys
            are the same as the keys in the `self.schema`, and the values are tensors of shape (n_gaussians, ch) 
            where ch is the number of channels for that deform component.
            The result always returns a detached tensor, which is not tracked by autograd.
        """
        
        assert self.deforms_mean is not None and self.deforms_2_mean is not None, "DeformsMARecoder not setup yet. Call setup() before update()."
        deforms_mean = self.deforms_mean.detach()
        return {
            key: deforms_mean[:, slice(*idx_range)] for key, idx_range in self.schema.items()
        }
    
    def clone_deforms_by_mask(self, mask: torch.Tensor, repeats: int):
        """ 
        Clone the deforms mean and var by the given mask and repeats. 
        This is useful for handling the case where the number of gaussians changes due to pruning or other reasons. 
        The new deforms mean and var will be appended to the existing ones.
        """
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
        """
        Filter the deforms by the given mask.
        This is useful for handling the case where some gaussians are pruned or masked out due to other reasons.
        The deforms mean and var will be filtered by the valid_mask, and the invalid ones will be removed from the MA recording.
        """
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