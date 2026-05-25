from dataclasses import dataclass
from typing import override
import torch
from torch import nn

from ..deform_model import DeformModel, DefromModelConfig, Deforms, GSParam

from internal.utils.gaussian_utils import GaussianTransformUtils

@dataclass
class DeformsWithFlow(Deforms):
    d_density: torch.Tensor
    
    @override
    @classmethod
    def cat_together_ch(cls) -> int:
        return 3 + 3 + 1 + 1   # [d_xyz, d_scale, d_rotation(quat_angle), d_density]
    
    @override
    @classmethod
    def ch_schema(cls) -> dict[str, tuple[int, int]]:
        return {
            "d_xyz": (0, 3),
            "d_rotation": (3, 6),
            "d_scaling": (6, 7),
            "d_density": (7, 8),
    }

    @override
    def cat_together(self) -> torch.Tensor:
        """ Concatenate the deforms into a single tensor for easier processing. The order is [d_xyz, d_scale, d_rotation(quat_angle), d_density].

        Returns:
            torch.Tensor: The concatenated tensor. shape: (n_gaussians, 3 + 3 + 1 + 1)
        """
        d_xyz, d_rotation, d_scale, d_density = self.d_xyz, self.d_rotation, self.d_scaling, self.d_density
        n_d_xyz, n_d_scale, n_d_rot, n_d_density = d_xyz.shape[0], d_scale.shape[0], d_rotation.shape[0], d_density.shape[0]
        assert n_d_xyz == n_d_scale == n_d_rot == n_d_density
        
        d_rotation_norm = torch.nn.functional.normalize(d_rotation, dim=-1)
        d_rotation_norm = d_rotation_norm.clamp(-1 + 1e-6, 1 - 1e-6)
        d_angle = 2 * torch.acos(d_rotation_norm[:, 0]).unsqueeze(-1)
        deform = torch.cat((d_xyz, d_scale, d_angle, d_density), dim=-1)
        assert deform.shape[-1] == self.cat_together_ch()
        return deform

@dataclass
class DeformSourceWithFlow(GSParam):
    density: torch.Tensor
    

@dataclass
class DeformWithFlowConfig(DefromModelConfig):
    def instantiate(self, *args, **kwargs) -> "DeformWithFlowModel":
        return DeformWithFlowModel(self)
    
class DeformWithFlowModel(DeformModel):
    def __init__(self, cfg: DeformWithFlowConfig = DeformWithFlowConfig()):
        super().__init__(cfg)
        self.cfg = cfg
    
    @override
    @staticmethod
    def deform(
        source: GSParam,
        deforms: Deforms
    ) -> GSParam:
        xyz = source.xyz + deforms.d_xyz
        scaling = source.scaling * ( 1 + deforms.d_scaling )
        rotation = GaussianTransformUtils.quat_multiply(source.rotation, deforms.d_rotation)

        assert isinstance(deforms, DeformsWithFlow), "DeformWithFlowModel requires deforms to be of type DeformsWithFlow"
        
        density = source.density + deforms.d_density

        return GSParam(xyz=xyz, rotation=rotation, scaling=scaling, density=density)