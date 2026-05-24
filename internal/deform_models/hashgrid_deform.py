"""
Copied from https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/utils/time_utils.py
"""
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from .deform_model import DeformModel, DefromModelConfig
from .network_factory import NetworkFactory
from ..encodings.vector_positional_encoding import VectorPositionalEncoding



class MLP(nn.Module):
    def __init__(
        self,
        D: int,
        W: int,
        layers: int,
        input_ch: int,
        output_ch: int,
        net_factory: NetworkFactory,
    ):
        super().__init__()
                
        def _layer(ch_in: int) -> nn.Module:
            return net_factory.get_network(
                n_input_dims=ch_in,
                n_output_dims=W,
                n_layers=D,
                n_neurons=W,   # n_neurons == ch_out == W
                activation="ReLU",
                output_activation="ReLU",
            )
        
        input_dims = [input_ch] + [input_ch+W] * (layers-1)
        self.skip_layers = nn.ModuleList([_layer(in_dim) for in_dim in input_dims])
        self.out_layer = net_factory.get_linear(W+input_ch, output_ch)
    
    def forward(self, x: torch.Tensor):
        h = x
        for layer in self.skip_layers:
            h = layer(h)
            h = torch.cat([x, h], dim=-1)
        return self.out_layer(h)
        
class Scale(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s
    def forward(self, x):
        return x * self.s

@dataclass
class HashGridDeformConfig(DefromModelConfig):
    D: int = 2  # deepth for each layer of MLP
    
    t_multires: int = 6
    
    combine_layers: int = 4
    combine_W: int = 128
    
    tcnn: bool = True
    
    x_multires: int = 7
    n_features_per_level: int = 4
    log2_hashmap_size: int = 7
    base_resolution: int = 16
    max_resolution: int = 128
    
    def instantiate(self, *args, **kwargs) -> Any:
        return HashGridDefromModel(self)


class HashGridDefromModel(DeformModel):
    def __init__(
            self,
            cfg: HashGridDeformConfig = HashGridDeformConfig(),
    ):
        super().__init__(cfg)
        self.network_factory = NetworkFactory(tcnn=cfg.tcnn)
        self.cfg = cfg

        self.embed_fn = self.network_factory.get_hashgrid_encoding(
            n_levels            =self.cfg.x_multires,
            n_features_per_level=self.cfg.n_features_per_level,
            log2_hashmap_size   =self.cfg.log2_hashmap_size,
            base_resolution     =self.cfg.base_resolution,
            max_resolution      =self.cfg.max_resolution
        )
        emb_x_ch = self.embed_fn.get_output_n_channels()
        
        self.embed_phase_fn = VectorPositionalEncoding(
            input_channels=1,
            n_frequencies=self.cfg.t_multires,
        )
        emb_t_ch = self.embed_phase_fn.get_output_n_channels()
        
        def _mlp(W: int, layers: int, input_ch: int):
            return MLP(
                D           =   self.cfg.D,
                W           =   W,
                layers      =   layers,
                input_ch    =   input_ch,
                output_ch   =   W,
                net_factory =   self.network_factory,
            )
        
        self.combine_mlp = _mlp(
            W=self.cfg.combine_W, 
            layers=self.cfg.combine_layers, 
            input_ch = (emb_t_ch + emb_x_ch)
        )
        
        _linear = self.network_factory.get_linear
        self.coronary_props_warp = _linear(emb_x_ch, 1)
        
        self.xyz_warp = _linear(self.cfg.combine_W, 3)
        self.scaling_warp = nn.Sequential(
            _linear(self.cfg.combine_W, 3),
            nn.Tanh(),      # new_scaling = scaling * (1 + d_scaling), so d_scaling should be in [-1, 1] to avoid negative scaling
        )
        
        _linear(self.cfg.combine_W, 3)
        self.axial_angle_warp = nn.Sequential(
            _linear(self.cfg.combine_W, 3),
            nn.Tanh(),      # w = |\vec{v}| <= \sqrt(1+1+1) = 1.73 rad = 99.2 degree  
        )
        

    def forward(
        self, 
        xyz: torch.Tensor,
        phase: torch.Tensor,
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the deformable model.
        Args:
            xyz: [n, 3] the coordinates of the points to be deformed
            phase: [1] the current phase of the cardiac cycle, normalized to [0, 1]
        Returns:
            d_xyz: [n, 3] \\in R^3, the translation for each point
            d_scaling: [n, 3] \\in [-1, 1], the scaling change for each point, where the new scaling will be scaling * (1 + d_scaling)
            d_rotation: [n, 4] \\in [-0.1*\\sqrt{3}, 0.1*\\sqrt{3}], the rotation for each point in quaternion format (w, x, y, z), where the new rotation will be rotation * d_rotation
        """
        phase_emb = self.embed_phase_fn(phase)
        x_emb = self.embed_fn(xyz)
        
        assert not torch.any(torch.isnan(x_emb)), "NaN detected in x_emb"
        
        h_combine = self.combine_mlp(torch.cat([x_emb, phase_emb], dim=-1))
        
        d_xyz = self.xyz_warp(h_combine)
        d_scaling = self.scaling_warp(h_combine)
        
        # \omega = |\vec{v}|
        # q = (\cos(\omega/2), \frac{\vec{v}}{\omega} \cdot \sin(\omega/2))
        #   = (\cos(\omega/2), vec{v}/2 \cdot \text{sinc}(\frac{\omega}{2\pi}))
        # torch.sinc(x) = sin(pi*x)/(pi*x)
        axial_angle: torch.Tensor = self.axial_angle_warp(h_combine).float()
        omega = torch.sqrt(torch.sum(axial_angle**2, dim=-1, keepdim=True) + 1e-10)
        q_w = torch.cos(omega / 2.)
        q_v = axial_angle / 2. * torch.sinc(omega / (2. * torch.pi))
        
        d_rotation = torch.cat([q_w, q_v], dim=-1).to(xyz)
        d_rotation = torch.nn.functional.normalize(d_rotation)
        
        return d_xyz, d_scaling, d_rotation