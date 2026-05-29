"""
Copied from https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/utils/time_utils.py
"""
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from .deform_with_flow import DeformWithFlowModel, DeformWithFlowConfig, DeformsWithFlow
from ...utils.network_factory import NetworkFactory
from ...encodings.vector_positional_encoding import VectorPositionalEncoding



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
class HashGridDeformConfig(DeformWithFlowConfig):
    D: int = 2  # deepth for each layer of MLP
    
    t_multires: int = 6
    phase_multires: int|None = None
    
    # If True, the t and phase branches will be combined in the same MLP to output `h``
    # Otherwise: 
    #   x_emb + t_emb -> h_flow -> d_density;
    #   x_emb + phase_emb -> h_mov -> d_xyz, d_rot, d_scale;
    t_phase_combined: bool = True
    
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


class HashGridDefromModel(DeformWithFlowModel):
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
        
        self.embed_t_fn = VectorPositionalEncoding(
            input_channels=1,
            n_frequencies=self.cfg.t_multires,
        )
        emb_t_ch = self.embed_t_fn.get_output_n_channels()
        
        if self.cfg.phase_multires is not None:
            self.embed_phase_fn = VectorPositionalEncoding(
                input_channels=1,
                n_frequencies=self.cfg.phase_multires,
            )
            emb_phase_ch = self.embed_phase_fn.get_output_n_channels()
        else:
            self.embed_phase_fn = None
            emb_phase_ch = 0
        
        def _mlp(W: int, layers: int, input_ch: int):
            return MLP(
                D           =   self.cfg.D,
                W           =   W,
                layers      =   layers,
                input_ch    =   input_ch,
                output_ch   =   W,
                net_factory =   self.network_factory,
            )
        
        if self.cfg.t_phase_combined:
            self.combine_mlp = _mlp(
                W=self.cfg.combine_W, 
                layers=self.cfg.combine_layers, 
                input_ch = (emb_x_ch + emb_t_ch + emb_phase_ch)
            )
            self.combine_phase_mlp = None
        else:
            assert self.embed_phase_fn is not None, "phase_multires must be set to a positive integer to enable phase encoding branch."
            self.combine_mlp = _mlp(
                W=self.cfg.combine_W, 
                layers=self.cfg.combine_layers, 
                input_ch = (emb_t_ch + emb_x_ch)
            )
            self.combine_phase_mlp = _mlp(
                W=self.cfg.combine_W,
                layers=self.cfg.combine_layers,
                input_ch = (emb_phase_ch + emb_x_ch),
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
        
        self.density_warp = nn.Sequential(
            _linear(self.cfg.combine_W, 1),
            Scale(0.001),    # limit the density to match original scale of density, which is around 0.001 for x-ray observation
        )
        

    def forward(
        self, 
        xyz: torch.Tensor,
        t: torch.Tensor,
        phase: torch.Tensor|None = None,
    )-> DeformsWithFlow:
        """
        Forward pass of the deformable model.
        Args:
            xyz: [n, 3] the coordinates of the points to be deformed
            t: [1] the current time, normalized to [0, 1]
        Returns:
            d_xyz: [n, 3] \\in R^3, the translation for each point
            d_scaling: [n, 3] \\in [-1, 1], the scaling change for each point, where the new scaling will be scaling * (1 + d_scaling)
            d_rotation: [n, 4] \\in [-0.1*\\sqrt{3}, 0.1*\\sqrt{3}], the rotation for each point in quaternion format (w, x, y, z), where the new rotation will be rotation * d_rotation
            d_density: [n, 1], the density change for each point, where the new density will be density + d_density
        """
        x_emb = self.embed_fn(xyz)
        assert not torch.any(torch.isnan(x_emb)), "NaN detected in x_emb"
        
        t_emb = self.embed_t_fn(t)
        
        if phase is None:
            if self.embed_phase_fn is not None:
                import warnings
                warnings.simplefilter("once", category=UserWarning)
                warnings.warn("phase is not provided but phase_multires is set. The phase encoder branch will be disabled.")
            h = self.combine_mlp(torch.cat([x_emb, t_emb], dim=-1))
            h_mov = h
            h_flow = h
        else:
            assert self.embed_phase_fn is not None, "phase is provided but phase_multires is not set. Please set phase_multires to a positive integer to enable phase encoding."
            phase_emb = self.embed_phase_fn(phase)
            if self.combine_phase_mlp is not None:
                h_flow = self.combine_mlp(torch.cat([x_emb, t_emb], dim=-1))
                h_mov = self.combine_phase_mlp(torch.cat([x_emb, phase_emb], dim=-1))   # cadiac motion is related to the cardiac phase
            else:
                h = self.combine_mlp(torch.cat([x_emb, t_emb, phase_emb], dim=-1))
                h_mov = h
                h_flow = h
        
        
        d_xyz = self.xyz_warp(h_mov)
        d_scaling = self.scaling_warp(h_mov)
        
        # \omega = |\vec{v}|
        # q = (\cos(\omega/2), \frac{\vec{v}}{\omega} \cdot \sin(\omega/2))
        #   = (\cos(\omega/2), vec{v}/2 \cdot \text{sinc}(\frac{\omega}{2\pi}))
        # torch.sinc(x) = sin(pi*x)/(pi*x)
        axial_angle: torch.Tensor = self.axial_angle_warp(h_mov).float()
        omega = torch.sqrt(torch.sum(axial_angle**2, dim=-1, keepdim=True) + 1e-10)
        q_w = torch.cos(omega / 2.)
        q_v = axial_angle / 2. * torch.sinc(omega / (2. * torch.pi))
        
        d_rotation = torch.cat([q_w, q_v], dim=-1).to(xyz)
        d_rotation = torch.nn.functional.normalize(d_rotation)
        
        d_density = self.density_warp(h_flow)

        return DeformsWithFlow(d_xyz=d_xyz, d_scaling=d_scaling, d_rotation=d_rotation, d_density=d_density)