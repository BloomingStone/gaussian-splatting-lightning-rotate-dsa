"""
Copied from https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/utils/time_utils.py
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from internal.utils.network_factory import NetworkFactory
from internal.encodings.vector_positional_encoding import VectorPositionalEncoding


def get_phase_embedder(
    multires: int, 
    input_ch: int
) -> tuple[nn.Module, int]:
    phase_encoder = VectorPositionalEncoding(input_channels=input_ch, n_frequencies=multires)
    return phase_encoder, phase_encoder.get_output_n_channels()


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
        

@dataclass
class DeformModelConfig:
    D: int = 2  # deepth for each layer of MLP
    
    t_multires: int = 6
    
    combine_layers: int = 4
    combine_W: int = 128
    
    tcnn: bool = True
    
    x_multires: int = 8
    n_features_per_level: int = 4
    log2_hashmap_size: int = 19
    base_resolution: int = 16
    max_resolution: int = 2048

class DeformModel(nn.Module):
    def __init__(
            self,
            cfg: DeformModelConfig = DeformModelConfig(),
    ):
        super().__init__()
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
        
        self.embed_phase_fn, emb_t_ch = get_phase_embedder(self.cfg.t_multires, input_ch=1)
        
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
        self.scaling_warp = _linear(self.cfg.combine_W, 3)
        self.axial_angle_warp = _linear(self.cfg.combine_W, 3)


    def forward(
            self, 
            xyz: torch.Tensor,
            phase: torch.Tensor,
        )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        phase_emb = self.embed_phase_fn(phase)
        x_emb = self.embed_fn(xyz)
        
        assert not torch.any(torch.isnan(x_emb)), "NaN detected in x_emb"
        coronary_props = F.sigmoid(self.coronary_props_warp(x_emb))
        
        h_combine = self.combine_mlp(torch.cat([x_emb, phase_emb], dim=-1))
        
        d_xyz = self.xyz_warp(h_combine) * coronary_props
        d_scaling = self.scaling_warp(h_combine) * coronary_props
        
        # \omega = |\vec{v}|
        # q = (\cos(\omega/2), \frac{\vec{v}}{\omega} \cdot \sin(\omega/2))
        #   = (\cos(\omega/2), vec{v}/2 \cdot \text{sinc}(\frac{\omega}{2\pi}))
        # torch.sinc(x) = sin(pi*x)/(pi*x)
        axial_angle: torch.Tensor = self.axial_angle_warp(h_combine)  * coronary_props
        axial_angle = axial_angle.float()
        omega = torch.sqrt(torch.sum(axial_angle**2, dim=-1, keepdim=True) + 1e-10)
        q_w = torch.cos(omega / 2.)
        q_v = axial_angle / 2. * torch.sinc(omega / (2. * torch.pi))
        
        d_rotation = torch.cat([q_w, q_v], dim=-1).to(xyz)
        
        return d_xyz, d_scaling, d_rotation, coronary_props