"""
Copied from https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/utils/time_utils.py
"""
from typing import override
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F
from internal.encodings.vector_positional_encoding import VectorPositionalEncoding
from internal.utils.network_factory import NetworkFactory
from internal.encodings.xray_phase_encoding import PhaseEncoding


def get_spatical_embedder(multires: int, input_ch: int=1) -> tuple[nn.Module, int]:
    encoder = VectorPositionalEncoding(input_channels=input_ch, n_frequencies=multires, log_sampling=True)
    return encoder, encoder.get_output_n_channels()


def get_phase_embedder(
    network_factory: NetworkFactory, 
    multires: int, 
    input_ch: int = 1, 
    n_layers: int = 0, 
    n_neurons: int = 0
) -> tuple[nn.Module, int]:
    phase_encoder = PhaseEncoding(input_channels=input_ch, n_frequencies=multires)
    output_ch = phase_encoder.get_output_n_channels()
    if n_layers <= 0 or n_neurons <= 0:
        return phase_encoder, output_ch
    return TimeNetwork(network_factory, phase_encoder, D=n_layers, W=n_neurons, output_ch=output_ch), output_ch


class TimeNetwork(nn.Module):
    def __init__(self, network_factory, encoding, D=2, W=256, output_ch=30):
        super().__init__()
        self.embed_time_fn = encoding

        self.timenet = network_factory.get_network(
            n_input_dims=encoding.get_output_n_channels(),
            n_output_dims=output_ch,
            n_layers=D,
            n_neurons=W,
            activation="ReLU",
            output_activation="None",  # vanilla implementation does not have ReLU on output layer
        )

    def forward(self, t):
        return self.timenet(self.embed_time_fn(t))


@dataclass
class DeformModelConfig:
    D: int = 8
    W: int = 256
    multires: int = 10
    layers: list[int] = field(default_factory=lambda: [4, 4])
    t_D: int = 0
    t_W: int = 0
    t_multires: int = 6

    def __post_init__(self):
        assert all([l > 0 for l in self.layers]), "All layer sizes must be positive."
        assert sum(self.layers) == self.D, "Sum of layer sizes must equal D."

class DeformModel(nn.Module):
    def __init__(
            self,
            network_factory: NetworkFactory,
            cfg: DeformModelConfig = DeformModelConfig(),
    ):
        super().__init__()
        self.network_factory = network_factory
        self.cfg = cfg
        
        mlp_input_ch = self._build_embedding()
        self._build_hidden_layers(mlp_input_ch)
        self._build_deform_linears()


    def _build_embedding(self) -> int:
        self.embed_phase_fn, embed_phase_output_ch = get_phase_embedder(
            self.network_factory, 
            self.cfg.t_multires, 
            n_layers=self.cfg.t_D, 
            n_neurons=self.cfg.t_W
        )
        self.embed_fn, embed_xyz_output_ch = get_spatical_embedder(self.cfg.multires, input_ch=3)
        mlp_input_ch = embed_phase_output_ch + embed_xyz_output_ch
        return mlp_input_ch
    
    
    def _build_hidden_layers(self, mlp_input_ch: int):
        W, D = self.cfg.W, self.cfg.D
        layers = self.cfg.layers[:-1]   # exclude the last layer for output_linear
        last_layer_size = D - sum(layers)
        
        input_dims = [mlp_input_ch + W for _ in layers]
        input_dims[0] = mlp_input_ch    # first layer input dim is mlp_input_ch
        
        def _linear(ch_i: int, d: int):
            return self.network_factory.get_network(
                n_input_dims=ch_i,
                n_output_dims=W,
                n_layers=d,
                n_neurons=W,
                activation="ReLU",
                output_activation="ReLU",
            )
            
        self.skip_layers = nn.ModuleList([
            _linear(in_dim, d) 
            for in_dim, d in zip(input_dims, layers)
        ])
        
        self.output_linear = _linear(mlp_input_ch + W, last_layer_size)
    
    
    def _build_deform_linears(self):
        W = self.cfg.W
        _linear = self.network_factory.get_linear
        self.gaussian_warp = _linear(W, 3)
        self.gaussian_scaling = _linear(W, 3)
        self.gaussian_rotation = _linear(W, 4)

    def forward(
            self, 
            xyz: torch.Tensor, 
            phase: torch.Tensor,
        )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._forward_h(xyz, phase)

        d_xyz = self.gaussian_warp(h)
        d_scaling = self.gaussian_scaling(h)
        d_rotation = F.normalize(self.gaussian_rotation(h), dim=-1) # normalize to unit quaternion
        
        return d_xyz, d_scaling, d_rotation
    
    
    def _forward_h(self, xyz: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        phase_emb = self.embed_phase_fn(phase)
        x_emb = self.embed_fn(xyz)
        
        # query deformable field
        h = torch.cat((phase_emb, x_emb), dim=-1)
        for layer in self.skip_layers:
            h = layer(h)
            h = torch.cat([phase_emb, x_emb, h], dim=-1)
        h = self.output_linear(h)
        return h.to(xyz.dtype)