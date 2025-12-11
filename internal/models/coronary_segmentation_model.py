"""
Copied from https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/utils/time_utils.py
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F
from internal.encodings.vector_positional_encoding import VectorPositionalEncoding
from internal.utils.network_factory import NetworkFactory


def get_spatical_embedder(multires: int, input_ch: int=1) -> tuple[nn.Module, int]:
    encoder = VectorPositionalEncoding(input_channels=input_ch, n_frequencies=multires, log_sampling=True)
    return encoder, encoder.get_output_n_channels()


@dataclass
class SegModelConfig:
    D: int = 8
    W: int = 256
    multires: int = 10
    layers: list[int] = field(default_factory=lambda: [4, 4])
    ch_color: int = 1
    ch_motion: int = 7

    def __post_init__(self):
        assert all([l > 0 for l in self.layers]), "All layer sizes must be positive."
        assert sum(self.layers) == self.D, "Sum of layer sizes must equal D."

class SegModel(nn.Module):
    def __init__(
            self,
            network_factory: NetworkFactory,
            cfg: SegModelConfig = SegModelConfig(),
    ):
        super().__init__()
        self.network_factory = network_factory
        self.cfg = cfg
        
        mlp_input_ch = self._build_embedding()
        self._build_hidden_layers(mlp_input_ch)
        self._build_warp()


    def _build_embedding(self) -> int:
        self.embed_fn, embed_xyz_output_ch = get_spatical_embedder(self.cfg.multires, input_ch=3)
        
        mlp_input_ch = embed_xyz_output_ch + self.cfg.ch_color + self.cfg.ch_motion * 2
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
    
    
    def _build_warp(self):
        W = self.cfg.W
        self.warp = self.network_factory.get_linear(W, 1)

    def forward(
            self, 
            xyz: torch.Tensor,
            gray: torch.Tensor,
            motion_mean: torch.Tensor,
            motion_var: torch.Tensor
    ):
        tensor = self.embed_fn(xyz)
        tensor = torch.cat((tensor, gray, motion_mean, motion_var), dim=-1)
        
        h = tensor
        for layer in self.skip_layers:
            h = layer(h)
            h = torch.cat([tensor, h], dim=-1)
        h = self.output_linear(h)

        return torch.sigmoid(self.warp(h))
    