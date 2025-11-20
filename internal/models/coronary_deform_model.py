"""
Copied from https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/utils/time_utils.py
"""

import torch
import torch.nn as nn
from internal.encodings.vector_positional_encoding import VectorPositionalEncoding
from internal.utils.network_factory import NetworkFactory
from internal.encodings.xray_phase_encoding import PhaseEncoding
from internal.utils.rigid_utils import exp_se3


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


class DeformModel(nn.Module):
    def __init__(
            self,
            network_factory: NetworkFactory,
            D=8,
            W=256,
            multires=10,
            t_D=0,
            t_W=0,
            t_multires=6,
            is_6dof=False,
            chunk: int = -1,
            init_value: float = 1e-5,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.t_multires = t_multires
        self.chunk = chunk

        self.skips = [D // 2]

        self.embed_phase_fn, embed_phase_output_ch = get_phase_embedder(network_factory, self.t_multires, 1, n_layers=t_D, n_neurons=t_W)
        self.embed_fn, embed_xyz_output_ch = get_spatical_embedder(multires, 3)
        mlp_input_ch = embed_phase_output_ch + embed_xyz_output_ch

        # build deformable field
        skip_layer_list = []
        initialized_layers = 0
        n_input_dims = mlp_input_ch
        for i in self.skips:
            n_layers = i - initialized_layers + (1 if initialized_layers == 0 else 0)
            skip_layer_list.append(network_factory.get_network(
                n_input_dims=n_input_dims,
                n_output_dims=W,
                n_layers=n_layers,
                n_neurons=W,
                activation="ReLU",
                output_activation="ReLU",
            ))
            n_input_dims = W + mlp_input_ch
            initialized_layers += n_layers
        self.skip_layers = nn.ModuleList(skip_layer_list)
        self.output_linear = network_factory.get_network(
            n_input_dims=n_input_dims,
            n_output_dims=W,
            n_layers=D - initialized_layers,
            n_neurons=W,
            activation="ReLU",
            output_activation="ReLU",
        )

        self.is_6dof = is_6dof

        if is_6dof:
            self.branch_w = network_factory.get_linear(W, 3)
            self.branch_v = network_factory.get_linear(W, 3)
        else:
            self.gaussian_warp = network_factory.get_linear(W, 3)
        self.gaussian_rotation = network_factory.get_linear(W, 4)
        self.gaussian_scaling = network_factory.get_linear(W, 3)

        # initialize all learnable parameters to a small constant to avoid large random init
        self._init_params(init_value)

    def _init_params(self, val: float = 1e-5):
        """Set all learnable parameters to a constant small value.

        This loops over named parameters to ensure any parameter provided by
        custom modules returned by `network_factory` are also initialized.
        """
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.requires_grad:
                    p.data.fill_(val)

    def forward(self, x, phase):
        phase_emb = self.embed_phase_fn(phase)
        x_emb = self.embed_fn(x)

        if self.chunk > 0:
            chunks = []
            n_gaussians = x.shape[0]
            for i in range(0, n_gaussians, self.chunk):
                chunks.append((
                    phase_emb[i:i + self.chunk],
                    x_emb[i:i + self.chunk]
                ))
        else:
            chunks = [(phase_emb, x_emb)]

        d_xyz_chunks = []
        scaling_chunks = []
        rotation_chunks = []
        # query deformable field
        for chunk in chunks:
            h = torch.cat(chunk, dim=-1)
            for i, layer in enumerate(self.skip_layers):
                h = layer(h)
                h = torch.cat([*chunk, h], -1)
            h = self.output_linear(h)

            if self.is_6dof:
                w = self.branch_w(h)
                v = self.branch_v(h)
                theta = torch.norm(w, dim=-1, keepdim=True)
                w = w / theta + 1e-5
                v = v / theta + 1e-5
                screw_axis = torch.cat([w, v], dim=-1)
                d_xyz = exp_se3(screw_axis, theta)
            else:
                d_xyz = self.gaussian_warp(h)
            scaling = self.gaussian_scaling(h)
            rotation = self.gaussian_rotation(h)

            d_xyz_chunks.append(d_xyz)
            scaling_chunks.append(scaling)
            rotation_chunks.append(rotation)

        return torch.concat(d_xyz_chunks, dim=0), torch.concat(rotation_chunks, dim=0), torch.concat(scaling_chunks, dim=0)
