from typing import Literal
import torch
from torch import nn
import numpy as np


class NetworkWithSkipLayers(torch.nn.Module):
    def __init__(self, skip_layers, output_layers) -> None:
        super().__init__()

        self.skip_layers = skip_layers
        self.output_layers = output_layers

    def forward(self, x):
        input = x
        for i in self.skip_layers:
            y = i(input)
            input = torch.concat([x, y], dim=-1)
        return self.output_layers(input)


class PyTorchHashGridEncoding(nn.Module):
    """PyTorch implementation of multi-resolution hash grid encoding"""
    def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16, max_resolution=2048):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        
        # Calculate growth factor
        # Uniform in log space to ensure smooth transition between levels
        # \log r_l = \log r_{base} + l \cdot \frac{ \log r_{max} - \log r_{base} }{ L_n - 1 }
        # r_l = r_{base} \exp(\frac{ \log r_{max} - \log r_{base} }{ L_n - 1 }) ^ l
        if n_levels > 1:
            self.growth_factor = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1))
        else:
            self.growth_factor = 1.0
        
        # Create hash tables for each level
        self.hash_tables = nn.ModuleList()
        self.resolutions = []
        
        for level in range(n_levels):
            resolution = int(base_resolution * (self.growth_factor ** level))
            self.resolutions.append(resolution)
            hash_size = 2 ** log2_hashmap_size
            # Each hash table stores features for grid points
            hash_table = nn.Embedding(hash_size, n_features_per_level)
            # Initialize with small random values
            nn.init.normal_(hash_table.weight, 0, 1e-4)
            self.hash_tables.append(hash_table)
    
    def hash_function(self, coords, resolution):
        """Hash function for 3D coordinates"""
        # coords: (..., 3) in [0, 1]
        scaled_coords = coords * resolution
        # Get integer coordinates
        int_coords = torch.floor(scaled_coords).long()
        
        # Simple hash function
        hash_val = int_coords[..., 0] * 73856093 + int_coords[..., 1] * 19349663 + int_coords[..., 2] * 83492791
        hash_val = hash_val % (2 ** self.log2_hashmap_size)
        return hash_val
    
    def trilinear_interpolation(self, coords, resolution, hash_table):
        """Trilinear interpolation for one level"""
        # coords: (..., 3) in [0, 1]
        scaled_coords = coords * resolution
        int_coords = torch.floor(scaled_coords)
        frac_coords = scaled_coords - int_coords
        
        # Get 8 corner indices
        corners = torch.stack([
            int_coords + torch.tensor([0, 0, 0], device=coords.device, dtype=torch.long),
            int_coords + torch.tensor([1, 0, 0], device=coords.device, dtype=torch.long),
            int_coords + torch.tensor([0, 1, 0], device=coords.device, dtype=torch.long),
            int_coords + torch.tensor([1, 1, 0], device=coords.device, dtype=torch.long),
            int_coords + torch.tensor([0, 0, 1], device=coords.device, dtype=torch.long),
            int_coords + torch.tensor([1, 0, 1], device=coords.device, dtype=torch.long),
            int_coords + torch.tensor([0, 1, 1], device=coords.device, dtype=torch.long),
            int_coords + torch.tensor([1, 1, 1], device=coords.device, dtype=torch.long),
        ], dim=-2)  # Shape: (..., 8, 3)
        
        # Hash all corners
        corner_hashes = []
        for i in range(8):
            corner_coords = corners[..., i, :] / resolution  # Normalize back to [0, 1]
            corner_hash = self.hash_function(corner_coords, resolution)
            corner_hashes.append(corner_hash)
        corner_hashes = torch.stack(corner_hashes, dim=-1)  # Shape: (..., 8)
        
        # Get features for all corners
        corner_features = hash_table(corner_hashes)  # Shape: (..., 8, n_features)
        
        # Trilinear weights
        weights = torch.stack([
            (1 - frac_coords[..., 0]) * (1 - frac_coords[..., 1]) * (1 - frac_coords[..., 2]),
            frac_coords[..., 0] * (1 - frac_coords[..., 1]) * (1 - frac_coords[..., 2]),
            (1 - frac_coords[..., 0]) * frac_coords[..., 1] * (1 - frac_coords[..., 2]),
            frac_coords[..., 0] * frac_coords[..., 1] * (1 - frac_coords[..., 2]),
            (1 - frac_coords[..., 0]) * (1 - frac_coords[..., 1]) * frac_coords[..., 2],
            frac_coords[..., 0] * (1 - frac_coords[..., 1]) * frac_coords[..., 2],
            (1 - frac_coords[..., 0]) * frac_coords[..., 1] * frac_coords[..., 2],
            frac_coords[..., 0] * frac_coords[..., 1] * frac_coords[..., 2],
        ], dim=-1)  # Shape: (..., 8)
        
        # Interpolate
        result = torch.sum(corner_features * weights.unsqueeze(-1), dim=-2)
        return result
    
    def forward(self, x):
        assert x.min() >= 0 and x.max() <= 1, "Input coordinates should be normalized to [0, 1]"
        # Encode at each level and concatenate
        encodings = []
        for level in range(self.n_levels):
            level_encoding = self.trilinear_interpolation(x, self.resolutions[level], self.hash_tables[level])
            encodings.append(level_encoding)
        
        return torch.cat(encodings, dim=-1)
    
    def get_output_n_channels(self):
        return self.n_levels * self.n_features_per_level


class HashGridEncoding(nn.Module):
    def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16, max_resolution=2048, use_tcnn=True):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.use_tcnn = use_tcnn
        
        # Calculate growth factor
        if n_levels > 1:
            self.growth_factor = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1))
        else:
            self.growth_factor = 1.0
        
        # Scene bounds: -150 to 150
        self.scene_min = -150.0
        self.scene_max = 150.0
        self.scene_range = self.scene_max - self.scene_min
        
        if use_tcnn:
            import tinycudann as tcnn
            self.tcnn_available = True
            self.encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": self.n_levels,
                    "n_features_per_level": self.n_features_per_level,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": self.base_resolution,
                    "per_level_scale": self.growth_factor,
                    "interpolation": "Smoothstep"
                }
            )
        else:
            self.tcnn_available = False
            self.encoder = PyTorchHashGridEncoding(
                n_levels=n_levels,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolution,
                max_resolution=max_resolution
            )
    
    def forward(self, x):
        # Normalize coordinates from [-128, 128] to [0, 1]
        x_norm = (x - self.scene_min) / self.scene_range
        x_norm = x_norm.clamp(0.0, 1.0)
        return self.encoder(x_norm)
    
    def get_output_n_channels(self):
        return self.n_levels * self.n_features_per_level


class NetworkFactory:
    seed: int = 1337

    def __init__(self, tcnn: bool = True):
        self.tcnn = tcnn

    def _get_seed(self):
        try:
            return NetworkFactory.seed
        finally:
            NetworkFactory.seed += 1

    def get_linear(self, in_features: int, out_features: int):
        return self.get_network(
            n_input_dims=in_features,
            n_output_dims=out_features,
            n_layers=1,
            n_neurons=out_features,
            activation="ReLU",
            output_activation="None",
        )

    def get_hashgrid_encoding(self, n_levels=16, n_features_per_level=4, log2_hashmap_size=19, base_resolution=16, max_resolution=2048):
        """Get hashgrid encoding for spatial coordinates"""
        return HashGridEncoding(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            use_tcnn=self.tcnn
        )

    def get_network(
            self,
            n_input_dims: int,
            n_output_dims: int,
            n_layers: int,
            n_neurons: int,
            activation: Literal["ReLU", "None"],
            output_activation: Literal["ReLU", "Sigmoid", "None"],
    ):
        assert n_layers > 0 and n_neurons > 0

        if self.tcnn is True:
            import tinycudann as tcnn
            otype = "FullyFusedMLP"
            if n_neurons not in (16, 32, 64, 128) or n_layers == 1:
                otype = "CutlassMLP"
            return tcnn.Network(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                network_config={
                    "otype": otype,
                    "activation": activation,
                    "output_activation": output_activation,
                    "n_neurons": n_neurons,
                    "n_hidden_layers": n_layers - 1,
                },
                seed=self._get_seed(),
            )

        # PyTorch
        model_list = []
        # hidden layers
        in_features = n_input_dims
        for i in range(n_layers - 1):
            model_list += self._get_torch_layer(in_features, n_neurons, activation)
            in_features = n_neurons  # next layer's in_features
        # output layer
        model_list += self._get_torch_layer(in_features, n_output_dims, output_activation)

        return nn.Sequential(*model_list)

    def get_network_with_skip_layers(
            self,
            n_input_dims: int,
            n_output_dims: int,
            n_layers: int,
            n_neurons: int,
            activation: Literal["ReLU", "None"],
            output_activation: Literal["ReLU", "Sigmoid", "None"],
            skips: list[int] = [],
    ):
        original_n_input_dims = n_input_dims

        # build skip layers
        skip_layer_list = []
        initialized_layers = 0
        n_input_dims = original_n_input_dims
        for i in skips:
            n_layers_to_create = i - initialized_layers
            skip_layer_list.append(self.get_network(
                n_input_dims=n_input_dims,
                n_output_dims=n_neurons,
                n_layers=n_layers_to_create,
                n_neurons=n_neurons,
                activation=activation,
                output_activation=activation,
            ))
            n_input_dims = n_neurons + original_n_input_dims
            initialized_layers += n_layers_to_create
        skip_layers = nn.ModuleList(skip_layer_list)

        # build left layers
        output = self.get_network(
            n_input_dims=n_input_dims,
            n_output_dims=n_output_dims,
            n_layers=n_layers - initialized_layers,
            n_neurons=n_neurons,
            activation=activation,
            output_activation=output_activation,
        )

        return NetworkWithSkipLayers(skip_layers, output)

    def _get_torch_activation(self, name: str):
        if name == "None":
            return None
        if name == "ReLU":
            return nn.ReLU()
        if name == "Sigmoid":
            return nn.Sigmoid()
        raise ValueError("unsupported activation type {}".format(name))

    def _get_torch_layer(self, in_features: int, out_features: int, activation_name: str) -> list:
        model_list = []
        layer = nn.Linear(in_features, out_features)
        activation = self._get_torch_activation(activation_name)
        model_list.append(layer)
        if activation is not None:
            model_list.append(activation)

        return model_list
