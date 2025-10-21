import torch
from torch import nn
import einops


class PhaseEncoding(nn.Module):
    def __init__(
        self, 
        input_channels: int = 1, 
        T: int = 1,
        n_frequencies: int = 1,
    ):
        r"""
        Defines a function that embeds periodic t with Period T to (t, sin(2\pi f / T t), cos(2\pi f / T t), ...), f = [1, 2, ..., n_frequencies]
        in_channels: number of input channels (3 for both xyz and direction)
        T: period of the phase
        n_frequencies: number of frequencies to use. 
        """
        super().__init__()
        self.input_channels = input_channels
        self.T = T
        self.n_frequencies = n_frequencies
        self.output_channels = input_channels * (2 * n_frequencies + 1)

        max_frequencies = n_frequencies
        self.freq_bands = torch.linspace(1, max_frequencies, steps=n_frequencies)
        self.freq_bands = self.freq_bands * 2 * torch.pi / T
        
    
    def forward(self, t: torch.Tensor):
        r"""
        Embeds t to (t, sin(2\pi f / T t), cos(2\pi f / T t), ...)
        f = [1, 2, ..., self.n_frequencies]
        
        Inputs:
            t: (B, self.in_channels)
        
        Outputs:
            out: (B, self.out_channels)
        """
        angles = torch.einsum('BC, F -> BCF', t, self.freq_bands)
        sin_res = torch.sin(angles)
        cos_res = torch.cos(angles)
        sincos_res = torch.stack([sin_res, cos_res], dim=-1) # (B, C, n_frequencies, 2)
        sincos_res = einops.rearrange(sincos_res, 'B C F D -> B (C F D)')
        return torch.cat([t, sincos_res], dim=-1)
