import torch
from torch import nn
import einops


class PhaseEncoding(nn.Module):
    freq_bands: torch.Tensor
    
    def __init__(
        self, 
        input_channels: int = 1, 
        T: float = 1.,
        n_frequencies: int = 1,
        include_input: bool = False,
    ):
        r"""
        Defines a function that embeds periodic t with Period T to (sin(2\pi f / T t), cos(2\pi f / T t), ...), 
        f = [1, 2, ..., n_frequencies]. Attention, there is no DC component (t) in the output, which is different 
        from the original positional encoding in NeRF.
        
        Args:
            in_channels: number of input channels (3 for both xyz and direction)
            T: period of the phase
            n_frequencies: number of frequencies to use. 
        """
        super().__init__()
        self.input_channels = input_channels
        self.T = T
        self.n_frequencies = n_frequencies
        self.output_channels = input_channels * 2 * n_frequencies

        max_frequencies = n_frequencies
        freq_bands = 2 ** torch.linspace(0, max_frequencies, steps=n_frequencies)
        freq_bands = freq_bands * 2 * torch.pi / T
        
        self.register_buffer("freq_bands", freq_bands)
        
        self.include_input = include_input
        if self.include_input:
            self.output_channels += input_channels
        
    
    def forward(self, t: torch.Tensor):
        r"""
        Embeds t to (sin(2\pi f / T t), cos(2\pi f / T t), ...)
        f = [1, 2, ..., self.n_frequencies]
        
        Inputs:
            t: (B, self.in_channels)
        
        Outputs:
            out: (B, self.out_channels)
        """
        t = torch.fmod(t, self.T)
        angles = torch.einsum('BC, F -> BCF', t, self.freq_bands)
        sin_res = torch.sin(angles)
        cos_res = torch.cos(angles)
        sincos_res = torch.stack([sin_res, cos_res], dim=-1) # (B, C, n_frequencies, 2)
        sincos_res = einops.rearrange(sincos_res, 'B C F D -> B (C F D)')
        if self.include_input:
            return torch.cat([t, sincos_res], dim=-1)
        else:
            return sincos_res
    
    def get_output_n_channels(self) -> int:
        return self.output_channels
