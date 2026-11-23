import torch
import einops


class VectorPositionalEncoding(torch.nn.Module):
    freq_bands: torch.Tensor
    
    def __init__(self, input_channels: int, n_frequencies: int, log_sampling: bool = True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        The vectorize version of positional encoding.
        """
        super().__init__()
        self.n_frequencies = n_frequencies
        self.input_channels = input_channels
        self.output_channels = input_channels * (2 * n_frequencies + 1)

        max_frequencies = n_frequencies - 1
        if log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_frequencies, steps=n_frequencies)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_frequencies, steps=n_frequencies)
        
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        angles = torch.einsum('BC, F -> BCF', x, self.freq_bands)
        sin_res = torch.sin(angles)
        cos_res = torch.cos(angles)
        sincos_res = torch.stack([sin_res, cos_res], dim=-1) # (B, C, n_frequencies, 2)
        sincos_res = einops.rearrange(sincos_res, 'B C F D -> B (C F D)')
        return torch.cat([x, sincos_res], dim=-1)

    def get_output_n_channels(self) -> int:
        return self.output_channels
