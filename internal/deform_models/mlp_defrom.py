from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .deform_model import DeformModel, DefromModelConfig
from ..encodings.vector_positional_encoding import VectorPositionalEncoding
from ..utils.rigid_utils import exp_se3


@dataclass
class MLPDefromConfig(DefromModelConfig):
    D: int = 8
    W: int = 256

    x_multires: int = 10
    phase_n_frequencies: int = 1

    is_6dof: bool = False
    chunk: int = -1

    def instantiate(self, *args, **kwargs) -> Any:
        return MLPDefromModel(self)


class MLPDefromModel(DeformModel):
    def __init__(self, cfg: MLPDefromConfig = MLPDefromConfig()):
        super().__init__(cfg)
        self.cfg = cfg

        if self.cfg.D < 2:
            raise ValueError(f"D must be >= 2, got {self.cfg.D}")

        self.embed_xyz_fn = VectorPositionalEncoding(
            input_channels=3,
            n_frequencies=self.cfg.x_multires,
            log_sampling=True,
        )
        self.embed_phase_fn = VectorPositionalEncoding(
            input_channels=1,
            n_frequencies=self.cfg.phase_n_frequencies,
        )

        xyz_ch = self.embed_xyz_fn.get_output_n_channels()
        phase_ch = self.embed_phase_fn.get_output_n_channels()
        self.input_ch = xyz_ch + phase_ch

        self.skips = [self.cfg.D // 2]
        self.skip_layers = nn.ModuleList()

        initialized_layers = 0
        n_input_dims = self.input_ch
        for i in self.skips:
            n_layers = i - initialized_layers + (1 if initialized_layers == 0 else 0)
            self.skip_layers.append(self._make_mlp_block(n_input_dims, self.cfg.W, n_layers))
            n_input_dims = self.cfg.W + self.input_ch
            initialized_layers += n_layers

        self.output_linear = self._make_mlp_block(
            n_input_dims,
            self.cfg.W,
            self.cfg.D - initialized_layers,
        )

        self.is_6dof = self.cfg.is_6dof
        if self.is_6dof:
            self.branch_w = nn.Linear(self.cfg.W, 3)
            self.branch_v = nn.Linear(self.cfg.W, 3)
        else:
            self.xyz_warp = nn.Linear(self.cfg.W, 3)

        self.scaling_warp = nn.Sequential(
            nn.Linear(self.cfg.W, 3),
            nn.Tanh(),
        )
        self.axial_angle_warp = nn.Sequential(
            nn.Linear(self.cfg.W, 3),
            nn.Tanh(),
        )

    @staticmethod
    def _make_mlp_block(in_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        layers: list[nn.Module] = []
        cur_in = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(cur_in, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            cur_in = hidden_dim
        return nn.Sequential(*layers)

    @staticmethod
    def _axial_angle_to_quaternion(axial_angle: torch.Tensor) -> torch.Tensor:
        omega = torch.sqrt(torch.sum(axial_angle**2, dim=-1, keepdim=True) + 1e-10)
        q_w = torch.cos(omega / 2.0)
        q_v = axial_angle / 2.0 * torch.sinc(omega / (2.0 * torch.pi))
        d_rotation = torch.cat([q_w, q_v], dim=-1)
        return torch.nn.functional.normalize(d_rotation)

    def _forward_chunk(self, x_input: torch.Tensor, phase_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.cat([x_input, phase_input], dim=-1)
        for l in self.skip_layers:
            h = l(h)
            h = torch.cat([x_input, phase_input, h], dim=-1)
        h = self.output_linear(h)

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            theta_safe = theta + 1e-5
            w = w / theta_safe
            v = v / theta_safe
            screw_axis = torch.cat([w, v], dim=-1)
            transform = exp_se3(screw_axis, theta.squeeze(-1))
            d_xyz = transform[:, :3, 3]
        else:
            d_xyz = self.xyz_warp(h)

        d_scaling = self.scaling_warp(h)
        axial_angle = self.axial_angle_warp(h).float()
        d_rotation = self._axial_angle_to_quaternion(axial_angle).to(d_xyz)
        return d_xyz, d_scaling, d_rotation

    def forward(
        self,
        xyz: torch.Tensor,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if phase.ndim == 1:
            phase = phase.unsqueeze(-1)
        if phase.shape[-1] != 1:
            phase = phase[..., :1]

        if phase.shape[0] == 1 and xyz.shape[0] != 1:
            phase = phase.expand(xyz.shape[0], -1)
        if phase.shape[0] != xyz.shape[0]:
            raise RuntimeError(
                f"Batch mismatch between xyz and phase: {xyz.shape[0]} vs {phase.shape[0]}"
            )

        x_emb = self.embed_xyz_fn(xyz)
        phase_emb = self.embed_phase_fn(phase)

        if self.cfg.chunk > 0:
            d_xyz_list: list[torch.Tensor] = []
            d_scaling_list: list[torch.Tensor] = []
            d_rotation_list: list[torch.Tensor] = []
            for i in range(0, xyz.shape[0], self.cfg.chunk):
                d_xyz, d_scaling, d_rotation = self._forward_chunk(
                    x_emb[i:i + self.cfg.chunk],
                    phase_emb[i:i + self.cfg.chunk],
                )
                d_xyz_list.append(d_xyz)
                d_scaling_list.append(d_scaling)
                d_rotation_list.append(d_rotation)

            return (
                torch.cat(d_xyz_list, dim=0),
                torch.cat(d_scaling_list, dim=0),
                torch.cat(d_rotation_list, dim=0),
            )

        return self._forward_chunk(x_emb, phase_emb)
