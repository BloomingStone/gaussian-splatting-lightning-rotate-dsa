from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .deform_model import DeformModel, DefromModelConfig
from ..encodings.vector_positional_encoding import VectorPositionalEncoding
from ..encodings.xray_phase_encoding import PhaseEncoding


class SineLayer(nn.Module):
	def __init__(
		self,
		in_features: int,
		out_features: int,
		omega_0: float,
		is_first: bool = False,
	):
		super().__init__()
		self.omega_0 = omega_0
		self.is_first = is_first
		self.linear = nn.Linear(in_features, out_features)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		in_features = self.linear.in_features
		if self.is_first:
			bound = 1.0 / in_features
		else:
			bound = (6.0 / in_features) ** 0.5 / self.omega_0
		with torch.no_grad():
			self.linear.weight.uniform_(-bound, bound)
			self.linear.bias.uniform_(-bound, bound)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.sin(self.omega_0 * self.linear(x))


class SirenBackbone(nn.Module):
	def __init__(
		self,
		input_ch: int,
		hidden_dim: int,
		hidden_layers: int,
		first_omega_0: float,
		hidden_omega_0: float,
	):
		super().__init__()
		if hidden_layers < 1:
			raise ValueError(f"hidden_layers must be >= 1, got {hidden_layers}")

		layers: list[nn.Module] = [
			SineLayer(input_ch, hidden_dim, omega_0=first_omega_0, is_first=True)
		]
		for _ in range(hidden_layers - 1):
			layers.append(
				SineLayer(hidden_dim, hidden_dim, omega_0=hidden_omega_0, is_first=False)
			)
		self.layers = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layers(x)


@dataclass
class SirenDeformConfig(DefromModelConfig):
	hidden_dim: int = 128
	hidden_layers: int = 4
	first_omega_0: float = 30.0
	hidden_omega_0: float = 30.0

	x_n_frequencies: int = 7
	phase_period: float = 1.0
	phase_n_frequencies: int = 1

	def instantiate(self, *args, **kwargs) -> Any:
		return SirenDeformModel(self)


class SirenDeformModel(DeformModel):
	def __init__(self, cfg: SirenDeformConfig = SirenDeformConfig()):
		super().__init__(cfg)
		self.cfg = cfg

		self.embed_xyz_fn = VectorPositionalEncoding(
			input_channels=3,
			n_frequencies=self.cfg.x_n_frequencies,
		)
		self.embed_phase_fn = PhaseEncoding(
			input_channels=1,
			T=self.cfg.phase_period,
			n_frequencies=self.cfg.phase_n_frequencies,
		)

		emb_x_ch = self.embed_xyz_fn.get_output_n_channels()
		emb_phase_ch = self.embed_phase_fn.get_output_n_channels()
		self.backbone = SirenBackbone(
			input_ch=emb_x_ch + emb_phase_ch,
			hidden_dim=self.cfg.hidden_dim,
			hidden_layers=self.cfg.hidden_layers,
			first_omega_0=self.cfg.first_omega_0,
			hidden_omega_0=self.cfg.hidden_omega_0,
		)

		self.xyz_warp = nn.Linear(self.cfg.hidden_dim, 3)
		self.scaling_warp = nn.Sequential(
			nn.Linear(self.cfg.hidden_dim, 3),
			nn.Tanh(),
		)
		self.axial_angle_warp = nn.Sequential(
			nn.Linear(self.cfg.hidden_dim, 3),
			nn.Tanh(),
		)

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
		h = self.backbone(torch.cat([x_emb, phase_emb], dim=-1))

		d_xyz = self.xyz_warp(h)
		d_scaling = self.scaling_warp(h)

		axial_angle = self.axial_angle_warp(h).float()
		omega = torch.sqrt(torch.sum(axial_angle**2, dim=-1, keepdim=True) + 1e-10)
		q_w = torch.cos(omega / 2.0)
		q_v = axial_angle / 2.0 * torch.sinc(omega / (2.0 * torch.pi))

		d_rotation = torch.cat([q_w, q_v], dim=-1).to(xyz)
		d_rotation = torch.nn.functional.normalize(d_rotation)
		return d_xyz, d_scaling, d_rotation
