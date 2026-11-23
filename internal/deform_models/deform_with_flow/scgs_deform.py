"""SC-GS style sparse-control-point deformation field.

Implements the deformation model of SC-GS (Sparse-Controlled Gaussian
Splatting, CVPR 2024, arXiv:2312.14937) adapted to cardiac-phase-driven motion:

* A small set of **learnable control points** (``n_control_points``, e.g. 64),
  each carrying a rigid 6-DoF transform (rotation + translation) that varies
  with the cardiac phase.
* Each Gaussian is deformed by a **skinning-weight blend** of its K nearest
  control points: ``mu' = sum_k w_k (R_k (mu - p_k) + p_k + t_k)``, where
  ``w_k`` comes from a tiny MLP and is softmax-normalised over the K neighbours.
* The rotation of each Gaussian is blended from the same weights (quaternion
  mixture, sign-corrected).

Because the number of control points is far smaller than the number of
Gaussians, the displacement field is low-frequency by construction — this
directly counteracts the high-frequency overfitting that fragments thin
tubular structures (the coronary) in the hash-grid field, while the per-point
6-DoF rigid blend keeps local fidelity higher than a pure B-spline field.

The control-point locations themselves are optimised (they are ``nn.Parameter``
initialised by farthest-point sampling on the first forward pass).
"""
from dataclasses import dataclass, field
from typing import Any, override

import torch
import torch.nn as nn
from torch.nn import functional as F

from .deform_with_flow import DeformWithFlowModel, DeformWithFlowConfig, DeformsWithFlow
from ...encodings.vector_positional_encoding import VectorPositionalEncoding


def _skew(v: torch.Tensor) -> torch.Tensor:
    """[..., 3] -> [..., 3, 3] skew-symmetric cross-product matrix."""
    zero = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], zero], dim=-1),
    ], dim=-2)


def _rodrigues(rotvec: torch.Tensor) -> torch.Tensor:
    """[..., 3] axis-angle -> [..., 3, 3] rotation matrix (Rodrigues formula)."""
    theta = rotvec.norm(dim=-1, keepdim=True).clamp_min(1e-10)  # [..., 1]
    v = rotvec / theta  # unit axis [..., 3]
    K = _skew(v)  # [..., 3, 3]
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    sin_t = torch.sin(theta)[..., None]  # [..., 1, 1]
    cos_t = torch.cos(theta)[..., None]  # [..., 1, 1]
    return eye + sin_t * K + (1.0 - cos_t) * (K @ K)


def _axial_angle_to_quaternion(axial_angle: torch.Tensor) -> torch.Tensor:
    """[..., 3] axis-angle -> [..., 4] unit quaternion (w, x, y, z)."""
    omega = torch.sqrt(torch.sum(axial_angle ** 2, dim=-1, keepdim=True) + 1e-10)
    q_w = torch.cos(omega / 2.0)
    q_v = axial_angle / 2.0 * torch.sinc(omega / (2.0 * torch.pi))
    q = torch.cat([q_w, q_v], dim=-1)
    return F.normalize(q, dim=-1)


@dataclass
class SparseControlDeformConfig(DeformWithFlowConfig):
    n_control_points: int = 64
    """Number of learnable control points (far fewer than the Gaussians)."""

    k_nearest: int = 4
    """K nearest control points blended per Gaussian."""

    phase_multires: int = 6
    """Frequency count for the cardiac-phase positional encoding."""

    max_rotation: float = 0.3
    """rad; soft clamp of the per-control-point rotation magnitude."""

    disp_init_std: float = 1e-3
    rotvec_init_std: float = 1e-3
    scale_init_std: float = 1e-3
    scale_max: float = 0.5
    """soft clamp of per-control-point log-scale; new_scale = scale * (1 + d_scaling)."""

    skinning_mlp_layers: int = 2
    skinning_mlp_width: int = 32
    skinning_temp: float = 1.0
    """Softmax temperature for the skinning weights."""

    # two-stage motion/flow compatibility (stage 1: motion-only, d_density = 0)
    enable_flow: bool = True
    freeze_motion: bool = False

    def instantiate(self, *args, **kwargs) -> Any:
        return SparseControlDeformModel(self)


class SparseControlDeformModel(DeformWithFlowModel):
    def __init__(self, cfg: SparseControlDeformConfig = SparseControlDeformConfig()):
        super().__init__(cfg)
        self.cfg = cfg

        self.embed_phase_fn = VectorPositionalEncoding(
            input_channels=1,
            n_frequencies=cfg.phase_multires,
        )
        n_phase_basis = self.embed_phase_fn.get_output_n_channels()

        # per-control-point motion coefficients: [M, J, 3]
        self.disp_coeff = nn.Parameter(
            torch.randn(cfg.n_control_points, n_phase_basis, 3) * cfg.disp_init_std
        )
        self.rotvec_coeff = nn.Parameter(
            torch.randn(cfg.n_control_points, n_phase_basis, 3) * cfg.rotvec_init_std
        )
        self.scale_coeff = nn.Parameter(
            torch.randn(cfg.n_control_points, n_phase_basis, 3) * cfg.scale_init_std
        )

        # learnable control-point locations; filled by FPS on first forward pass
        self.control_points = nn.Parameter(torch.zeros(cfg.n_control_points, 3))
        self.register_buffer("_cp_initialized", torch.tensor(False), persistent=False)

        # skinning-weight MLP: (rel, xyz) -> scalar weight
        layers: list[nn.Module] = []
        in_ch = 6
        for i in range(cfg.skinning_mlp_layers):
            out_ch = cfg.skinning_mlp_width
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(nn.ReLU())
            in_ch = out_ch
        layers.append(nn.Linear(in_ch, 1))
        self.skinning_mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # lazy control-point initialisation (farthest-point sampling)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _maybe_init_control_points(self, xyz: torch.Tensor) -> None:
        if self._cp_initialized.item():
            return
        n = xyz.shape[0]
        m = min(self.cfg.n_control_points, n)
        idx = torch.zeros(m, dtype=torch.long, device=xyz.device)
        idx[0] = torch.randint(0, n, (1,), device=xyz.device)
        dists = torch.full((n,), float("inf"), device=xyz.device, dtype=xyz.dtype)
        for i in range(1, m):
            d = ((xyz - xyz[idx[i - 1]]) ** 2).sum(-1)
            dists = torch.minimum(dists, d)
            idx[i] = torch.argmax(dists)
        self.control_points.data = xyz[idx].clone()
        self._cp_initialized.fill_(True)

    # ------------------------------------------------------------------
    def forward(
        self,
        xyz: torch.Tensor,
        t: torch.Tensor,
        phase: torch.Tensor | None = None,
        density: torch.Tensor | None = None,
    ) -> DeformsWithFlow:
        self._maybe_init_control_points(xyz)

        phase = self.get_phase(t, phase)
        if phase.ndim == 1:
            phase = phase.unsqueeze(-1)
        if phase.shape[-1] != 1:
            phase = phase[..., :1]

        t_basis = self.embed_phase_fn(phase)  # [N, J]
        if t_basis.shape[0] == 1 and xyz.shape[0] != 1:
            t_basis = t_basis.expand(xyz.shape[0], -1)
        if t_basis.shape[0] != xyz.shape[0]:
            raise RuntimeError(
                f"Batch mismatch between xyz and phase basis: {xyz.shape[0]} vs {t_basis.shape[0]}"
            )

        # per-control-point motion at each point's phase: [N, M, 3]
        cp_disp = torch.einsum("mjc,nj->nmc", self.disp_coeff, t_basis)
        cp_rotvec = torch.einsum("mjc,nj->nmc", self.rotvec_coeff, t_basis)
        cp_rotvec = torch.tanh(cp_rotvec) * self.cfg.max_rotation
        cp_scale = torch.einsum("mjc,nj->nmc", self.scale_coeff, t_basis)
        cp_scale = torch.tanh(cp_scale) * self.cfg.scale_max

        # K nearest control points per Gaussian
        dist = torch.cdist(xyz, self.control_points)  # [N, M]
        k = min(self.cfg.k_nearest, self.cfg.n_control_points)
        _, knn_idx = torch.topk(dist, k=k, dim=1, largest=False)  # [N, K]
        idx_exp = knn_idx.unsqueeze(-1).expand(-1, -1, 3)

        p_k = self.control_points[knn_idx]                    # [N, K, 3]
        disp_k = cp_disp.gather(1, idx_exp)                   # [N, K, 3]
        rotvec_k = cp_rotvec.gather(1, idx_exp)               # [N, K, 3]
        scale_k = cp_scale.gather(1, idx_exp)                 # [N, K, 3]

        # skinning weights
        rel = xyz.unsqueeze(1) - p_k                          # [N, K, 3]
        feat = torch.cat([rel, xyz.unsqueeze(1).expand(-1, k, -1)], dim=-1)  # [N, K, 6]
        w = self.skinning_mlp(feat).squeeze(-1)               # [N, K]
        w = F.softmax(w / self.cfg.skinning_temp, dim=-1)     # [N, K]

        # blend of rigid transforms: mu' = sum_k w_k (R_k (mu - p_k) + p_k + t_k)
        R_k = _rodrigues(rotvec_k)                            # [N, K, 3, 3]
        R_rel = (R_k @ rel.unsqueeze(-1)).squeeze(-1)         # [N, K, 3]
        transformed = R_rel + p_k + disp_k                    # [N, K, 3]
        xyz_new = (w.unsqueeze(-1) * transformed).sum(dim=1)  # [N, 3]
        d_xyz = xyz_new - xyz

        # quaternion mixture (sign-corrected against the first neighbour)
        q_k = _axial_angle_to_quaternion(rotvec_k)            # [N, K, 4]
        q_ref = q_k[:, :1, :]
        sign = torch.where((q_ref * q_k).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        q_k = q_k * sign
        q_mix = (w.unsqueeze(-1) * q_k).sum(dim=1)            # [N, 4]
        d_rotation = F.normalize(q_mix, dim=-1)

        # pure rigid motion: no scale change -> replaced by per-control-point
        # log-scale blend (SC-GS predicts d_scaling too; X-ray reconstruction
        # relies on the scale degree of freedom to concentrate density along rays)
        d_scaling = (w.unsqueeze(-1) * scale_k).sum(dim=1)    # [N, 3]

        # stage 1: motion-only (no flow branch yet)
        d_density = torch.zeros((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)

        return DeformsWithFlow(
            d_xyz=d_xyz,
            d_scaling=d_scaling,
            d_rotation=d_rotation,
            d_density=d_density,
        )
