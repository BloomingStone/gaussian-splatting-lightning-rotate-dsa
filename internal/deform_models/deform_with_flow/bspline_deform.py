"""Low-frequency B-spline deformation field for stage-1 cardiac motion.

Instead of a hash-grid MLP predicting per-point ``d_xyz / d_rotation / d_scale``
directly, this model parameterises a **smooth control-point displacement field**
``u(x, phase)`` and derives:

* ``d_xyz = u``,
* ``d_rotation`` from the rotation part of the deformation gradient
  ``F = I + ∇u`` (small-angle curl approximation, clamped),
* ``d_scale = 0`` (no scaling, keeps the field close to isometric).

The field is low-frequency by construction (cubic B-spline over a coarse grid),
and two regularizers are exposed to encourage a smooth / manifold (diffeomorphic)
deformation:

* ``manifold``: penalises ``det(F) < eps`` (fold-overs),
* ``smoothness``: discrete-Laplacian penalty on the control-point coefficients.

The flow branch (t-guided ``d_density``) is kept for two-stage compatibility and
mirrors :class:`HashGridDefromModel`.
"""
from dataclasses import dataclass
from typing import Any, override

import torch
import torch.nn as nn
from torch.nn import functional as F

from .deform_with_flow import DeformWithFlowModel, DeformWithFlowConfig, DeformsWithFlow
from .hashgrid_deform import MLP, Scale
from ..network_factory import NetworkFactory
from ...encodings.vector_positional_encoding import VectorPositionalEncoding


@dataclass
class BSplineDeformConfig(DeformWithFlowConfig):
    # --- motion field ---
    grid_resolution: tuple[int, int, int] = (8, 8, 8)
    phase_multires: int = 6
    disp_init_std: float = 1e-3
    bounds_padding_ratio: float = 0.05
    xyz_min: tuple[float, float, float] | None = None
    xyz_max: tuple[float, float, float] | None = None
    max_rotation: float = 0.3        # rad, soft clamp of |omega|
    manifold_eps: float = 0.1        # det(F) below this is penalised

    # --- flow branch (same as hashgrid) ---
    t_multires: int = 6
    flow_x_multires: int = 13
    flow_log2_hashmap_size: int = 19
    flow_base_resolution: int = 16
    flow_max_resolution: int = 2048
    flow_n_features_per_level: int = 4
    flow_mlp_layers: int = 4
    flow_mlp_width: int = 128
    tcnn: bool = True

    enable_flow: bool = True
    freeze_motion: bool = False

    def instantiate(self, *args, **kwargs) -> Any:
        return BSplineDeformModel(self)


class BSplineDeformModel(DeformWithFlowModel):
    def __init__(self, cfg: BSplineDeformConfig = BSplineDeformConfig()):
        super().__init__(cfg)
        self.cfg = cfg
        self.network_factory = NetworkFactory(tcnn=cfg.tcnn)

        # --- motion: low-frequency B-spline displacement field ---
        self.embed_phase_fn = VectorPositionalEncoding(input_channels=1, n_frequencies=cfg.phase_multires)
        n_phase_basis = self.embed_phase_fn.get_output_n_channels()

        nx, ny, nz = cfg.grid_resolution
        self.grid_resolution = (nx, ny, nz)
        self.n_control_points = nx * ny * nz
        self.disp_coeff = nn.Parameter(torch.randn(self.n_control_points, n_phase_basis, 3) * cfg.disp_init_std)

        self.register_buffer("grid_min", torch.zeros(3), persistent=False)
        self.register_buffer("grid_max", torch.ones(3), persistent=False)
        self._grid_bounds_initialized = False

        # --- flow branch (t-guided density change) ---
        self.flow_embed_fn = self.network_factory.get_hashgrid_encoding(
            n_levels=cfg.flow_x_multires,
            n_features_per_level=cfg.flow_n_features_per_level,
            log2_hashmap_size=cfg.flow_log2_hashmap_size,
            base_resolution=cfg.flow_base_resolution,
            max_resolution=cfg.flow_max_resolution,
        )
        flow_x_ch = self.flow_embed_fn.get_output_n_channels()

        self.flow_embed_t_fn = VectorPositionalEncoding(input_channels=1, n_frequencies=cfg.t_multires)
        flow_t_ch = self.flow_embed_t_fn.get_output_n_channels()

        self.flow_mlp = MLP(
            D=2,
            W=cfg.flow_mlp_width,
            layers=cfg.flow_mlp_layers,
            input_ch=flow_x_ch + flow_t_ch,
            output_ch=cfg.flow_mlp_width,
            net_factory=self.network_factory,
        )
        self.density_warp = nn.Sequential(
            self.network_factory.get_linear(cfg.flow_mlp_width, 1),
            Scale(0.001),
        )

        self._last_regularization: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # trainable-flag control (two-stage)
    # ------------------------------------------------------------------
    def _flow_param_prefixes(self) -> tuple[str, ...]:
        return ("flow_embed_fn.", "flow_embed_t_fn.", "flow_mlp.", "density_warp.")

    @override
    def configure_trainable(self) -> None:
        if self.cfg.enable_flow and not self.cfg.freeze_motion:
            return

        for p in self.parameters():
            p.requires_grad = True

        flow_prefixes = self._flow_param_prefixes()

        if not self.cfg.enable_flow:
            for name, p in self.named_parameters():
                if name.startswith(flow_prefixes):
                    p.requires_grad = False

        if self.cfg.freeze_motion:
            for name, p in self.named_parameters():
                if not name.startswith(flow_prefixes):
                    p.requires_grad = False

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        xyz: torch.Tensor,
        t: torch.Tensor,
        phase: torch.Tensor | None = None,
    ) -> DeformsWithFlow:
        self._maybe_init_grid_bounds(xyz)

        phase = self.get_phase(t, phase)
        if phase.ndim == 1:
            phase = phase.unsqueeze(-1)
        if phase.shape[-1] != 1:
            phase = phase[..., :1]

        # Analytic Jacobian of the B-spline displacement w.r.t. xyz. It does not
        # rely on autograd, so it works under both torch.no_grad() (in-loop
        # validation) and torch.inference_mode() (gs-val / evaluate), where
        # torch.enable_grad() can NOT re-enable gradient tracking.
        u, jac = self._displacement_with_jacobian(xyz, phase)

        d_xyz = u
        d_scaling = torch.zeros_like(u)
        d_rotation = self._rotation_from_jacobian(jac)

        # flow branch
        if self.cfg.enable_flow:
            x_emb = self.flow_embed_fn(xyz)
            t_emb = self.flow_embed_t_fn(t)
            h = self.flow_mlp(torch.cat([x_emb, t_emb], dim=-1))
            d_density = self.density_warp(h)
        else:
            d_density = torch.zeros((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)

        # field regularizers (manifold + smoothness), exposed to the renderer
        det_f = torch.linalg.det(torch.eye(3, device=xyz.device, dtype=xyz.dtype).unsqueeze(0) + jac)
        manifold = F.softplus(self.cfg.manifold_eps - det_f).mean()
        self._last_regularization = {
            "manifold": manifold,
            "smoothness": self._grid_laplacian_penalty(),
        }

        return DeformsWithFlow(d_xyz=d_xyz, d_scaling=d_scaling, d_rotation=d_rotation, d_density=d_density)

    def get_last_regularization(self) -> dict[str, torch.Tensor] | None:
        return self._last_regularization

    # ------------------------------------------------------------------
    # motion field helpers
    # ------------------------------------------------------------------
    def _displacement(self, xyz: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        basis = self.embed_phase_fn(phase)                 # [N, n_phase_basis]
        return self._interpolate_bspline(self.disp_coeff, basis, xyz)  # [N, 3]

    def _displacement_with_jacobian(
        self, xyz: torch.Tensor, phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        basis = self.embed_phase_fn(phase)                 # [N, n_phase_basis]
        return self._interpolate_bspline(self.disp_coeff, basis, xyz, return_grad=True)

    def _rotation_from_jacobian(self, jac: torch.Tensor) -> torch.Tensor:
        # infinitesimal rotation vector = 0.5 * curl(u)
        omega = torch.stack([
            0.5 * (jac[:, 2, 1] - jac[:, 1, 2]),
            0.5 * (jac[:, 0, 2] - jac[:, 2, 0]),
            0.5 * (jac[:, 1, 0] - jac[:, 0, 1]),
        ], dim=-1)  # [N, 3]

        omega_norm = torch.sqrt((omega ** 2).sum(dim=-1, keepdim=True) + 1e-10)
        scale = torch.tanh(omega_norm / self.cfg.max_rotation) * self.cfg.max_rotation / (omega_norm + 1e-10)
        omega = omega * scale
        return self._axial_angle_to_quaternion(omega)

    def _grid_laplacian_penalty(self) -> torch.Tensor:
        """Discrete-Laplacian magnitude penalty on the control-point coefficients."""
        nx, ny, nz = self.grid_resolution
        c = self.disp_coeff.reshape(nx, ny, nz, *self.disp_coeff.shape[1:])
        lap = torch.zeros((), device=c.device, dtype=c.dtype)
        for dim in range(3):
            plus = torch.roll(c, 1, dims=dim)
            minus = torch.roll(c, -1, dims=dim)
            lap = lap + ((plus - 2.0 * c + minus) ** 2).mean()
        return lap

    # ------------------------------------------------------------------
    # B-spline interpolation
    # ------------------------------------------------------------------
    def _interpolate_bspline(
        self, cp_coeff: torch.Tensor, basis: torch.Tensor, xyz: torch.Tensor,
        return_grad: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Interpolate the B-spline control-point displacement at ``xyz``.

        ``cp_coeff``: [M, n_phase_basis, 3] (raw control-point coefficients),
        ``basis``: [N, n_phase_basis] (phase encoding).

        Control points are gathered *first* (only the 4^3 neighbouring
        coefficients per point) and then mixed with ``basis``, so the peak
        intermediate tensor is O(N * 64 * n_basis * 3) instead of O(N * M * 3)
        — the latter OOMs for large grids (e.g. 128^3).
        """
        nx, ny, nz = self.grid_resolution

        denom = torch.clamp(self.grid_max - self.grid_min, min=1e-6)
        uvw = (xyz - self.grid_min) / denom
        uvw = torch.clamp(uvw, min=0.0, max=1.0)
        scale = torch.tensor([nx - 1, ny - 1, nz - 1], dtype=xyz.dtype, device=xyz.device)
        grid_pos = uvw * scale

        ix_base, wx = self._cubic_bspline_indices_and_weights(grid_pos[:, 0], nx)
        iy_base, wy = self._cubic_bspline_indices_and_weights(grid_pos[:, 1], ny)
        iz_base, wz = self._cubic_bspline_indices_and_weights(grid_pos[:, 2], nz)

        offsets = torch.arange(4, device=xyz.device)
        ix = torch.clamp(ix_base[:, None] + offsets[None, :], min=0, max=nx - 1)
        iy = torch.clamp(iy_base[:, None] + offsets[None, :], min=0, max=ny - 1)
        iz = torch.clamp(iz_base[:, None] + offsets[None, :], min=0, max=nz - 1)

        linear_idx = (
            ix[:, :, None, None] * (ny * nz)
            + iy[:, None, :, None] * nz
            + iz[:, None, None, :]
        ).reshape(xyz.shape[0], -1).long()

        weights = torch.einsum("ni,nj,nk->nijk", wx, wy, wz).reshape(xyz.shape[0], -1)

        # gather neighbouring control-point coefficients: [N, 64, n_basis, 3]
        sampled = cp_coeff[linear_idx]
        # per-point displacement after mixing with the phase basis: [N, 64, 3]
        cp_disp = torch.einsum("nkbc,nb->nkc", sampled, basis)
        u = (weights.unsqueeze(-1) * cp_disp).sum(dim=1)  # [N, 3]

        if not return_grad:
            return u

        # Analytic Jacobian d u_i / d xyz_j (derivative of the cubic B-spline
        # basis, chain-ruled through grid_pos = (xyz - grid_min) / denom * (res-1)).
        d_scale = scale / denom  # [3]
        dwx = self._cubic_bspline_deriv(grid_pos[:, 0], nx) * d_scale[0]
        dwy = self._cubic_bspline_deriv(grid_pos[:, 1], ny) * d_scale[1]
        dwz = self._cubic_bspline_deriv(grid_pos[:, 2], nz) * d_scale[2]

        w_dx = torch.einsum("ni,nj,nk->nijk", dwx, wy, wz).reshape(xyz.shape[0], -1)
        w_dy = torch.einsum("ni,nj,nk->nijk", wx, dwy, wz).reshape(xyz.shape[0], -1)
        w_dz = torch.einsum("ni,nj,nk->nijk", wx, wy, dwz).reshape(xyz.shape[0], -1)

        du_dx = (w_dx.unsqueeze(-1) * cp_disp).sum(dim=1)  # [N, 3]
        du_dy = (w_dy.unsqueeze(-1) * cp_disp).sum(dim=1)
        du_dz = (w_dz.unsqueeze(-1) * cp_disp).sum(dim=1)
        jac = torch.stack([du_dx, du_dy, du_dz], dim=-1)   # [N, 3, 3], jac[i, j] = d u_i / d x_j
        return u, jac

    @staticmethod
    def _cubic_bspline_indices_and_weights(grid_coord: torch.Tensor, resolution: int) -> tuple[torch.Tensor, torch.Tensor]:
        _ = resolution
        left = torch.floor(grid_coord).to(torch.long)
        frac = grid_coord - left.to(grid_coord.dtype)
        i0 = left - 1

        one_minus = 1.0 - frac
        w0 = (one_minus ** 3) / 6.0
        w1 = (3.0 * frac ** 3 - 6.0 * frac ** 2 + 4.0) / 6.0
        w2 = (-3.0 * frac ** 3 + 3.0 * frac ** 2 + 3.0 * frac + 1.0) / 6.0
        w3 = (frac ** 3) / 6.0

        w = torch.stack([w0, w1, w2, w3], dim=-1)
        return i0, w

    @staticmethod
    def _cubic_bspline_deriv(grid_coord: torch.Tensor, resolution: int) -> torch.Tensor:
        """Derivative of the cubic B-spline weights w.r.t. grid_coord (== w.r.t. frac)."""
        _ = resolution
        left = torch.floor(grid_coord).to(torch.long)
        frac = grid_coord - left.to(grid_coord.dtype)

        one_minus = 1.0 - frac
        d0 = -(one_minus ** 2) / 2.0
        d1 = (3.0 * frac ** 2 - 4.0 * frac) / 2.0
        d2 = (-3.0 * frac ** 2 + 2.0 * frac + 1.0) / 2.0
        d3 = (frac ** 2) / 2.0

        return torch.stack([d0, d1, d2, d3], dim=-1)

    def _maybe_init_grid_bounds(self, xyz: torch.Tensor) -> None:
        if self._grid_bounds_initialized:
            return

        if self.cfg.xyz_min is not None and self.cfg.xyz_max is not None:
            grid_min = torch.tensor(self.cfg.xyz_min, dtype=xyz.dtype, device=xyz.device)
            grid_max = torch.tensor(self.cfg.xyz_max, dtype=xyz.dtype, device=xyz.device)
        else:
            xyz_min = xyz.min(dim=0).values
            xyz_max = xyz.max(dim=0).values
            span = torch.clamp(xyz_max - xyz_min, min=1e-6)
            pad = span * self.cfg.bounds_padding_ratio
            grid_min = xyz_min - pad
            grid_max = xyz_max + pad

        self.grid_min = grid_min
        self.grid_max = grid_max
        self._grid_bounds_initialized = True

    @staticmethod
    def _axial_angle_to_quaternion(axial_angle: torch.Tensor) -> torch.Tensor:
        omega = torch.sqrt(torch.sum(axial_angle ** 2, dim=-1, keepdim=True) + 1e-10)
        q_w = torch.cos(omega / 2.0)
        q_v = axial_angle / 2.0 * torch.sinc(omega / (2.0 * torch.pi))

        d_rotation = torch.cat([q_w, q_v], dim=-1).to(axial_angle)
        return torch.nn.functional.normalize(d_rotation)
