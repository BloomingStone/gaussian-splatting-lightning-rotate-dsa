from dataclasses import dataclass

import torch
from torch import nn

from .deform_model import DeformModel, DefromModelConfig, Deforms, GSParam
from ..encodings.vector_positional_encoding import VectorPositionalEncoding
from internal.utils.gaussian_utils import GaussianTransformUtils


class SafeExponential(nn.Module):
    def __init__(self, max_value: float = 10.0):
        super().__init__()
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.clamp(x, max=self.max_value))

@dataclass
class ControlPointDeformConfig(DefromModelConfig):
    grid_resolution: tuple[int, int, int] = (10, 10, 10)
    t_n_frequencies: int = 1
    interpolation_method: str = "bspline"
    bounds_padding_ratio: float = 0.05
    xyz_min: tuple[float, float, float] | None = None
    xyz_max: tuple[float, float, float] | None = None
    init_coeff_std: float = 1e-3

    def instantiate(self, *args, **kwargs):
        return ControlPointDeformModel(self)


class ControlPointDeformModel(DeformModel):
    def __init__(self, cfg: ControlPointDeformConfig = ControlPointDeformConfig()):
        super().__init__(cfg)
        self.cfg = cfg

        self.embed_t_fn = VectorPositionalEncoding(
            input_channels=1,
            n_frequencies=cfg.t_n_frequencies,
        )
        self.n_t_basis = self.embed_t_fn.get_output_n_channels()

        nx, ny, nz = cfg.grid_resolution
        self.grid_resolution = (nx, ny, nz)
        self.n_control_points = nx * ny * nz

        std = cfg.init_coeff_std
        self.disp_coeff = nn.Parameter(torch.randn(self.n_control_points, self.n_t_basis, 3) * std)
        self.rotvec_coeff = nn.Parameter(torch.randn(self.n_control_points, self.n_t_basis, 3) * std)
        self.logscale_coeff = nn.Parameter(torch.randn(self.n_control_points, self.n_t_basis, 3) * std)

        self.register_buffer("grid_min", torch.zeros(3), persistent=False)
        self.register_buffer("grid_max", torch.ones(3), persistent=False)
        self._grid_bounds_initialized = False
        
        self.xyz_activation = nn.Identity()
        self.scaling_activation = SafeExponential()
        self.rotation_activation = nn.Tanh()

    def forward(
        self,
        xyz: torch.Tensor,
        t: torch.Tensor,
        phase: torch.Tensor|None = None,
    ) -> Deforms:
        self._maybe_init_grid_bounds(xyz)

        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.shape[-1] != 1:
            t = t[..., :1]

        t_basis = self.embed_t_fn(t)
        if t_basis.shape[0] == 1 and xyz.shape[0] != 1:
            t_basis = t_basis.expand(xyz.shape[0], -1)
        if t_basis.shape[0] != xyz.shape[0]:
            raise RuntimeError(
                f"Batch mismatch between xyz and t basis: {xyz.shape[0]} vs {t_basis.shape[0]}"
            )

        d_xyz = self._interpolate_field(self.disp_coeff, xyz, t_basis)
        interp_rotvec = self._interpolate_field(self.rotvec_coeff, xyz, t_basis)
        interp_logscale = self._interpolate_field(self.logscale_coeff, xyz, t_basis)

        d_xyz = self.xyz_activation(d_xyz)
        d_scaling = self.scaling_activation(interp_logscale)
        d_rotation = self.rotation_activation(interp_rotvec)
        d_rotation = self._axial_angle_to_quaternion(d_rotation)

        return Deforms(d_xyz=d_xyz, d_scaling=d_scaling, d_rotation=d_rotation)

    def _interpolate_field(
        self,
        coeff: torch.Tensor,
        xyz: torch.Tensor,
        t_basis: torch.Tensor,
    ) -> torch.Tensor:
        cp_values = self._predict_control_point_motion(coeff, t_basis)

        method = self.cfg.interpolation_method.lower()
        if method == "bspline":
            return self._interpolate_bspline(cp_values, xyz)
        if method == "rbf":
            return self._interpolate_rbf(cp_values, xyz)
        if method == "tps":
            return self._interpolate_tps(cp_values, xyz)
        raise ValueError(f"Unknown interpolation method: {self.cfg.interpolation_method}")

    def _predict_control_point_motion(self, coeff: torch.Tensor, t_basis: torch.Tensor) -> torch.Tensor:
        # coeff: [M, J, 3], t_basis: [N, J] -> [N, M, 3]
        return torch.einsum("mjc,nj->nmc", coeff, t_basis)

    def _interpolate_bspline(self, cp_values: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        # cp_values: [N, M, 3], xyz: [N, 3] -> [N, 3]
        nx, ny, nz = self.grid_resolution

        denom = torch.clamp(self.grid_max - self.grid_min, min=1e-6)
        uvw = (xyz - self.grid_min) / denom
        uvw = torch.clamp(uvw, min=0.0, max=1.0)
        grid_pos = uvw * torch.tensor([nx - 1, ny - 1, nz - 1], dtype=xyz.dtype, device=xyz.device)

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
        sampled = cp_values.gather(
            dim=1,
            index=linear_idx.unsqueeze(-1).expand(-1, -1, 3),
        )

        return (weights.unsqueeze(-1) * sampled).sum(dim=1)

    def _interpolate_rbf(self, cp_values: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        _ = cp_values, xyz
        raise NotImplementedError("RBF interpolation is reserved for a future implementation")

    def _interpolate_tps(self, cp_values: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        _ = cp_values, xyz
        raise NotImplementedError("TPS interpolation is reserved for a future implementation")

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
        omega = torch.sqrt(torch.sum(axial_angle**2, dim=-1, keepdim=True) + 1e-10)
        q_w = torch.cos(omega / 2.)
        q_v = axial_angle / 2. * torch.sinc(omega / (2. * torch.pi))
        
        d_rotation = torch.cat([q_w, q_v], dim=-1).to(axial_angle)
        d_rotation = torch.nn.functional.normalize(d_rotation)
        return d_rotation
    
    @staticmethod
    def deform(
        source: GSParam,
        deforms: Deforms
    ) -> GSParam:
        xyz = source.xyz + deforms.d_xyz
        scaling = source.scaling * ( 1 + deforms.d_scaling )
        rotation = GaussianTransformUtils.quat_multiply(source.rotation, deforms.d_rotation)
        
        return GSParam(xyz=xyz, rotation=rotation, scaling=scaling)       