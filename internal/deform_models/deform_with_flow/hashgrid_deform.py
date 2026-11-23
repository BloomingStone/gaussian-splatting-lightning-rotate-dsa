"""
Copied from https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/utils/time_utils.py
"""
from dataclasses import dataclass
from typing import Any, override

import torch
import torch.nn as nn
from torch.nn import functional as F
from .deform_with_flow import DeformWithFlowModel, DeformWithFlowConfig, DeformsWithFlow
from ..network_factory import NetworkFactory
from ...encodings.vector_positional_encoding import VectorPositionalEncoding



class MLP(nn.Module):
    def __init__(
        self,
        D: int,
        W: int,
        layers: int,
        input_ch: int,
        output_ch: int,
        net_factory: NetworkFactory,
    ):
        super().__init__()
                
        def _layer(ch_in: int) -> nn.Module:
            return net_factory.get_network(
                n_input_dims=ch_in,
                n_output_dims=W,
                n_layers=D,
                n_neurons=W,   # n_neurons == ch_out == W
                activation="ReLU",
                output_activation="ReLU",
            )
        
        input_dims = [input_ch] + [input_ch+W] * (layers-1)
        self.skip_layers = nn.ModuleList([_layer(in_dim) for in_dim in input_dims])
        self.out_layer = net_factory.get_linear(W+input_ch, output_ch)
    
    def forward(self, x: torch.Tensor):
        h = x
        for layer in self.skip_layers:
            h = layer(h)
            h = torch.cat([x, h], dim=-1)
        return self.out_layer(h)
        
class Scale(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s
    def forward(self, x):
        return x * self.s


class ScaleShift(nn.Module):
    """y = a * x + b, used to remap tanh output from [-1,1] to [b-a, b+a]."""
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
    def forward(self, x):
        return self.a * x + self.b

@dataclass
class HashGridDeformConfig(DeformWithFlowConfig):
    D: int = 2  # deepth for each layer of MLP
    
    t_multires: int = 6
    phase_multires: int|None = None
    
    # If True, the t and phase branches will be combined in the same MLP to output `h``
    # Otherwise: 
    #   x_emb + t_emb -> h_flow -> d_density;
    #   x_emb + phase_emb -> h_mov -> d_xyz, d_rot, d_scale;
    t_phase_combined: bool = True
    
    # Two-stage motion / flow separation:
    #   enable_flow=False  -> freeze the t-guided flow branch (embed_t, combine_mlp,
    #                         density_warp) and keep d_density at zero (stage 1).
    #   freeze_motion=True -> freeze everything except the t-guided flow branch
    #                         (stage 2, after loading stage-1 weights).
    enable_flow: bool = True
    freeze_motion: bool = False
    
    combine_layers: int = 4
    combine_W: int = 128

    d_scaling_min: float = -1.0
    """Lower bound of d_scaling; new_scale = scale * (1 + d_scaling).
    -1.0 (default) allows full compression (original behaviour); raising it
    (e.g. -0.5) prevents needle artifacts from over-compressed scales."""

    # Disable the rotation warp entirely: d_rotation is forced to the identity
    # quaternion.  Rotation output may couple with viewpoint and overfit.
    enable_rotation: bool = True

    # Spatial-adaptive smoothness on the displacement field.  The insight is
    # that most of the volume is static and motion is local: a uniform smoothness
    # term over-smooths the moving vessel while wasting capacity on the static
    # background.  We therefore penalise ||du/dx||^2 weighted by (1 - normalized
    # density), so low-density (static background) regions are forced smooth/rigid
    # while high-density (vessel) regions keep fine deformation freedom.
    enable_field_smooth: bool = False
    smooth_density_q_lo: float = 0.05
    smooth_density_q_hi: float = 0.95
    smooth_weight_floor: float = 0.05   # keep a tiny weight everywhere so the field is never fully unconstrained
    smooth_eps: float = 1e-2            # finite-difference step (fraction of scene scale) for the smoothness probe

    tcnn: bool = True
    
    x_multires: int = 7
    n_features_per_level: int = 4
    log2_hashmap_size: int = 7
    base_resolution: int = 16
    max_resolution: int = 128

    # Coarse-to-fine level scheduling: progressively activate hashgrid level
    # groups to prevent high-frequency overfitting (coronary fragmentation)
    # before the low-frequency vessel skeleton is established.
    hashgrid_coarse_to_fine: bool = False
    hashgrid_ctf_groups: tuple[int, ...] = (5, 4, 4)   # levels per group (sum == x_multires)
    hashgrid_ctf_schedule: tuple[int, ...] = (4000, 12000)  # step at which group 1..k-1 activates

    def instantiate(self, *args, **kwargs) -> Any:
        return HashGridDefromModel(self)


class HashGridDefromModel(DeformWithFlowModel):
    def __init__(
            self,
            cfg: HashGridDeformConfig = HashGridDeformConfig(),
    ):
        super().__init__(cfg)
        self.network_factory = NetworkFactory(tcnn=cfg.tcnn)
        self.cfg = cfg

        if self.cfg.hashgrid_coarse_to_fine:
            from ..network_factory import CoarseToFineHashGridEncoding
            self.embed_fn = CoarseToFineHashGridEncoding(
                n_levels=self.cfg.x_multires,
                n_features_per_level=self.cfg.n_features_per_level,
                log2_hashmap_size=self.cfg.log2_hashmap_size,
                base_resolution=self.cfg.base_resolution,
                max_resolution=self.cfg.max_resolution,
                group_levels=self.cfg.hashgrid_ctf_groups,
            )
        else:
            self.embed_fn = self.network_factory.get_hashgrid_encoding(
                n_levels            =self.cfg.x_multires,
                n_features_per_level=self.cfg.n_features_per_level,
                log2_hashmap_size   =self.cfg.log2_hashmap_size,
                base_resolution     =self.cfg.base_resolution,
                max_resolution      =self.cfg.max_resolution
            )
        emb_x_ch = self.embed_fn.get_output_n_channels()
        
        self.embed_t_fn = VectorPositionalEncoding(
            input_channels=1,
            n_frequencies=self.cfg.t_multires,
        )
        emb_t_ch = self.embed_t_fn.get_output_n_channels()
        
        if self.cfg.phase_multires is not None:
            self.embed_phase_fn = VectorPositionalEncoding(
                input_channels=1,
                n_frequencies=self.cfg.phase_multires,
            )
            emb_phase_ch = self.embed_phase_fn.get_output_n_channels()
        else:
            self.embed_phase_fn = None
            emb_phase_ch = 0
        
        def _mlp(W: int, layers: int, input_ch: int):
            return MLP(
                D           =   self.cfg.D,
                W           =   W,
                layers      =   layers,
                input_ch    =   input_ch,
                output_ch   =   W,
                net_factory =   self.network_factory,
            )
        
        if self.cfg.t_phase_combined:
            self.combine_mlp = _mlp(
                W=self.cfg.combine_W, 
                layers=self.cfg.combine_layers, 
                input_ch = (emb_x_ch + emb_t_ch + emb_phase_ch)
            )
            self.combine_phase_mlp = None
        else:
            assert self.embed_phase_fn is not None, "phase_multires must be set to a positive integer to enable phase encoding branch."
            self.combine_mlp = _mlp(
                W=self.cfg.combine_W, 
                layers=self.cfg.combine_layers, 
                input_ch = (emb_t_ch + emb_x_ch)
            )
            self.combine_phase_mlp = _mlp(
                W=self.cfg.combine_W,
                layers=self.cfg.combine_layers,
                input_ch = (emb_phase_ch + emb_x_ch),
            )
        
        _linear = self.network_factory.get_linear
        self.coronary_props_warp = _linear(emb_x_ch, 1)
        
        self.xyz_warp = _linear(self.cfg.combine_W, 3)
        # new_scaling = scaling * (1 + d_scaling); remap tanh [-1,1] to
        # [d_scaling_min, 1] to bound scale compression.
        _dmin = self.cfg.d_scaling_min
        self.scaling_warp = nn.Sequential(
            _linear(self.cfg.combine_W, 3),
            nn.Tanh(),
            ScaleShift((1.0 - _dmin) / 2.0, (1.0 + _dmin) / 2.0),
        )
        
        _linear(self.cfg.combine_W, 3)
        if self.cfg.enable_rotation:
            self.axial_angle_warp = nn.Sequential(
                _linear(self.cfg.combine_W, 3),
                nn.Tanh(),      # w = |\vec{v}| <= sqrt(1+1+1) = 1.73 rad = 99.2 degree  
            )
        else:
            self.axial_angle_warp = None
        
        self.density_warp = nn.Sequential(
            _linear(self.cfg.combine_W, 1),
            Scale(0.001),    # limit the density to match original scale of density, which is around 0.001 for x-ray observation
        )

        self._last_regularization: dict[str, torch.Tensor] | None = None

    def set_step(self, step: int) -> None:
        """Update coarse-to-fine level gates from the current training step.

        Group 0 is always active; group k (k >= 1) activates once ``step``
        reaches ``hashgrid_ctf_schedule[k-1]``.  No-op when coarse-to-fine is
        disabled (the default embedding has no per-level gates).
        """
        if not self.cfg.hashgrid_coarse_to_fine:
            return
        n_groups = len(self.cfg.hashgrid_ctf_groups)
        assert len(self.cfg.hashgrid_ctf_schedule) == n_groups - 1, (
            f"hashgrid_ctf_schedule 长度应为组数-1={n_groups - 1}，"
            f"实际 {len(self.cfg.hashgrid_ctf_schedule)}"
        )
        gates = [1.0] + [1.0 if step >= thr else 0.0 for thr in self.cfg.hashgrid_ctf_schedule]
        self.embed_fn.set_gates(gates)

    def _flow_param_prefixes(self) -> tuple[str, ...]:
        """Parameter-name prefixes that belong to the t-guided flow branch."""
        return ("embed_t_fn.", "combine_mlp.", "density_warp.")

    @override
    def configure_trainable(self) -> None:
        if self.cfg.enable_flow and not self.cfg.freeze_motion:
            return

        for p in self.parameters():
            p.requires_grad = True

        flow_prefixes = self._flow_param_prefixes()

        if not self.cfg.enable_flow:
            assert not self.cfg.t_phase_combined, (
                "enable_flow=False requires t_phase_combined=False: when combined, "
                "combine_mlp is shared by the motion and flow branches and cannot be frozen."
            )
            for name, p in self.named_parameters():
                if name.startswith(flow_prefixes):
                    p.requires_grad = False

        if self.cfg.freeze_motion:
            assert not self.cfg.t_phase_combined, (
                "freeze_motion=True requires t_phase_combined=False: when combined, "
                "combine_mlp is shared by the motion and flow branches and cannot be frozen."
            )
            for name, p in self.named_parameters():
                if not name.startswith(flow_prefixes):
                    p.requires_grad = False
        

    def forward(
        self, 
        xyz: torch.Tensor,
        t: torch.Tensor,
        phase: torch.Tensor|None = None,
        density: torch.Tensor|None = None,
    )-> DeformsWithFlow:
        """
        Forward pass of the deformable model.
        Args:
            xyz: [n, 3] the coordinates of the points to be deformed
            t: [1] the current time, normalized to [0, 1]
            density: [n, 1] base density of the gaussians (opacity-related), used
                only to build the spatial weight of the smoothness regularizer.
        Returns:
            d_xyz: [n, 3] \\in R^3, the translation for each point
            d_scaling: [n, 3] \\in [-1, 1], the scaling change for each point, where the new scaling will be scaling * (1 + d_scaling)
            d_rotation: [n, 4] \\in [-0.1*\\sqrt{3}, 0.1*\\sqrt{3}], the rotation for each point in quaternion format (w, x, y, z), where the new rotation will be rotation * d_rotation
            d_density: [n, 1], the density change for each point, where the new density will be density + d_density
        """
        compute_smooth = self.cfg.enable_field_smooth and self.training and density is not None

        x_emb = self.embed_fn(xyz)
        # assert not torch.any(torch.isnan(x_emb)), "NaN detected in x_emb"
        
        t_emb = self.embed_t_fn(t)
        
        if phase is None:
            if self.embed_phase_fn is not None:
                import warnings
                warnings.simplefilter("once", category=UserWarning)
                warnings.warn("phase is not provided but phase_multires is set. The phase encoder branch will be disabled.")
            h = self.combine_mlp(torch.cat([x_emb, t_emb], dim=-1))
            h_mov = h
            h_flow = h
        else:
            assert self.embed_phase_fn is not None, "phase is provided but phase_multires is not set. Please set phase_multires to a positive integer to enable phase encoding."
            phase_emb = self.embed_phase_fn(phase)
            if self.combine_phase_mlp is not None:
                h_flow = self.combine_mlp(torch.cat([x_emb, t_emb], dim=-1))
                h_mov = self.combine_phase_mlp(torch.cat([x_emb, phase_emb], dim=-1))   # cadiac motion is related to the cardiac phase
            else:
                h = self.combine_mlp(torch.cat([x_emb, t_emb, phase_emb], dim=-1))
                h_mov = h
                h_flow = h
        
        
        d_xyz = self.xyz_warp(h_mov)
        d_scaling = self.scaling_warp(h_mov)
        
        if self.axial_angle_warp is not None:
            # \omega = |\vec{v}|
            # q = (\cos(\omega/2), \frac{\vec{v}}{\omega} \cdot \sin(\omega/2))
            #   = (\cos(\omega/2), vec{v}/2 \cdot \text{sinc}(\frac{\omega}{2\pi}))
            # torch.sinc(x) = sin(pi*x)/(pi*x)
            axial_angle: torch.Tensor = self.axial_angle_warp(h_mov).float()
            omega = torch.sqrt(torch.sum(axial_angle**2, dim=-1, keepdim=True) + 1e-10)
            q_w = torch.cos(omega / 2.)
            q_v = axial_angle / 2. * torch.sinc(omega / (2. * torch.pi))
            
            d_rotation = torch.cat([q_w, q_v], dim=-1).to(xyz)
            d_rotation = torch.nn.functional.normalize(d_rotation)
        else:
            # rotation warp disabled: identity quaternion, no rotation change
            d_rotation = xyz.new_zeros((xyz.shape[0], 4))
            d_rotation[:, 0] = 1.0
        
        if self.cfg.enable_flow:
            d_density = self.density_warp(h_flow)
        else:
            # stage 1: motion-only, keep density unchanged
            d_density = torch.zeros((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)

        if compute_smooth:
            # Finite-difference smoothness probe.  tcnn's hash encoding does not
            # support double backward, so we cannot take an analytic Jacobian via
            # autograd.grad(..., create_graph=True).  Instead we evaluate the
            # displacement at a randomly perturbed neighbour and penalise the
            # directional derivative: du/dn ~= (u(x+eps*n) - u(x)) / eps.  This
            # only requires ordinary (first-order) autograd, so gradients flow
            # back to the network parameters through the extra forward pass.
            eps = self.cfg.smooth_eps
            noise = F.normalize(torch.randn_like(xyz), dim=-1)
            xyz_p = xyz.detach() + eps * noise
            d_xyz_p = self._dxyz_at(xyz_p, t, phase)  # [N, 3]
            dir_deriv = (d_xyz_p - d_xyz) / eps        # [N, 3]

            # spatial weight: low density (static background) -> high weight
            d = density.detach().squeeze(-1)  # [N]
            lo = torch.quantile(d, self.cfg.smooth_density_q_lo)
            hi = torch.quantile(d, self.cfg.smooth_density_q_hi)
            span = (hi - lo).clamp_min(1e-6)
            d_norm = ((d - lo) / span).clamp(0.0, 1.0)
            w = (1.0 - d_norm).clamp_min(self.cfg.smooth_weight_floor)  # [N]

            smoothness = (w[:, None] * (dir_deriv ** 2)).mean()

            # manifold: fold-over penalty on det(I + du/dn * n^T) is too weak;
            # use the squared displacement of the probe instead as a cheap proxy
            # is not meaningful.  Keep a light L2 on the directional derivative.
            self._last_regularization = {
                "smoothness": smoothness,
                "manifold": torch.zeros((), device=xyz.device, dtype=xyz.dtype),
            }
        else:
            self._last_regularization = None

        return DeformsWithFlow(d_xyz=d_xyz, d_scaling=d_scaling, d_rotation=d_rotation, d_density=d_density)

    def _dxyz_at(
        self,
        xyz: torch.Tensor,
        t: torch.Tensor,
        phase: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute the motion displacement d_xyz at xyz.  Used by the main forward
        and re-used by the finite-difference smoothness probe (which needs the
        displacement at a perturbed location)."""
        x_emb = self.embed_fn(xyz)
        t_emb = self.embed_t_fn(t)
        if phase is None:
            h = self.combine_mlp(torch.cat([x_emb, t_emb], dim=-1))
            h_mov = h
        else:
            phase_emb = self.embed_phase_fn(phase)
            if self.combine_phase_mlp is not None:
                h_mov = self.combine_phase_mlp(torch.cat([x_emb, phase_emb], dim=-1))
            else:
                h_mov = self.combine_mlp(torch.cat([x_emb, t_emb, phase_emb], dim=-1))
        return self.xyz_warp(h_mov)

    def get_last_regularization(self) -> dict[str, torch.Tensor] | None:
        return self._last_regularization