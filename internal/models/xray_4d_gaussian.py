from dataclasses import dataclass, field
from typing import override, Mapping, Callable, Any
from abc import ABC

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from lightning import LightningModule

from .gaussian import (
    Gaussian, 
    GaussianModel, 
    HasVanillaGetters, 
    HasMeanGetter, 
    HasMeanGetter,
    HasScaleGetter,
    HasRotationGetter,
)
from ..utils.general_utils import (
    inverse_sigmoid,
    strip_symmetric,
    build_scaling_rotation,
)
from ..schedulers import ExponentialDecayScheduler
from ..optimizers import OptimizerConfig, Adam

@dataclass
class XrayExponentialDecayScheduler(ExponentialDecayScheduler):
    lr_final = 0.0000016
    max_steps = 30_000


@dataclass
class OptimizationConfig:
    means_lr_init: float = 1e-5
    
    means_lr_scheduler: ExponentialDecayScheduler = field(default_factory=XrayExponentialDecayScheduler)
    
    spatial_lr_scale: float = -1  # auto calculate from camera poses if <= 0
    
    density_lr: float = 0.02
    density_dc_lr: float | None = None
    density_res_lr: float | None = None
    res_to_dc_lr_ratio: float = 0.2
    density_freq_res_max: int = 4
    density_omega: float = 2.0 * np.pi
    density_frequency_up_interval: int = 1_000

    scales_lr: float = 0.005

    rotations_lr: float = 0.001

    log_gradients: bool = False
    grad_log_interval: int = 100

    optimizer: OptimizerConfig = field(default_factory=Adam)

    def get_density_dc_lr(self) -> float:
        return self.density_lr if self.density_dc_lr is None else self.density_dc_lr

    def get_density_res_lr(self) -> float:
        """ If density_res_lr is not set, it will be res_to_dc_lr_ratio times of density_dc_lr """
        if self.density_res_lr is not None:
            return self.density_res_lr
        return self.get_density_dc_lr() * self.res_to_dc_lr_ratio
    
    def get_lr(self, key: str) -> float:
        if key in ("density", "density_dc"):
            return self.get_density_dc_lr()
        if "density_res" in key:
            return self.get_density_res_lr()
        return getattr(self, f"{key}_lr")

@dataclass
class GaussianInits:
    n_gs                 :int
    density_res_freq_max :int
    means                :Tensor = field(init=False)
    density_dc           :Tensor = field(init=False)
    density_res_freq     :Tensor = field(init=False)
    density_res_phase    :Tensor = field(init=False)
    scales               :Tensor = field(init=False)
    rotations            :Tensor = field(init=False)
    
    
    def __post_init__(self):
        self.scales = torch.ones(self.n_gs, 3, dtype=torch.float32) * 0.1
        self.density_dc = torch.ones(self.n_gs, 1, dtype=torch.float32)
        self.density_res_freq = torch.randn(self.n_gs, self.density_res_freq_max, dtype=torch.float32) * 0.01
        self.density_res_phase = torch.randn(self.n_gs, self.density_res_freq_max, dtype=torch.float32) * 0.01

        self.rotations = torch.zeros((self.n_gs, 4), dtype=torch.float32)
        self.rotations[:, 0] = 1

@dataclass
class Xray4DGaussian(Gaussian):
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "Xray4DGaussianModel":
        return Xray4DGaussianModel(self)

def _identity_act(x: Tensor) -> Tensor:
    return x


def inverse_softplus(beta: float = 1.0, epsilon: float = 1e-6):
    """
    stable inverse_softplus for FP16。
    y = log(exp(beta * x) - 1) / beta
    """
    def _forward(x: torch.Tensor) -> torch.Tensor:
        # Precision protection for small x use expm1(x) = exp(x) - 1
        stable_val = torch.log(torch.expm1(beta * x) + epsilon) / beta
        
        # when x is large, use linear approximation
        is_large = (beta * x) > 9.0 
        
        return torch.where(is_large, x, stable_val)

    return _forward

class HasDensityGetter(ABC):
    gaussians: nn.ParameterDict
    
    _density_dc_name: str = "density_dc"
    _density_res_freq_name: str = "density_res_freq"
    _density_res_phase_name: str = "density_res_phase"
    
    _active_F: torch.Tensor
    
    density_omega: float = 2.0 * np.pi
    
    def density_activateion(
        self, 
        x: Tensor,
        scale: float = 1e-3,
        softplus_beta: float = 10.0,
    ):
        return scale * torch.nn.Softplus(threshold=softplus_beta)(x)

    def _format_density_time(self, t: torch.Tensor | float) -> torch.Tensor:
        """Format time input t to shape (n_gaussians, 1) for density calculation"""
        d_f = self.density_res_freq
        n_gaussians = d_f.shape[0]
        t_tensor = torch.as_tensor(t, device=d_f.device, dtype=d_f.dtype)

        if t_tensor.ndim == 0:
            return t_tensor.view(1, 1).expand(n_gaussians, 1)

        if t_tensor.ndim == 1:
            if t_tensor.shape[0] == 1:
                return t_tensor.view(1, 1).expand(n_gaussians, 1)
            if t_tensor.shape[0] == n_gaussians:
                return t_tensor.unsqueeze(-1)
            raise ValueError(f"Expected time shape [1] or [{n_gaussians}], got {tuple(t_tensor.shape)}")

        if t_tensor.ndim == 2 and t_tensor.shape[1] == 1:
            if t_tensor.shape[0] == 1:
                return t_tensor.expand(n_gaussians, 1)
            if t_tensor.shape[0] == n_gaussians:
                return t_tensor
            raise ValueError(f"Expected time shape [1,1] or [{n_gaussians},1], got {tuple(t_tensor.shape)}")

        raise ValueError(f"Unsupported time shape: {tuple(t_tensor.shape)}")
    
    def out_density(
        self,
        dc: torch.Tensor,
        res_freq: torch.Tensor|None = None,
        res_phase: torch.Tensor|None = None,
        *,
        t: float|Tensor|None
    ):
        if res_freq is None and res_phase is None:
            return self.density_activateion(dc)
        
        assert res_freq is not None and res_phase is not None, "res_freq and res_phase must be both provided or both None"
        
        max_res_freq = res_freq.shape[1]
        f_m = min(self._active_F, max_res_freq)
        if f_m == 0:
            return self.density_activateion(dc)
        
        freq = res_freq[:, :f_m]
        phase = res_phase[:, :f_m]
        
        assert t is not None, "t must be provided when using freq_res"
        t_tensor = self._format_density_time(t)
        
        # K = f_m is the number of active frequencies, density_logits = dc + sum_{k=1}^K freq_k * cos(ω * k * t + φ_k)
        # density_logits = dc + sum_{k=1}^K freq_k * cos(omega * k * t + phase_k)
        # density = scale * softplus( \beta * density_logits )
        x = torch.arange(1, freq.shape[1] + 1, device=freq.device, dtype=freq.dtype) # (n_active_freq,), [1, 2, ..., K]
        x = (self.density_omega * x).unsqueeze(0) * t_tensor    # (n_gaussians, n_active_freq), [ωt, 2ωt, ..., 3ωt]
        x = dc + torch.sum(freq * torch.cos(x + phase), dim=-1)[:, None]   # (n_gaussians, 1)
        
        torch.cuda.empty_cache()  # clear cache to avoid OOM due to intermediate tensors in density calculation
        return self.density_activateion(x)

    def get_density(self, t: float|Tensor) -> torch.Tensor:
        """Return activated density at time t"""
        return self.out_density(self.density_dc, self.density_res_freq, self.density_res_phase, t=t)
    
    @property
    def active_F(self) -> int:
        if self.__getattribute__("_active_F") is None:
            return self.density_freq_res_max
        else:
            return int(self._active_F.item())
    
    @property
    def density_freq_res_max(self):
        return self.density_res_freq.shape[1]
    
    def update_active_frequency(self, global_step: int, updated_interval: int):
        if not hasattr(self, "_active_F") or self._active_F is None:
            import warnings
            warnings.warn("active_F buffer is not initialized, initializing to freq_res_max")
            self._active_F = torch.tensor(self.density_freq_res_max, dtype=torch.int16)

        if self._active_F >= self.density_freq_res_max:
            return
        
        if global_step % updated_interval == 0:
            self._active_F += 1

    def get_density_mean(
        self,
    ) -> torch.Tensor:
        """Return mean density estimate from DC component only"""
        return self.out_density(self.density_dc, t=None)
    
    def get_density_res_energy(self) -> torch.Tensor:
        """Return the energy of high frequency components, which can be used for recognize contrast flow"""
        if self.density_res_freq.shape[1] == 0:
            return torch.zeros_like(self.density_dc)
        freq_res = self.density_res_freq[:, :self.active_F]
        return torch.sum(freq_res ** 2, dim=-1, keepdim=True)
    
    @property
    def density_mean(self) -> torch.Tensor:
        return self.get_density_mean()
    
    @property
    def density_dc(self) -> torch.Tensor:
        return self.gaussians[self._density_dc_name]

    @density_dc.setter
    def density_dc(self, v):
        self.gaussians[self._density_dc_name] = v

    @property
    def density_res_freq(self) -> torch.Tensor:
        return self.gaussians[self._density_res_freq_name]

    @density_res_freq.setter
    def density_res_freq(self, v):
        self.gaussians[self._density_res_freq_name] = v
    
    @property
    def density_res_phase(self) -> torch.Tensor:
        return self.gaussians[self._density_res_phase_name]
    
    @density_res_phase.setter
    def density_res_phase(self, v):
        self.gaussians[self._density_res_phase_name] = v


class Xray4DGaussianModel(
    HasVanillaGetters,
    GaussianModel,
    HasMeanGetter,
    HasDensityGetter,
    HasScaleGetter,
    HasRotationGetter,
):
    gaussians: nn.ParameterDict
    
    d_motion_mean: torch.Tensor     # E(motion), shape = (N, 3+3+1)  motion = (d_xyz, d_scale, d_rotation(quat_angle))
    d_motion_2_mean: torch.Tensor
    
    
    def __init__(self, config: Xray4DGaussian) -> None:
        super().__init__()
        self.config = config

        # initialize active_F for density
        self.register_buffer("_active_F", torch.tensor(0, dtype=torch.int16), persistent=True)

        self._keys = (
            self._mean_name,
            self._density_dc_name,
            self._density_res_freq_name,
            self._density_res_phase_name,
            self._scale_name,
            self._rotation_name,
        )

        self.is_pre_activated = False
        
        self.scale_activation = torch.nn.Softplus(threshold=10.0)
        self.scale_inverse_activation = inverse_softplus()
        self.rotation_activation = F.normalize
        self.rotation_inverse_activation = _identity_act
        
        self.ema_lambda = 0.95
        self._grad_hook_registered = False

    @staticmethod
    def _grad_stats(parameters: list[torch.Tensor]) -> dict[str, float]:
        total_sq_norm = 0.0
        max_abs = 0.0
        nan_count = 0.0
        inf_count = 0.0
        grad_param_count = 0.0
        grad_elem_count = 0.0

        for p in parameters:
            g = p.grad
            if g is None:
                continue
            grad_param_count += 1.0
            grad_elem_count += float(g.numel())
            g32 = g.detach().float()
            nan_count += float(torch.isnan(g32).sum().item())
            inf_count += float(torch.isinf(g32).sum().item())
            total_sq_norm += float(torch.square(g32).sum().item())
            max_abs = max(max_abs, float(g32.abs().max().item()))

        return {
            "l2": total_sq_norm ** 0.5,
            "max_abs": max_abs,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "grad_param_count": grad_param_count,
        }

    def _register_grad_monitor_hook(self, module: LightningModule):
        if self._grad_hook_registered:
            return

        def _log_gradients(outputs, batch, gaussian_model, global_step: int, pl_module: LightningModule):
            optimization_config = self.config.optimization
            if optimization_config.log_gradients is False:
                return
            if optimization_config.grad_log_interval <= 0:
                return
            if global_step % optimization_config.grad_log_interval != 0:
                return
            if pl_module.global_rank != 0:
                return
            if pl_module.logger is None:
                return

            metrics_to_log: dict[str, float] = {}
            for key in self._keys:
                stats = self._grad_stats([self.gaussians[key]])
                metrics_to_log[f"stats/gs_grad/{key}/l2"] = stats["l2"]
                metrics_to_log[f"stats/gs_grad/{key}/max_abs"] = stats["max_abs"]
                metrics_to_log[f"stats/gs_grad/{key}/nan_count"] = stats["nan_count"]
                metrics_to_log[f"stats/gs_grad/{key}/inf_count"] = stats["inf_count"]
            metrics_to_log["stats/gs_active_F"] = float(self._active_F)
            metrics_to_log["stats/gs_density_freq_res_mean"] = self.density_res_freq.mean().item()
            
            pl_module.logger.log_metrics(metrics_to_log, step=pl_module.trainer.global_step)

        module.on_after_backward_hooks.append(_log_gradients)
        self._grad_hook_registered = True

    @property
    def n_gaussians(self) -> int:
        return self.gaussians["means"].shape[0]
    
    @override
    @staticmethod
    def setup_gaussians_container():
        return nn.ParameterDict()
    
    def _init_motions(self, n_gaussians: int, device):
        self.motion_ch = 3 + 3 + 1
        self.register_buffer("d_motion_mean", torch.zeros(n_gaussians, self.motion_ch, device=device, requires_grad=False))
        self.register_buffer("d_motion_2_mean", torch.zeros(n_gaussians, self.motion_ch, device=device, requires_grad=False))
    
    def update_motions(
        self, 
        d_xyz: torch.Tensor, 
        d_scale: torch.Tensor, 
        d_rotation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_d_xyz, n_d_scale, n_d_rot = d_xyz.shape[0], d_scale.shape[0], d_rotation.shape[0]
        n_gaussian = self.n_gaussians
        assert n_d_xyz == n_d_scale == n_d_rot == n_gaussian
        
        d_rotation_norm = torch.nn.functional.normalize(d_rotation, dim=-1)
        d_rotation_norm = d_rotation_norm.clamp(-1 + 1e-6, 1 - 1e-6)
        d_angle = 2 * torch.acos(d_rotation_norm[:, 0]).unsqueeze(-1)
        motion = torch.cat((d_xyz, d_scale, d_angle), dim=-1)
        
        motion_2 = torch.square(motion)
        k = self.ema_lambda
        d_motion_mean = k * self.d_motion_mean + (1-k) * motion
        d_motion_2_mean = k * self.d_motion_2_mean + (1-k) * motion_2
        
        self.d_motion_mean = d_motion_mean.detach()
        self.d_motion_2_mean = d_motion_2_mean.detach()

        d_motion_var = d_motion_2_mean - torch.square(d_motion_mean)
        
        return d_motion_mean, d_motion_var
    
    def get_motion_var(self) -> torch.Tensor:
        return self.d_motion_2_mean - torch.square(self.d_motion_mean)
    
    def get_motion_mean(self) -> torch.Tensor:
        return self.d_motion_mean
    
    def clone_motion_by_mask(self, mask: torch.Tensor, repeats: int):
        new_d_motion_mean = self.d_motion_mean[mask].repeat(repeats, 1)
        new_d_motion_2_mean = self.d_motion_2_mean[mask].repeat(repeats, 1)
        
        self.d_motion_mean = torch.cat(
            (self.d_motion_mean, new_d_motion_mean), 
            dim=0
        ).detach()
        self.d_motion_2_mean = torch.cat(
            (self.d_motion_2_mean, new_d_motion_2_mean),
            dim = 0
        ).detach()
        
        assert self.n_gaussians == self.d_motion_mean.shape[0] == self.d_motion_2_mean.shape[0]
    
    
    def filter_motion_by_mask(self, valid_mask: torch.Tensor):
        self.d_motion_mean = self.d_motion_mean[valid_mask].detach()
        self.d_motion_2_mean = self.d_motion_2_mean[valid_mask].detach()
        
        assert self.n_gaussians == self.d_motion_mean.shape[0] == self.d_motion_2_mean.shape[0]
        

    # --- Part2: Set up gaussians' parameters, optimizers and schedulers.
    @override
    def setup_from_pcd(
            self, xyz: Tensor, 
            rgb: Any, 
            *args, 
            **kwargs 
        ):
        """ 
        Init gaussian from point cloud of coronary central line
        """
        if isinstance(xyz, np.ndarray):
            xyz_coronary = torch.tensor(xyz, dtype=torch.float)
        else:
            xyz_coronary = xyz.float()
        
        fused_point_cloud = xyz_coronary

        from simple_knn._C import distCUDA2
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.cuda()), 0.0000001).to(fused_point_cloud.device)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        n_gaussians = fused_point_cloud.shape[0]
        inits = GaussianInits(n_gaussians, density_res_freq_max=self.config.optimization.density_freq_res_max)
        
        self.set_properties({
            self._mean_name:                fused_point_cloud,
            self._density_dc_name:          inits.density_dc,
            self._density_res_freq_name:    inits.density_res_freq,
            self._density_res_phase_name:   inits.density_res_phase,
            self._scale_name:               scales,
            self._rotation_name:            inits.rotations,
        })
        self._init_motions(n_gaussians, device=torch.device("cuda"))
    
    @override
    def setup_from_number(self, n: int, *args, **kwargs):
        inits = GaussianInits(n, density_res_freq_max=self.config.optimization.density_freq_res_max)

        self.set_properties({
            self._mean_name:                inits.means,
            self._density_dc_name:          inits.density_dc,
            self._density_res_freq_name:    inits.density_res_freq,
            self._density_res_phase_name:   inits.density_res_phase,
            self._scale_name:               inits.scales,
            self._rotation_name:            inits.rotations,
        })
        self._init_motions(n, device=torch.device("cuda"))
        
    @override
    def set_properties(self, properties: Mapping[str, Any]):
        """
        Set all raw properties.
        This setter will not update optimizers.
        """

        for name in properties:
            if name == "density":
                self.density_freq = properties[name]
                continue
            self.gaussians[name] = properties[name]

    @override
    def setup_from_tensors(self, tensors: dict[str, torch.Tensor], active_sh_degree: int = -1, *args, **kwargs):
        raise NotImplementedError()
    
    @override
    def training_setup(self, module: LightningModule) -> tuple[
        list[torch.optim.Optimizer] | torch.optim.Optimizer | None,
        list[torch.optim.lr_scheduler.LRScheduler] | torch.optim.lr_scheduler.LRScheduler | None
    ]:
        self._register_grad_monitor_hook(module)

        # ---- adaptively scaled according to the camera distribution radius
        spatial_lr_scale = self.config.optimization.spatial_lr_scale
        if spatial_lr_scale <= 0:
            spatial_lr_scale = module.trainer.datamodule.dataparser_outputs.camera_extent   # type: ignore
        assert spatial_lr_scale > 0
        optimization_config = self.config.optimization
        
        if optimization_config.means_lr_scheduler.lr_final is not None:
            optimization_config.means_lr_scheduler.lr_final *= spatial_lr_scale
        
        # --- init optimizer and scheduler for means
        def _add_optimizer_after_backward_hook_if_available(optimizer, pl_module):
            hook = getattr(optimizer, "on_after_backward", None)
            if hook is None:
                return
            pl_module.on_after_backward_hooks.append(hook)
            
        optimizer_factory = self.config.optimization.optimizer
        
        opt_list: list[torch.optim.Optimizer] = []
        schedule_list: list[torch.optim.lr_scheduler.LRScheduler] = []
        
        name = "means"
        lr = optimization_config.means_lr_init
        means_optimizer = optimizer_factory.instantiate(
            [
                {
                    'params': [self.gaussians[name]], 
                    "name": name
                }
            ],
            lr=lr,
            eps=1e-15,
        )
        _add_optimizer_after_backward_hook_if_available(means_optimizer, module)
        
        means_scheduler = optimization_config.means_lr_scheduler.instantiate().get_scheduler(
            means_optimizer, lr,
        )
        
        opt_list.append(means_optimizer)
        schedule_list.append(means_scheduler)
        
        # --- init optimizer and scheduler for other parameters
        params = []
        for key in self._keys:
            if key == self._mean_name:
                continue
            if key == self._density_dc_name:
                params.append({
                    "params": [self.gaussians[key]],
                    "lr": optimization_config.get_density_dc_lr(),
                    "name": key,
                })
                continue
            if key in (self._density_res_freq_name, self._density_res_phase_name):
                params.append({
                    "params": [self.gaussians[key]],
                    "lr": optimization_config.get_density_res_lr(),
                    "name": key,
                })
                continue
            params.append({
                "params": [self.gaussians[key]],
                "lr": optimization_config.get_lr(key),
                "name": key,
            })
        constant_lr_optimizer = optimizer_factory.instantiate(params, lr=0.0, eps=1e-15)
        _add_optimizer_after_backward_hook_if_available(constant_lr_optimizer, module)
        opt_list.append(constant_lr_optimizer)
        
        print(f"spatial_lr_scale={spatial_lr_scale}, learning_rates=")
        print(f"means: {optimization_config.means_lr_init} -> {optimization_config.means_lr_scheduler.lr_final}")
        print(
            "density: dc={} res={} (res/dc={})".format(
                optimization_config.get_density_dc_lr(),
                optimization_config.get_density_res_lr(),
                optimization_config.get_density_res_lr() / optimization_config.get_density_dc_lr(),
            )
        )
        for p in params:
            print(f"  {p['name']}: {p['lr']}")
        
        return opt_list, schedule_list

    # --- Part3: Implement other abstract methods for .gaussian.GaussianModel

    @override
    def get_property_names(self) -> tuple[str, ...]:
        return self._keys

    @override
    def on_train_batch_end(self, step: int, module: LightningModule):
        optimization_config = self.config.optimization
        self.update_active_frequency(step, optimization_config.density_frequency_up_interval)

    # --- Part4: Implement activaties for HasNewGetters
    
    ## covariance
    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @override
    def get_covariance(self, scaling_modifier: float = 1.):
        return self.build_covariance_from_scaling_rotation(
            self.get_scales(),
            scaling_modifier,
            self.get_rotations(),
        )
    
    # --- Part5: Implement abstract methods for HasVanillaGetters
    
    @property
    @override
    def get_scaling(self):
        return self.get_scales()

    @property
    @override
    def get_rotation(self):
        return self.get_rotations()

    @property
    @override
    def get_xyz(self):
        return self.get_means()

    @property
    @override
    def get_features(self):
        raise NotImplementedError()
        # return torch.exp(-self.get_density())

    @property
    @override
    def get_opacity(self):
        raise NotImplementedError()
        # return torch.exp(-self.get_density())
    
    # --- Part6: pre-activate all parameters before inference
    
    def pre_activate_all_properties(self, t: torch.Tensor | float | int = 0.0):
        raise NotImplementedError("pre_activate_all_properties is not implemented yet")
    
    def get_non_pre_activated_properties(self):
        raise NotImplementedError("get_non_pre_activated_properties is not implemented yet")