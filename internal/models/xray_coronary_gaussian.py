from dataclasses import dataclass, field
from typing import override, Mapping, Callable, Any
from abc import ABC

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from lightning import LightningModule
from jaxtyping import Float32

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

    scales_lr: float = 0.005

    rotations_lr: float = 0.001

    optimizer: OptimizerConfig = field(default_factory=Adam)
    
    def get_lr(self, key: str) -> float:
        return getattr(self, f"{key}_lr")

@dataclass
class GaussianInits:
    n_gs       :int
    means      :Float32[Tensor, "n_gs 3"]           = field(init=False)
    density       :Float32[Tensor, "n_gs 1"]        = field(init=False)
    scales     :Float32[Tensor, "n_gs 3"]           = field(init=False)
    rotations  :Float32[Tensor, "n_gs 4"]           = field(init=False)
    
    def __post_init__(self):
        self.scales = torch.zeros(self.n_gs, 3, dtype=torch.float32)
        self.means = torch.zeros(self.n_gs, 3, dtype=torch.float32)
        self.density = torch.ones(self.n_gs, 1, dtype=torch.float32)
        
        self.rotations = torch.zeros((self.n_gs, 4), dtype=torch.float32)
        self.rotations[:, 0] = 1

@dataclass
class XrayCoronaryGaussian(Gaussian):
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "XrayCoronaryGaussianModel":
        return XrayCoronaryGaussianModel(self)

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
    _density_name: str = "density"
    density_activation: Callable[[torch.Tensor], torch.Tensor]
    density_inverse_activation: Callable[[torch.Tensor], torch.Tensor]
    def get_density(self) -> torch.Tensor:
        """Return activated density"""
        return self.density_activation(self.density)

    @property
    def density(self) -> torch.Tensor:
        """Return raw density"""
        return self.gaussians[self._density_name]

    @density.setter
    def density(self, v):
        """Set raw density"""
        self.gaussians[self._density_name] = v


class XrayCoronaryGaussianModel(
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
    
    
    def __init__(self, config: XrayCoronaryGaussian) -> None:
        super().__init__()
        self.config = config

        self._keys = (
            "means", "density", "scales", "rotations"
        )

        self.is_pre_activated = False
        
        self.scale_activation = torch.nn.Softplus(threshold=10.0)
        self.scale_inverse_activation = inverse_softplus()
        self.density_activation = lambda x: 1e-3 * torch.nn.Softplus(threshold=10.0)(x)    # scale by 1e-3
        self.density_inverse_activation = lambda x: inverse_softplus()(x * 1e3)     # reverse scale by 1e3
        self.rotation_activation = F.normalize
        self.rotation_inverse_activation = _identity_act
        
        self.ema_lambda = 0.95

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
            self, xyz: Float32[Tensor|np.ndarray, "n_gaussians 3"], 
            rgb: Any, 
            *args, 
            **kwargs 
        ):
        """ 
        Init gaussian from point cloud of coronary central line
        """
        if isinstance(xyz, np.ndarray):
            xyz_coronary = torch.tensor(xyz, dtype=torch.float)
        
        fused_point_cloud = xyz_coronary

        from simple_knn._C import distCUDA2
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.cuda()), 0.0000001).to(fused_point_cloud.device)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        n_gaussians = fused_point_cloud.shape[0]
        inits = GaussianInits(n_gaussians)
        
        self.set_properties({
            "means":     fused_point_cloud,
            "density":   inits.density,
            "scales":    scales,
            "rotations": inits.rotations,
        })
        self._init_motions(n_gaussians, device=torch.device("cuda"))
    
    @override
    def setup_from_number(self, n: int, *args, **kwargs):
        inits = GaussianInits(n)

        self.set_properties({
            "means":     inits.means,
            "density":   inits.density,
            "scales":    inits.scales,
            "rotations": inits.rotations,
        })
        self._init_motions(n, device=torch.device("cuda"))
        
    @override
    def set_properties(self, properties: Mapping[str, Any]):
        """
        Set all raw properties.
        This setter will not update optimizers.
        """

        for name in properties:
            self.gaussians[name] = properties[name]

    @staticmethod
    def _get_backgound_gaussian_from_xyz(xyz: Float32[Tensor, "n_gaussians 3"]) -> Float32[Tensor, "n_gaussians 3"]:
        """
        sample points from sphere around coronary's xyz.
        """
        center = xyz.mean(dim=0, keepdim=True)
        d = torch.norm(xyz - center, dim=1)
        radius = d.max() * 1.2
        N = xyz.shape[0]
        device = xyz.device
        dirs = torch.randn(N, 3, device=device)
        dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)

        # p ∝ V ∝ r**3 therefor r ∝ p**{1/3}
        p = torch.rand(N, 1, device=device)
        r = radius * (p ** (1/3))
        
        return center + dirs * r

    @override
    def setup_from_tensors(self, tensors: dict[str, torch.Tensor], active_sh_degree: int = -1, *args, **kwargs):
        raise NotImplementedError()
    
    @override
    def training_setup(self, module: LightningModule) -> tuple[
        list[torch.optim.Optimizer] | torch.optim.Optimizer | None,
        list[torch.optim.lr_scheduler.LRScheduler] | torch.optim.lr_scheduler.LRScheduler | None
    ]:
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
            if key == "means":
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
        for p in params:
            print(f"  {p['name']}: {p['lr']}")
        
        return opt_list, schedule_list

    # --- Part3: Implement other abstract methods for .gaussian.GaussianModel

    @override
    def get_property_names(self) -> tuple[str, ...]:
        return self._keys

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
    
    def pre_activate_all_properties(self):
        self.is_pre_activated = True
        
        # replace parameters with pre-activated versions
        self.scales = self.get_scales()
        self.rotations = self.get_rotations()
        self.density = self.get_density()

        self.scale_activation = _identity_act
        self.scale_inverse_activation = _identity_act
        self.rotation_activation = _identity_act
        self.rotation_inverse_activation = _identity_act
        self.density_activation = _identity_act
        self.density_inverse_activation = _identity_act
    
    def get_non_pre_activated_properties(self):
        if self.is_pre_activated is True:
            activated_properties = self.properties
            keys = list(activated_properties.keys())
            non_pre_activated_properties = {}
            scales = "scales"
            density = "density"
            non_pre_activated_properties[scales] = self.scale_inverse_activation(activated_properties[scales])
            keys.remove(scales)
            non_pre_activated_properties[density] = self.density_inverse_activation(activated_properties[density])
            keys.remove(density)

            for key in keys:
                non_pre_activated_properties[key] = activated_properties[key]

            return non_pre_activated_properties
        else:
            return self.properties 