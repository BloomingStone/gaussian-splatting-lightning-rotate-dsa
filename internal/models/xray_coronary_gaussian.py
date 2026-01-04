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
    HasOpacityGetter,
)
from internal.utils.general_utils import (
    inverse_sigmoid,
    strip_symmetric,
    build_scaling_rotation,
)
from internal.schedulers import ExponentialDecayScheduler
from internal.optimizers import OptimizerConfig, Adam

@dataclass
class XrayExponentialDecayScheduler(ExponentialDecayScheduler):
    lr_final = 0.0000016
    max_steps = 30_000


@dataclass
class OptimizationConfig:
    means_lr_init: float = 1e-5
    
    means_lr_scheduler: ExponentialDecayScheduler = field(default_factory=XrayExponentialDecayScheduler)
    
    spatial_lr_scale: float = -1  # auto calculate from camera poses if <= 0
    
    gray_lr: float = 0.0025

    opacities_lr: float = 0.05

    scales_lr: float = 0.005

    rotations_lr: float = 0.001

    optimizer: OptimizerConfig = field(default_factory=Adam)
    
    def get_lr(self, key: str) -> float:
        return getattr(self, f"{key}_lr")

@dataclass
class GaussianIniter:
    n_gs       :int
    means      :Float32[Tensor, "n_gs 3"]           = field(init=False)
    gray       :Float32[Tensor, "n_gs 1"]           = field(init=False)
    scales     :Float32[Tensor, "n_gs 3"]           = field(init=False)
    opacities  :Float32[Tensor, "n_gs 1"]           = field(init=False)
    rotations  :Float32[Tensor, "n_gs 4"]           = field(init=False)
    
    def __post_init__(self):
        self.scales = torch.zeros(self.n_gs, 3, dtype=torch.float32)
        self.means = torch.zeros(self.n_gs, 3, dtype=torch.float32)
        self.gray = torch.ones(self.n_gs, 1, dtype=torch.float32)*0.5
        
        self.opacities = inverse_sigmoid(0.01 * torch.ones((self.n_gs, 1), dtype=torch.float32))
        
        self.rotations = torch.zeros((self.n_gs, 4), dtype=torch.float32)
        self.rotations[:, 0] = 1

@dataclass
class XrayCoronaryGaussian(Gaussian):
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "XrayCoronaryGaussianModel":
        return XrayCoronaryGaussianModel(self)

def _identity_act(x: Tensor) -> Tensor:
    return x

class HasGrayGetter(ABC):
    gaussians: nn.ParameterDict
    _gray_name: str = "gray"
    gray_activation: Callable[[torch.Tensor], torch.Tensor]
    gray_inverse_activation: Callable[[torch.Tensor], torch.Tensor]

    def get_gray(self) -> torch.Tensor:
        """Return activated gray"""
        return self.gray_activation(self.gray)

    @property
    def gray(self) -> torch.Tensor:
        """Return raw gray"""
        return self.gaussians[self._gray_name]

    @gray.setter
    def gray(self, v):
        """Set raw gray"""
        self.gaussians[self._gray_name] = v


class XrayCoronaryGaussianModel(
    HasVanillaGetters,
    GaussianModel,
    HasMeanGetter,
    HasGrayGetter,
    HasScaleGetter,
    HasRotationGetter,
    HasOpacityGetter,
):
    gaussians: nn.ParameterDict
    
    d_motion_mean: torch.Tensor     # E(motion), shape = (N, 3+3+1)  motion = (d_xyz, d_scale, d_rotation(quat_angle))
    d_motion_2_mean: torch.Tensor
    
    
    def __init__(self, config: XrayCoronaryGaussian) -> None:
        super().__init__()
        self.config = config

        self._keys = (
            "means", "gray", "opacities", "scales", "rotations"
        )

        self.is_pre_activated = False
        
        self.scale_activation = torch.exp
        self.scale_inverse_activation = torch.log
        self.gray_activation = torch.sigmoid
        self.gray_inverse_activation = inverse_sigmoid
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = inverse_sigmoid
        self.rotation_activation = F.normalize
        self.rotation_inverse_activation = _identity_act
        
        self.ema_lambda = 0.95

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
        d_rotation_norm.clamp_(-1 + 1e-6, 1 - 1e-6)
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
        )
        self.d_motion_2_mean = torch.cat(
            (self.d_motion_2_mean, new_d_motion_2_mean),
            dim = 0
        )
        
        assert self.n_gaussians == self.d_motion_mean.shape[0] == self.d_motion_2_mean.shape[0]
    
    
    def filter_motion_by_mask(self, valid_mask: torch.Tensor):
        self.d_motion_mean = self.d_motion_mean[valid_mask]
        self.d_motion_2_mean = self.d_motion_2_mean[valid_mask]
        
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
        

        xyz_background = self._get_backgound_gaussian_from_xyz(xyz_coronary)
        fused_point_cloud = torch.cat([xyz_coronary, xyz_background]).float()

        from simple_knn._C import distCUDA2
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.cuda()), 0.0000001).to(fused_point_cloud.device)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        n_gaussians = fused_point_cloud.shape[0]
        inits = GaussianIniter(n_gaussians)
        
        self.set_properties({
            "means":     fused_point_cloud,
            "gray":      inits.gray,
            "opacities": inits.opacities,
            "scales":    scales,
            "rotations": inits.rotations,
        })
        self._init_motions(n_gaussians, device=torch.device("cuda"))
    
    @override
    def setup_from_number(self, n: int, *args, **kwargs):
        inits = GaussianIniter(n)

        self.set_properties({
            "means":     inits.means,
            "gray":      inits.gray,
            "opacities": inits.opacities,
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
        return self.get_gray()

    @property
    @override
    def get_opacity(self):
        return self.get_opacities()
    
    # --- Part6: pre-activate all parameters before inference
    
    def pre_activate_all_properties(self):
        self.is_pre_activated = True
        
        # replace parameters with pre-activated versions
        self.scales = self.get_scales()
        self.rotations = self.get_rotations()
        self.opacities = self.get_opacities()
        self.gray = self.get_gray()

        self.scale_activation = _identity_act
        self.scale_inverse_activation = _identity_act
        self.rotation_activation = _identity_act
        self.rotation_inverse_activation = _identity_act
        self.opacity_activation = _identity_act
        self.opacity_inverse_activation = _identity_act
        self.gray_activation = _identity_act
        self.gray_inverse_activation = _identity_act
    
    def get_non_pre_activated_properties(self):
        if self.is_pre_activated is True:
            activated_properties = self.properties
            keys = list(activated_properties.keys())
            non_pre_activated_properties = {}
            for suffix in ["_coronary", "_background"]:
                scales = "scales" + suffix
                opacities = "opacities" + suffix
                gray = "gray" + suffix
                non_pre_activated_properties[scales] = torch.log(activated_properties[scales])
                keys.remove(scales)
                non_pre_activated_properties[opacities] = inverse_sigmoid(activated_properties[opacities])
                keys.remove(opacities)
                non_pre_activated_properties[gray] = torch.log(activated_properties[gray])
                keys.remove(gray)

            for key in keys:
                non_pre_activated_properties[key] = activated_properties[key]

            return non_pre_activated_properties
        else:
            return self.properties 