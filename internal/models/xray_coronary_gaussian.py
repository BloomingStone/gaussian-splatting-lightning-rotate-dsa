from dataclasses import dataclass, field
from typing import override, Iterator, Iterable, Mapping, Union, Callable, Any
import warnings
from abc import ABC

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from lightning import LightningModule
from enum import StrEnum, auto
from jaxtyping import UInt8, Float32

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
    inverse_softplus
)
from internal.schedulers import ExponentialDecayScheduler
from internal.optimizers import OptimizerConfig, Adam

class XrayGassianState(StrEnum):
    CORONARY = auto()
    BACKGROUND = auto()
    WHOLE = auto()

def _split_key(key: str) -> tuple[str, XrayGassianState|None]:
    state2suffix = {
        XrayGassianState.CORONARY: f"_{XrayGassianState.CORONARY.name.lower()}",
        XrayGassianState.BACKGROUND: f"_{XrayGassianState.BACKGROUND.name.lower()}",
        XrayGassianState.WHOLE: f"_{XrayGassianState.WHOLE.name.lower()}"
    }
    for state, suffix in state2suffix.items():
        if key.lower().endswith(suffix):
            return key[:-len(suffix)], state
    return key, None

@dataclass
class XrayExponentialDecayScheduler(ExponentialDecayScheduler):
    lr_final = 0.0000016
    max_steps = 30_000


@dataclass
class OptimizationConfig:
    means_lr_init_coronary: float = 1e-5
    means_lr_init_background: float = 0.00016
    
    means_lr_scheduler: ExponentialDecayScheduler = field(default_factory=XrayExponentialDecayScheduler)
    
    spatial_lr_scale: float = -1  # auto calculate from camera poses if <= 0

    opacities_lr: float = 0.05

    scales_lr: float = 0.005

    rotations_lr: float = 0.001

    optimizer: OptimizerConfig = field(default_factory=Adam)

    def get_lr(self, key: str) -> float:
        key, _ = _split_key(key)
        return getattr(self, f"{key}_lr")


inputT = torch.Tensor | nn.Parameter
class XrayGaussianParameterDict(nn.ParameterDict):
    def __init__(
        self,
        *args,
        state: XrayGassianState = XrayGassianState.WHOLE,
        n_coronary_gs: int|None = None,
        n_background_gs: int|None = None,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._state = state
        
        self.coronary_gs = nn.ParameterDict()
        self.background_gs = nn.ParameterDict()
        self._n_coronary_gs: int|None = n_coronary_gs
        self._n_background_gs: int|None = n_background_gs
    
    @property
    def state(self) -> XrayGassianState:
        return self._state
    
    @state.setter
    def state(self, state: XrayGassianState|str):
        self._state = XrayGassianState(state.lower())
    
    @property
    def n_coronary_gs(self) -> int|None:
        if len(self.coronary_gs) == 0:
            self._n_background_gs = None
        else:
            key_0 = next(iter(self.coronary_gs))
            self._n_coronary_gs = self.coronary_gs[key_0].shape[0]
        return self._n_coronary_gs
    
    @property
    def n_background_gs(self) -> int|None:
        if len(self.background_gs) == 0:
            self._n_background_gs = None
        else:
            key_0 = next(iter(self.background_gs))
            self._n_background_gs = self.background_gs[key_0].shape[0]
        return self._n_background_gs
    
    def init_n_gaussians(self, n_coronary_gs: int, n_background_gs: int):
        assert self._n_coronary_gs is None and self._n_background_gs is None, "n_coronary_gs and n_background_gs are already set"
        self._n_coronary_gs = n_coronary_gs
        self._n_background_gs = n_background_gs
    
    def _get_key_and_state(self, key: str) -> tuple[str, XrayGassianState]:
        key, state = _split_key(key)
        state = state if state is not None else self.state
        return key, state
    
    @staticmethod
    def _to_param(x: inputT) -> nn.Parameter:
        return nn.Parameter(x.requires_grad_(True)) if not isinstance(x, nn.Parameter) else x
    
    @override
    def __getitem__(self, key: str) -> nn.Parameter:
        key, state = self._get_key_and_state(key)
        if state == XrayGassianState.WHOLE:
            return self._to_param(torch.concat([self.coronary_gs[key], self.background_gs[key]]))
        else:
            return self.coronary_gs[key] if state == XrayGassianState.CORONARY else self.background_gs[key]
    
    @override
    def __setitem__(self, key: str, value: inputT | tuple[inputT, inputT]) -> None:
        """Set parameter values for coronary/background gaussians based on key and value type.
        
        Args:
            key (str): Parameter name, optionally suffixed with state (e.g. 'means_coronary')
            value (inputT | tuple[inputT, inputT]): 
                - If tuple: (coronary_value, background_value) for the first time init, 
                - If tensor: 
                    - when state is WHOLE, value is a tensor of shape (n_coronary_gs + n_background_gs,)
                    - when state is CORONARY or BACKGROUND, value is a tensor of shape (n_coronary_gs,) or (n_background_gs,)

        Raises:
            Exception: 
                - If value type doesn't match expected format
                - If trying to set single gaussian without whole initialization
        """
        key, state = self._get_key_and_state(key)
        if state == XrayGassianState.WHOLE:
            if isinstance(value, tuple) and len(value) == 2:
                assert isinstance(value[0], inputT) and isinstance(value[1], inputT)
                self.coronary_gs[key] = self._to_param(value[0])
                self.background_gs[key] = self._to_param(value[1])
                if self._n_coronary_gs is None:
                    self._n_coronary_gs = value[0].shape[0]
                if self._n_background_gs is None:
                    self._n_background_gs = value[1].shape[0]
            else:
                assert self._n_coronary_gs is not None and self._n_background_gs is not None, "Unknown n_coronary_gs and n_background_gs, use tuple input to init or call `init_n_gaussians` first"
                assert len(value) == self._n_coronary_gs + self._n_background_gs, "value length doesn't match n_coronary_gs and n_background_gs"
                self.coronary_gs[key] = self._to_param(value[:self._n_coronary_gs])
                self.background_gs[key] = self._to_param(value[self._n_coronary_gs:])
        else:
            assert key in self.coronary_gs and key in self.background_gs, "can not set single new gaussian, set whole gaussian parameter dict first or use keys like 'coronary_whole"
            assert isinstance(value, inputT)
            if state == XrayGassianState.CORONARY:
                self.coronary_gs[key] = self._to_param(value)
                self._n_coronary_gs = value.shape[0]
            else:
                self.background_gs[key] = self._to_param(value)
                self._n_background_gs = value.shape[0]
        
    @override
    def __delitem__(self, key: str) -> None:
        key, state = self._get_key_and_state(key)
        if state == XrayGassianState.WHOLE:
            del self.coronary_gs[key]
            del self.background_gs[key]
            if len(self.coronary_gs) == 0:
                self._n_coronary_gs = None
            if len(self.background_gs) == 0:
                self._n_background_gs = None
        else:
            raise Exception("can not delete single gaussian")
    
    @override
    def __len__(self) -> int:
        # assert len(self.coronary_gs) == len(self.background_gs)
        return len(self.coronary_gs)
    
    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self.coronary_gs)
    
    @override
    def __reversed__(self) -> Iterator[str]:
        return reversed(self.coronary_gs)

    @override
    def copy(self) -> "XrayGaussianParameterDict":
        res = XrayGaussianParameterDict()   # in WHOLE state
        res._n_coronary_gs = self._n_coronary_gs
        res._n_background_gs = self._n_background_gs
        
        old_state = self.state
        self.state = XrayGassianState.WHOLE
        
        for key in self:
            res[key] = self[key]
        
        res.state = old_state
        self.state = old_state
        return res
    
    @override
    def __contains__(self, key: str) -> bool:
        return key in self.coronary_gs
    
    @override
    def clear(self) -> None:
        self.state = XrayGassianState.WHOLE
        self.coronary_gs.clear()
        self.background_gs.clear()
        self._n_coronary_gs = None
        self._n_background_gs = None
    
    @override
    def popitem(self) -> tuple[str, nn.Parameter]:
        assert self.state == XrayGassianState.WHOLE, "can not popitem from single gaussian, popitem from whole gaussian parameter dict first or use keys like 'coronary_whole"
        k1, v1 = self.coronary_gs.popitem()
        k2, v2 = self.background_gs.popitem()
        if len(self.coronary_gs) == 0:
            self._n_coronary_gs = None
        if len(self.background_gs) == 0:
            self._n_background_gs = None
        assert k1 == k2
        return k1, self._to_param(torch.concat([v1, v2]))
    
    @override
    def keys(self) -> Iterable[str]:
        return self.coronary_gs.keys()
    
    @override
    def items(self) -> Iterable[tuple[str, nn.Parameter]]:
        for (k1, v1), (k2, v2) in zip(self.coronary_gs.items(), self.background_gs.items()):
            assert k1 == k2
            match self.state:
                case XrayGassianState.CORONARY:
                    yield k1, v1
                case XrayGassianState.BACKGROUND:
                    yield k1, v2
                case XrayGassianState.WHOLE:
                    yield k1, self._to_param(torch.concat([v1, v2]))
    
    @override
    def values(self) -> Iterable[nn.Parameter]:
        for v1, v2 in zip(self.coronary_gs.values(), self.background_gs.values()):
            match self.state:
                case XrayGassianState.CORONARY:
                    yield v1
                case XrayGassianState.BACKGROUND:
                    yield v2
                case XrayGassianState.WHOLE:
                    yield self._to_param(torch.concat([v1, v2]))
    
    @override
    def update(
        self, 
        parameters: Union[
            "XrayGaussianParameterDict",
            nn.ParameterDict,
            Mapping[str, tuple[inputT, inputT] | inputT]
        ]
    ) -> None:
        if self.state == XrayGassianState.WHOLE:
            if isinstance(parameters, XrayGaussianParameterDict):
                assert self._n_coronary_gs == parameters._n_coronary_gs and self._n_background_gs == parameters._n_background_gs, "n_coronary_gs and n_background_gs are not the same"
                self.coronary_gs.update(parameters.coronary_gs)
                self.background_gs.update(parameters.background_gs)
            else:
                for k, v in parameters.items():
                    self[k] = v
        else:
            warnings.warn(
                f"Only updating XrayGaussianParameterDict's single state-{self.state} gaussian. "
                f"Be sure this is the intended behavior.",
                UserWarning,
                stacklevel=2
            )
            if isinstance(parameters, XrayGaussianParameterDict):
                parameters.state = self.state
            for k, v in parameters.items():
                assert k in self, f"key {k} not in my_gs, can not add new sigular gaussian, set state to WHOLE first"
                self[k] = v

    @override
    def extra_repr(self) -> str:
        return f"state = {self.state}\n {super().extra_repr()}"


@dataclass
class GaussianIniter:
    n_gs       :int
    means      :Float32[Tensor, "n_gs 3"]           = field(init=False)
    scales     :Float32[Tensor, "n_gs 3"]           = field(init=False)
    opacities  :Float32[Tensor, "n_gs 1"]           = field(init=False)
    rotations  :Float32[Tensor, "n_gs 4"]           = field(init=False)
    
    def __post_init__(self):
        self.scales = torch.zeros(self.n_gs, 3, dtype=torch.float32)
        self.means = torch.zeros(self.n_gs, 3, dtype=torch.float32)
        
        self.opacities = torch.ones((self.n_gs, 1), dtype=torch.float32)
        
        self.rotations = torch.zeros((self.n_gs, 4), dtype=torch.float32)
        self.rotations[:, 0] = 1

@dataclass
class XrayCoronaryGaussian(Gaussian):
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "XrayCoronaryGaussianModel":
        return XrayCoronaryGaussianModel(self)

def _identity_act(x: Tensor) -> Tensor:
    return x

class XrayCoronaryGaussianModel(
    HasVanillaGetters,
    GaussianModel,
    HasMeanGetter,
    HasScaleGetter,
    HasRotationGetter,
    HasOpacityGetter,
):
    
    gaussians: XrayGaussianParameterDict
    def __init__(self, config: XrayCoronaryGaussian) -> None:
        super().__init__()
        self.config = config

        self._keys = [
            "means", "opacities", "scales", "rotations"
        ]

        self.is_pre_activated = False
        
        self.state = XrayGassianState.WHOLE
        assert isinstance(self.gaussians, XrayGaussianParameterDict)
        self.gaussians.state = self.state
        
        self.scale_activation = torch.exp
        self.scale_inverse_activation = torch.log
        self.opacity_activation = torch.nn.Softplus(beta=10)
        self.opacity_inverse_activation = inverse_softplus
        self.rotation_activation = F.normalize
        self.rotation_inverse_activation = _identity_act

    # --- Part1: Use `XrayGaussianState` and `XrayGaussianParameterDict` to control gaussian module's output point cloud
    
    @property
    def state(self) -> XrayGassianState:
        return self._state
    
    @state.setter
    def state(self, state: XrayGassianState|str):
        self._state = XrayGassianState(state)
        
        assert isinstance(self.gaussians, XrayGaussianParameterDict)
        self.gaussians.state = self.state
    
    @property
    def all_keys(self) -> list[str]:
        return self._keys
    
    @override
    @staticmethod
    def setup_gaussians_container():
        return XrayGaussianParameterDict()
    
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
        
        n_gaussians = xyz_coronary.shape[0]

        xyz_background = self._get_backgound_gaussian_from_xyz(xyz_coronary)
        fused_point_cloud = torch.cat([xyz_coronary, xyz_background]).float()

        from simple_knn._C import distCUDA2
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.cuda()), 0.0000001).to(fused_point_cloud.device)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales_coronary = scales[:n_gaussians]
        scales_background = scales[n_gaussians:]
        
        inits = GaussianIniter(n_gaussians)
        
        # uses XrayGaussianParameterDict's __setitem__ to set parameters
        self.state = XrayGassianState.WHOLE
        self.set_properties({
            "means":     (  xyz_coronary       ,   xyz_background      ),
            "opacities": (  inits.opacities    ,   inits.opacities.clone()     ),
            "scales":    (  scales_coronary    ,   scales_background.clone()   ),
            "rotations": (  inits.rotations    ,   inits.rotations.clone()       ),
        })
    
    @override
    def setup_from_number(self, n: int, *args, **kwargs):
        inits = GaussianIniter(n)
        
        # uses XrayGaussianParameterDict's __setitem__ to set parameters
        self.state = XrayGassianState.WHOLE
        self.set_properties({
            "means":     (  inits.means        ,   inits.means.clone()      ),
            "opacities": (  inits.opacities    ,   inits.opacities.clone()  ),
            "scales":    (  inits.scales       ,   inits.scales.clone()     ),
            "rotations": (  inits.rotations    ,   inits.rotations.clone()  ),
        })
        
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
        
        means_lr_init = {
            XrayGassianState.CORONARY: optimization_config.means_lr_init_coronary * spatial_lr_scale,
            XrayGassianState.BACKGROUND: optimization_config.means_lr_init_background * spatial_lr_scale
        }
        
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
        
        for state in [XrayGassianState.CORONARY, XrayGassianState.BACKGROUND]:
            name = f"means_{state.lower()}"
            lr = means_lr_init[state]
            means_optimizer = optimizer_factory.instantiate(
                [
                    {
                        'params': [self.gaussians[name]], 
                        "name": name,
                        "state": state.lower()
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
        for state in [XrayGassianState.CORONARY, XrayGassianState.BACKGROUND]:
            params = []
            for key in self._keys:
                if key == "means":
                    continue
                name = f"{key}_{state.lower()}"
                params.append({
                    "params": [self.gaussians[name]],
                    "lr": optimization_config.get_lr(name),
                    "name": f"{key}_{state}",
                    "state": state
                })
            constant_lr_optimizer = optimizer_factory.instantiate(params, lr=0.0, eps=1e-15)
            _add_optimizer_after_backward_hook_if_available(constant_lr_optimizer, module)
            opt_list.append(constant_lr_optimizer)
        
        print(f"spatial_lr_scale={spatial_lr_scale}, learning_rates=")
        print(f"means: {means_lr_init} -> {optimization_config.means_lr_scheduler.lr_final}")
        for p in params:
            print(f"  {p['name']}: {p['lr']}")
        
        return opt_list, schedule_list

    # --- Part3: Implement other abstract methods for .gaussian.GaussianModel

    @override
    def get_property_names(self) -> tuple[str, ...]:
        if self.state == XrayGassianState.WHOLE:
            return tuple(f"{key}_{XrayGassianState.CORONARY.lower()}" for key in self._keys) + tuple(f"{key}_{XrayGassianState.BACKGROUND.lower()}" for key in self._keys)
        else:
            return tuple(f"{key}_{self.state.lower()}" for key in self._keys)

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
        return self.get_opacities()

    @property
    @override
    def get_opacity(self):
        return self.get_opacities()
    
    # --- Part6: pre-activate all parameters before inference
    
    def pre_activate_all_properties(self):
        self.state = XrayGassianState.WHOLE
        self.is_pre_activated = True
        
        # replace parameters with pre-activated versions
        self.scales = self.get_scales()
        self.rotations = self.get_rotations()
        self.opacities = self.get_opacities()

        self.scale_activation = _identity_act
        self.scale_inverse_activation = _identity_act
        self.rotation_activation = _identity_act
        self.rotation_inverse_activation = _identity_act
        self.opacity_activation = _identity_act
        self.opacity_inverse_activation = _identity_act
    
    def get_non_pre_activated_properties(self):
        if self.is_pre_activated is True:
            activated_properties = self.properties
            keys = list(activated_properties.keys())
            non_pre_activated_properties = {}
            for suffix in ["_coronary", "_background"]:
                scales = "scales" + suffix
                opacities = "opacities" + suffix
                non_pre_activated_properties[scales] = torch.log(activated_properties[scales])
                keys.remove(scales)
                non_pre_activated_properties[opacities] = inverse_sigmoid(activated_properties[opacities])
                keys.remove(opacities)

            for key in keys:
                non_pre_activated_properties[key] = activated_properties[key]

            return non_pre_activated_properties
        else:
            return self.properties 