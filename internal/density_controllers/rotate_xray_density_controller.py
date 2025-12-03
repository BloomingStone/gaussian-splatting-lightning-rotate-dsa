from typing import Any
from dataclasses import dataclass, field
import torch
from torch import nn
from lightning import LightningModule

from internal.models.xray_coronary_gaussian import XrayCoronaryGaussianModel, XrayGassianState
from internal.utils.general_utils import build_rotation
from .density_controller import DensityController, DensityControllerImpl, Utils
from .vanilla_density_controller import VanillaDensityController

@dataclass
class ControllerConfig:
    percent_dense: float = 0.01

    densification_interval: int = 100

    opacity_reset_interval: int = 3000

    densify_from_iter: int = 500

    densify_until_iter: int = 15_000

    densify_grad_threshold: float = 0.0002

    cull_opacity_threshold: float = 0.005
    """threshold of opacity for culling gaussians."""

    cull_by_max_opacity: bool = False

    camera_extent_factor: float = 1.

    scene_extent_override: float = -1.

    absgrad: bool = False

@dataclass
class RotateXrayDensityController(DensityController):
    coronary_controller: ControllerConfig = field(default_factory=ControllerConfig)
    background_controller: ControllerConfig = field(default_factory=ControllerConfig)

    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        return RotateXrayDensityControllerImpl(self)


class RotateXrayDensityControllerImpl(DensityControllerImpl):
    def __init__(self, config: RotateXrayDensityController, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        
        controller_config_coronary = VanillaDensityController(
            percent_dense           =   config.coronary_controller.percent_dense,
            densification_interval  =   config.coronary_controller.densification_interval,
            opacity_reset_interval  =   config.coronary_controller.opacity_reset_interval,
            densify_from_iter       =   config.coronary_controller.densify_from_iter,
            densify_until_iter      =   config.coronary_controller.densify_until_iter,
            densify_grad_threshold  =   config.coronary_controller.densify_grad_threshold,
            cull_opacity_threshold  =   config.coronary_controller.cull_opacity_threshold,
            cull_by_max_opacity     =   config.coronary_controller.cull_by_max_opacity,
            camera_extent_factor    =   config.coronary_controller.camera_extent_factor,
            scene_extent_override   =   config.coronary_controller.scene_extent_override,
            absgrad                 =   config.coronary_controller.absgrad,
        )
        
        controller_config_background = VanillaDensityController(
            percent_dense           =   config.background_controller.percent_dense,
            densification_interval  =   config.background_controller.densification_interval,
            opacity_reset_interval  =   config.background_controller.opacity_reset_interval,
            densify_from_iter       =   config.background_controller.densify_from_iter,
            densify_until_iter      =   config.background_controller.densify_until_iter,
            densify_grad_threshold  =   config.background_controller.densify_grad_threshold,
            cull_opacity_threshold  =   config.background_controller.cull_opacity_threshold,
            cull_by_max_opacity     =   config.background_controller.cull_by_max_opacity,
            camera_extent_factor    =   config.background_controller.camera_extent_factor,
            scene_extent_override   =   config.background_controller.scene_extent_override,
            absgrad                 =   config.background_controller.absgrad,
        )
        
        self.controller_coronary = controller_config_coronary.instantiate()
        self.controller_background = controller_config_background.instantiate()
    
    def setup(self, stage: str, pl_module: LightningModule) -> None:
        assert isinstance(pl_module.gaussian_model, XrayCoronaryGaussianModel)
        pl_module.gaussian_model.state = XrayGassianState.CORONARY
        self.controller_coronary.setup(stage, pl_module)
        pl_module.gaussian_model.state = XrayGassianState.BACKGROUND
        self.controller_background.setup(stage, pl_module)
        pl_module.gaussian_model.state = XrayGassianState.WHOLE
    
    def before_backward(
        self, 
        outputs: dict[str, dict[XrayGassianState, torch.Tensor]], 
        batch: Any, 
        gaussian_model: XrayCoronaryGaussianModel, 
        optimizers: Any, 
        global_step: int, 
        pl_module: Any
    ) -> None:
        gaussian_model.state = XrayGassianState.CORONARY
        new_outpus = {s: x[gaussian_model.state] for s, x in outputs.items()}
        self.controller_coronary.before_backward(new_outpus, batch, gaussian_model, optimizers, global_step, pl_module) #type: ignore
        
        gaussian_model.state = XrayGassianState.BACKGROUND
        new_outpus = {s: x[gaussian_model.state] for s, x in outputs.items()}
        self.controller_background.before_backward(new_outpus, batch, gaussian_model, optimizers, global_step, pl_module)   #type: ignore
        
        gaussian_model.state = XrayGassianState.WHOLE
    
    def after_backward(
        self, 
        outputs: dict[str, dict[XrayGassianState, torch.Tensor]], 
        batch: Any, 
        gaussian_model: XrayCoronaryGaussianModel, 
        optimizers: list, 
        global_step: int, 
        pl_module: LightningModule
    ) -> None:
        gaussian_model.state = XrayGassianState.CORONARY
        new_outpus = {s: x[gaussian_model.state] for s, x in outputs.items()}
        new_otimizers = [opt for opt in optimizers if opt.param_groups[0]["state"] == gaussian_model.state]
        self.controller_coronary.after_backward(new_outpus, batch, gaussian_model, new_otimizers, global_step, pl_module)
        
        gaussian_model.state = XrayGassianState.BACKGROUND
        new_outpus = {s: x[gaussian_model.state] for s, x in outputs.items()}
        new_otimizers = [opt for opt in optimizers if opt.param_groups[0]["state"] == gaussian_model.state]
        self.controller_background.after_backward(new_outpus, batch, gaussian_model, new_otimizers, global_step, pl_module)
        
        gaussian_model.state = XrayGassianState.WHOLE
    
    def on_load_checkpoint(self, module, checkpoint):
        pass

    def after_density_changed(self, gaussian_model, optimizers: list, pl_module: LightningModule) -> None:
        """
        This interface will be invoked when the density is changed elsewhere
        """
        pass