from typing import Any
from dataclasses import dataclass
import torch
from torch import nn
from lightning import LightningModule

from internal.models.xray_coronary_gaussian import XrayCoronaryGaussianModel, XrayGassianState
from internal.utils.general_utils import build_rotation
from .density_controller import DensityController, DensityControllerImpl, Utils
from .vanilla_density_controller import VanillaDensityControllerImpl, VanillaDensityController


class MyVanillaDensityController(VanillaDensityController):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def instantiate(self, *args, **kwargs) -> "MyVanillaDensityControllerImpl":
        return MyVanillaDensityControllerImpl(self)

class MyVanillaDensityControllerImpl(VanillaDensityControllerImpl):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
    
    def after_backward(self, outputs: dict, batch, gaussian_model: XrayCoronaryGaussianModel, optimizers: list, global_step: int, pl_module: LightningModule) -> None:
        if global_step >= self.config.densify_until_iter:
            return

        with torch.no_grad():
            self.update_states(outputs, gaussian_model)

            # densify and pruning
            if global_step > self.config.densify_from_iter and global_step % self.config.densification_interval == 0:
                size_threshold = 20 if global_step > self.config.opacity_reset_interval else None
                self._densify_and_prune(
                    max_screen_size=size_threshold,
                    gaussian_model=gaussian_model,  #type: ignore
                    optimizers=optimizers,
                )

            if global_step % self.config.opacity_reset_interval == 0 or \
                    (
                        torch.all(pl_module.background_color == 1.) and global_step == self.config.densify_from_iter
                    ):
                self._reset_opacities(gaussian_model, optimizers)   #type: ignore
                self.opacity_reset_at = global_step

    def update_states(self, outputs, gaussian_model):
        viewspace_point_tensor, visibility_filter, radii = outputs["viewspace_points"], outputs["visibility_filter"], outputs["radii"]
        # retrieve viewspace_points_grad_scale if provided
        viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)

        # update states
        self.max_radii2D[visibility_filter] = torch.max(
            self.max_radii2D[visibility_filter],
            radii[visibility_filter]
        )
        xys_grad = viewspace_point_tensor.grad
        if self.config.absgrad is True:
            xys_grad = viewspace_point_tensor.absgrad
        
        assert isinstance(gaussian_model, XrayCoronaryGaussianModel)
        xys_grad = xys_grad.squeeze()
        state = gaussian_model.state
        if state == "coronary":
            n = gaussian_model.gaussians.n_coronary_gs
            assert n is not None
            xys_grad = xys_grad[:n]
        elif state == "background":
            n = gaussian_model.gaussians.n_background_gs
            assert n is not None
            xys_grad = xys_grad[-n:]
        else:
            raise ValueError(f"Unknown state: {state}")
        
        if xys_grad.shape[0] != visibility_filter.shape[0]:
            pass
        
        self._add_densification_stats(xys_grad, visibility_filter, scale=viewspace_points_grad_scale)
    

@dataclass
class RotateXrayDensityController(DensityController):
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

    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        return RotateXrayDensityControllerImpl(self)


class RotateXrayDensityControllerImpl(DensityControllerImpl):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        
        controller_config_coronary = MyVanillaDensityController(
            percent_dense=0.01,
            densification_interval=100,
            opacity_reset_interval=config.opacity_reset_interval,
            densify_from_iter=2000,
            densify_until_iter=config.densify_until_iter,
            densify_grad_threshold=config.densify_grad_threshold,
            cull_opacity_threshold=config.cull_opacity_threshold,
            cull_by_max_opacity=config.cull_by_max_opacity,
            camera_extent_factor=config.camera_extent_factor,
            scene_extent_override=config.scene_extent_override,
            absgrad=config.absgrad,
        )
        
        controller_config_background = MyVanillaDensityController(
            percent_dense=config.percent_dense,
            densification_interval=config.densification_interval,
            opacity_reset_interval=config.opacity_reset_interval,
            densify_from_iter=config.densify_from_iter,
            densify_until_iter=config.densify_until_iter,
            densify_grad_threshold=config.densify_grad_threshold,
            cull_opacity_threshold=config.cull_opacity_threshold,
            cull_by_max_opacity=config.cull_by_max_opacity,
            camera_extent_factor=config.camera_extent_factor,
            scene_extent_override=config.scene_extent_override,
            absgrad=config.absgrad,
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