from typing import Any, override
from dataclasses import dataclass, field
import torch
from torch import nn
from torch.optim import Optimizer
from lightning import LightningModule

from internal.models.xray_coronary_gaussian import XrayCoronaryGaussianModel, XrayGassianState
from internal.utils.general_utils import build_rotation
from .density_controller import DensityController, DensityControllerImpl, Utils
from .vanilla_density_controller import VanillaDensityControllerImpl, VanillaDensityController
from internal.gaussian_splatting import GaussianSplatting

@dataclass
class BackgroundDensityController(VanillaDensityController):
    absgrad: bool = True
    
    @override
    def instantiate(self, *args, **kwargs) -> "BackgroundDensityControllerImpl":
        return BackgroundDensityControllerImpl(self)

class BackgroundDensityControllerImpl(VanillaDensityControllerImpl):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
    
    @override
    def after_backward(
        self, 
        outputs: dict[str, torch.Tensor], 
        batch: Any, 
        gaussian_model: XrayCoronaryGaussianModel, 
        optimizers: list[Optimizer], 
        global_step: int, 
        pl_module: GaussianSplatting
    ) -> None:
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

    @override
    def update_states(
        self, 
        outputs,
        gaussian_model: XrayCoronaryGaussianModel
    ):
        viewspace_point_tensor, visibility_filter, radii = outputs["viewspace_points"], outputs["visibility_filter"], outputs["radii"]
        # retrieve viewspace_points_grad_scale if provided
        viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)

        # update states
        self.max_radii2D[visibility_filter] = torch.max(
            self.max_radii2D[visibility_filter],
            radii[visibility_filter]
        )
        if self.config.absgrad is True:
            xys_grad: torch.Tensor = viewspace_point_tensor.absgrad #type: ignore
        else:
            xys_grad: torch.Tensor = viewspace_point_tensor.grad    #type: ignore
        
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
class CoronaryDensityController(VanillaDensityController):
    movement_var_threshold_percentile: float = 0.90 # filter < 0.90th percentile
    coronary_feature_prune_from_iter: int = 3550    # has 50 offset from densify
    coronary_feature_prune_interval: int = 1000
    coronary_feature_prune_until_iter: int = 10000
    
    scale_2_threshold_percentile: float = 0.90      # filter > 0.90th percentile
    
    def instantiate(self, *args, **kwargs) -> "CoronaryDensityControllerImpl":
        return CoronaryDensityControllerImpl(self)

class CoronaryDensityControllerImpl(BackgroundDensityControllerImpl):
    def __init__(self, config: CoronaryDensityController, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.config = config
        
        self.movement_var_threshold = None
        self.scale_2_threshold = None
    
    @override
    def after_backward(
        self, 
        outputs: dict[str, torch.Tensor], 
        batch: Any, 
        gaussian_model: XrayCoronaryGaussianModel, 
        optimizers: list[Optimizer], 
        global_step: int, 
        pl_module: GaussianSplatting
    ) -> None:
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
            
            # prune by movement variance
            if (
                global_step >= self.config.coronary_feature_prune_from_iter and 
                global_step < self.config.coronary_feature_prune_until_iter and
                (global_step - self.config.coronary_feature_prune_from_iter) % self.config.coronary_feature_prune_interval == 0
            ):
                self._prune_by_movement_var_and_scale(gaussian_model, optimizers)

            if global_step % self.config.opacity_reset_interval == 0 or \
                    (
                        torch.all(pl_module.background_color == 1.) and global_step == self.config.densify_from_iter
                    ):
                self._reset_opacities(gaussian_model, optimizers)   #type: ignore
                self.opacity_reset_at = global_step
    
    def _prune_by_movement_var_and_scale(self, gaussian_model: XrayCoronaryGaussianModel, optimizers: list[Optimizer]):
        xyz_motion_norm = torch.norm(gaussian_model.get_motion_var()[:, :3], dim=1)
        if self.movement_var_threshold is None:
            self.movement_var_threshold = torch.quantile(xyz_motion_norm, self.config.movement_var_threshold_percentile)
        prune_mask = torch.where(
            xyz_motion_norm < self.movement_var_threshold,
            True,
            False
        )
        self._prune_points(prune_mask, gaussian_model, optimizers)
        
        # The Gaski elements at the coronary area have at most only one direction where the scale is relatively larger, 
        # that is, the "second largest" scale is relatively smaller.
        scale = gaussian_model.get_scales().squeeze()
        s = scale.sort(dim=-1).values
        s2 = s[:, 1]                    # "second largest" scale
        if self.scale_2_threshold is None:
            T = torch.quantile(s2, self.config.scale_2_threshold_percentile)
        prune_mask = torch.where(
            s2 > T,
            True,
            False
        )
        self._prune_points(prune_mask, gaussian_model, optimizers)
    
    
    @override
    def _densify_and_clone(self, grads, gaussian_model: XrayCoronaryGaussianModel, optimizers: list):
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # Exclude big Gaussians
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(gaussian_model.get_scales(), dim=1).values <= percent_dense * scene_extent,
        )

        # Copy selected Gaussians
        new_properties = {}
        for key, value in gaussian_model.properties.items():
            new_properties[key] = value[selected_pts_mask]

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)
        gaussian_model.clone_motion_by_mask(selected_pts_mask, repeats=1)
    
    @override
    def _densify_and_split(self, grads: torch.Tensor, gaussian_model: XrayCoronaryGaussianModel, optimizers: list, N: int = 2):
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        device = gaussian_model.get_property("means").device
        n_init_points = gaussian_model.n_gaussians
        scales = gaussian_model.get_scales()

        # The number of Gaussians and `grads` is different after cloning, so padding is required
        padded_grad = torch.zeros((n_init_points,), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # Exclude small Gaussians
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(
                scales,
                dim=1,
            ).values > percent_dense * scene_extent,
        )

        # Split
        new_properties = self._split_properties(gaussian_model, selected_pts_mask, N)

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)
        
        gaussian_model.clone_motion_by_mask(selected_pts_mask, repeats=N)
        
        # Prune selected Gaussians, since they are already split
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(
                N * int(selected_pts_mask.sum().item()),
                device=device,
                dtype=torch.bool,
            ),
        ))
        self._prune_points(prune_filter, gaussian_model, optimizers)
    
    @override
    def _prune_points(
        self,
        mask: torch.Tensor, 
        gaussian_model: XrayCoronaryGaussianModel, 
        optimizers: list[Optimizer],
    ):        
        """
        Args:
            mask: `True` indicating the Gaussians to be pruned
            gaussian_model
            optimizers
        """
        
        valid_points_mask = ~mask  # `True` to keep
        
        new_parameters = Utils.prune_properties(valid_points_mask, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters
        
        gaussian_model.filter_motion_by_mask(valid_points_mask)

        # prune states
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

@dataclass
class RotateXrayDensityController(DensityController):
    backgound_cfg: BackgroundDensityController = field(default_factory=BackgroundDensityController)
    coronary_cfg: CoronaryDensityController = field(default_factory=CoronaryDensityController)
    
    @override
    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        return RotateXrayDensityControllerImpl(self)


class RotateXrayDensityControllerImpl(DensityControllerImpl):
    def __init__(self, config: RotateXrayDensityController, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        
        self.controller_coronary = config.coronary_cfg.instantiate()
        self.controller_background = config.backgound_cfg.instantiate()
    
    @override
    def setup(self, stage: str, pl_module: LightningModule) -> None:
        assert isinstance(pl_module.gaussian_model, XrayCoronaryGaussianModel)
        pl_module.gaussian_model.state = XrayGassianState.CORONARY
        self.controller_coronary.setup(stage, pl_module)
        pl_module.gaussian_model.state = XrayGassianState.BACKGROUND
        self.controller_background.setup(stage, pl_module)
        pl_module.gaussian_model.state = XrayGassianState.WHOLE
    
    @override
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
    
    @override
    def after_backward(
        self, 
        outputs: dict[str, dict[XrayGassianState, torch.Tensor]], 
        batch: Any, 
        gaussian_model: XrayCoronaryGaussianModel, 
        optimizers: list, 
        global_step: int, 
        pl_module: GaussianSplatting
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
    
    @override
    def on_load_checkpoint(self, module, checkpoint):
        pass
    
    @override
    def after_density_changed(self, gaussian_model, optimizers: list, pl_module: LightningModule) -> None:
        """
        This interface will be invoked when the density is changed elsewhere
        """
        pass