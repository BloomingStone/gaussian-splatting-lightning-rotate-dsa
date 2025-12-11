import torch
from typing import override

from internal.models.xray_coronary_gaussian import XrayCoronaryGaussianModel
from .vanilla_density_controller import VanillaDensityControllerImpl, VanillaDensityController
from .density_controller import Utils


class RotateXrayDensityController(VanillaDensityController):
    densify_from_iter: int = 2000
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def instantiate(self, *args, **kwargs) -> "RotateXrayDensityControllerImpl":
        return RotateXrayDensityControllerImpl(self)

class RotateXrayDensityControllerImpl(VanillaDensityControllerImpl):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
    
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

    def _densify_and_split(self, grads, gaussian_model: XrayCoronaryGaussianModel, optimizers: list, N: int = 2):
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

    def _prune_points(self, mask, gaussian_model: XrayCoronaryGaussianModel, optimizers: list):
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