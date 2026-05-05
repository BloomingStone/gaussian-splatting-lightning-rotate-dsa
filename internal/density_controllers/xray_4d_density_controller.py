import torch
from collections import deque
from dataclasses import dataclass
from typing import override, cast

from .rotate_xray_density_controller import RotateXrayDensityController
from .rotate_xray_density_controller import RotateXrayDensityControllerImpl
from .density_controller import Utils
from ..gaussian_splatting import GaussianSplatting
from ..models.xray_4d_gaussian import GaussianInits, Xray4DGaussianModel

@dataclass
class Xray4DDensityController(RotateXrayDensityController):    
    
    def instantiate(self, *args, **kwargs) -> "Xray4DDensityControllerImpl":
        return Xray4DDensityControllerImpl(self)


class Xray4DDensityControllerImpl(RotateXrayDensityControllerImpl):
    def __init__(self, config: Xray4DDensityController, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.config = config
    
    @override
    def _densify_and_prune(self, max_screen_size, gaussian_model, optimizers: list):
        min_density = self.config.cull_density_threshold
        prune_extent = self.prune_extent

        # calculate mean grads
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Dynamic threshold: max p95 of recent 5 density-control steps.
        grad_norm = grads.norm(dim=-1)
        valid_grad_norm = grad_norm[torch.isfinite(grad_norm)]
        if valid_grad_norm.numel() > 0:
            grad_p9 = float(torch.quantile(valid_grad_norm, 0.98).item())
            self._recent_grad_p9.append(grad_p9)
            self._grad_threshold = max(self._recent_grad_p9)
        elif self._grad_threshold is None:
            self._grad_threshold = self.config.densify_grad_threshold

        # densify
        self._densify_and_clone(grads, gaussian_model, optimizers)
        self._densify_and_split(grads, gaussian_model, optimizers)

        # prune
        density_mean = gaussian_model.get_density_mean()
        prune_mask = (density_mean < min_density).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = gaussian_model.get_scales().max(dim=1).values > 0.1 * prune_extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self._prune_points(prune_mask, gaussian_model, optimizers)

        torch.cuda.empty_cache()


    def _reset_density(self, gaussian_model, optimizers: list):
        gaussian_model = cast(Xray4DGaussianModel, gaussian_model)
        inits = GaussianInits(
            n_gs=gaussian_model.n_gaussians, 
            density_res_freq_max=gaussian_model.density_freq_res_max
        )
        
        density_freq_dc_old = gaussian_model.density_dc
        
        new_density_freq_dc = torch.min(
            inits.density_dc.to(density_freq_dc_old.device),
            density_freq_dc_old,
        )
        
        new_parameters = Utils.replace_tensors_to_properties(
            tensors={
                gaussian_model._density_dc_name: new_density_freq_dc,
            }, 
            optimizers=optimizers
        )
        gaussian_model.update_properties(new_parameters)
    