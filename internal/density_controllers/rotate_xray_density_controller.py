import torch
from collections import deque
from dataclasses import dataclass
from typing import override, cast

from .vanilla_density_controller import VanillaDensityControllerImpl, DensityController
from . import utils as utils
from ..models.xray_coronary_gaussian import GaussianInits, XrayCoronaryGaussianModel
from ..renderers.renderer import RendererOutputs

@dataclass
class RotateXrayDensityController(DensityController):    
    # in C-arm X-ray, the scene extent = SOD > 1500 mm. so here we set percent_dense to 0.0005, 
    # which means the maximum scaling of a Gaussian is 1500 * 0.0005 = 0.75 mm.
    # the diameter of a coronary artery is around 2-3 mm, so 3*sigma = 2.25 probably covers the artery, which is a reasonable setting.
    percent_dense: float = 0.0005
    
    densification_interval: int = 300

    density_reset_interval: int = 2000

    densify_from_iter: int = 500
    
    densify_until_frac: float = 0.8
    
    densify_grad_percentile: float = 0.98

    # If > 0, use a fixed gradient threshold instead of dynamic percentile-based threshold.
    # Set to the average grad_threshold from baseline to compare fixed vs dynamic.
    fixed_grad_threshold: float = -1.0

    cull_density_threshold: float = 5e-4
    
    max_n_gaussians: float = 1e6

    camera_extent_factor: float = 1.

    scene_extent_override: float = -1.

    absgrad: bool = False
    
    def instantiate(self, *args, **kwargs) -> "RotateXrayDensityControllerImpl":
        return RotateXrayDensityControllerImpl(self)


class RotateXrayDensityControllerImpl(VanillaDensityControllerImpl):
    def __init__(self, config: RotateXrayDensityController, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.config = config
    
    @override
    def setup(self, stage: str, pl_module) -> None:
        super().setup(stage, pl_module)

        if stage == "fit":
            self.cameras_extent = pl_module.trainer.datamodule.dataparser_outputs.camera_extent * self.config.camera_extent_factor  #type: ignore
            self.prune_extent = pl_module.trainer.datamodule.prune_extent * self.config.camera_extent_factor    #type: ignore

            if self.config.scene_extent_override > 0:
                self.cameras_extent = self.config.scene_extent_override
                self.prune_extent = self.config.scene_extent_override
                print(f"Override scene extent with {self.config.scene_extent_override}")

            self._init_state(pl_module.gaussian_model.n_gaussians, pl_module.device)

        self.density_reset_at = -32768
        self._grad_threshold: float|None = None
        self._recent_grad_percentile: deque[float] = deque(maxlen=5)
    
        
    @override
    def before_backward(self, outputs: RendererOutputs, batch, gaussian_model, optimizers: list, global_step: int, pl_module) -> None:
        if global_step >= self.config.densify_until_frac*pl_module.trainer.max_steps:
            return

        outputs.meta["viewspace_points"].retain_grad()
        
    
    @override
    def after_backward(self, outputs: RendererOutputs, batch, gaussian_model, optimizers: list, global_step: int, pl_module) -> None:
        if global_step >= self.config.densify_until_frac*pl_module.trainer.max_steps:
            return

        with torch.no_grad():
            self.update_states(outputs)

            # densify and pruning
            # avoid densifying right after density reset
            if (
                global_step > self.config.densify_from_iter\
                and global_step % self.config.densification_interval == 0\
                and global_step % self.config.density_reset_interval >= self.config.densification_interval\
                and gaussian_model.n_gaussians < self.config.max_n_gaussians
            ): 
                size_threshold = 20 if global_step > self.config.density_reset_interval else None
                self._densify_and_prune(
                    max_screen_size=size_threshold,
                    gaussian_model=gaussian_model,
                    optimizers=optimizers,
                    pl_module=pl_module,
                )
                # log grad threshold for analysis (only on densify steps)
                if self._grad_threshold is not None:
                    pl_module.log(
                        "density/grad_threshold",
                        self._grad_threshold,
                        on_step=True,
                        on_epoch=False,
                        batch_size=1,
                    )

            if global_step % self.config.density_reset_interval == 0 or \
                    (
                        torch.all(pl_module.background_color == 1.) and global_step == self.config.densify_from_iter
                    ):
                self._reset_density(gaussian_model, optimizers)
                self.density_reset_at = global_step
    
    
    @override
    def _densify_and_prune(self, max_screen_size, gaussian_model, optimizers: list, pl_module=None):
        min_density = self.config.cull_density_threshold
        prune_extent = self.prune_extent

        # calculate mean grads
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Threshold: fixed or dynamic (max percentile of grad in recent 5 density-control steps).
        grad_norm = grads.norm(dim=-1)
        valid_grad_norm = grad_norm[torch.isfinite(grad_norm)]

        if self.config.fixed_grad_threshold > 0:
            self._grad_threshold = self.config.fixed_grad_threshold
            if pl_module is not None:
                pl_module.log(
                    "density/grad_threshold",
                    self._grad_threshold,
                    on_step=True,
                    on_epoch=False,
                    batch_size=1,
                )
        else:
            if valid_grad_norm.numel() > 0:
                grad_percentile = float(torch.quantile(valid_grad_norm, self.config.densify_grad_percentile).item())
                self._recent_grad_percentile.append(grad_percentile)
                self._grad_threshold = max(self._recent_grad_percentile)
            elif self._grad_threshold is None:
                self._grad_threshold = float(torch.quantile(valid_grad_norm, self.config.densify_grad_percentile).item())

        # densify
        self._densify_and_clone(grads, gaussian_model, optimizers)
        self._densify_and_split(grads, gaussian_model, optimizers)

        # prune
        prune_mask = (gaussian_model.get_density() < min_density).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = gaussian_model.get_scales().max(dim=1).values > 0.1 * prune_extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self._prune_points(prune_mask, gaussian_model, optimizers)

        torch.cuda.empty_cache()


    @override
    def _densify_and_clone(self, grads, gaussian_model, optimizers: list):
        gaussian_model = cast(XrayCoronaryGaussianModel, gaussian_model)
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        grad_norm = grads.norm(dim=-1)
        valid_grad_norm = grad_norm[torch.isfinite(grad_norm)]

        grad_threshold = self._grad_threshold if self._grad_threshold is not None else float(torch.quantile(valid_grad_norm, self.config.densify_grad_percentile).item())

        selected_pts_mask = grad_norm >= grad_threshold
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
        gaussian_model.deforms_recorder.clone_deforms_by_mask(selected_pts_mask, repeats=1)

    @override
    def _densify_and_split(self, grads, gaussian_model, optimizers: list, N: int = 2):
        gaussian_model = cast(XrayCoronaryGaussianModel, gaussian_model)
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        device = gaussian_model.get_property("means").device
        n_init_points = gaussian_model.n_gaussians
        scales = gaussian_model.get_scales()

        # The number of Gaussians and `grads` is different after cloning, so padding is required
        padded_grad = torch.zeros((n_init_points,), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        valid_grad_norm = padded_grad[torch.isfinite(padded_grad)]
        
        grad_threshold = self._grad_threshold if self._grad_threshold is not None else float(torch.quantile(valid_grad_norm, self.config.densify_grad_percentile).item())
        
        # Extract points that satisfy the gradient condition
        selected_pts_mask = (padded_grad >= grad_threshold)
        # Exclude small Gaussians
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(
                scales,
                dim=1,
            ).values > percent_dense * scene_extent,
        )
        
        # Exclude nan points
        properties = gaussian_model.get_properties()
        properties = torch.cat([v for v in properties.values()], dim=-1)
        nan_mask = torch.any(torch.isnan(properties), dim=-1)
        if nan_mask.any():
            print(f"Warning: NaN detected in properties. {nan_mask.sum().item()} points will be excluded.")
        selected_pts_mask = torch.logical_and(selected_pts_mask, ~nan_mask)

        # Split
        new_properties = self._split_properties(gaussian_model, selected_pts_mask, N)

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)
        
        gaussian_model.deforms_recorder.clone_deforms_by_mask(selected_pts_mask, repeats=N)

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
    def _prune_points(self, mask, gaussian_model, optimizers: list):
        """
        Args:
            mask: `True` indicating the Gaussians to be pruned
            gaussian_model
            optimizers
        """
        
        valid_points_mask = ~mask  # `True` to keep
        
        new_parameters = utils.prune_properties(valid_points_mask, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters
        
        gaussian_model.deforms_recorder.filter_deforms_by_mask(valid_points_mask)

        # prune states
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def _reset_density(self, gaussian_model, optimizers: list):
        inits = GaussianInits(gaussian_model.n_gaussians)
        density = gaussian_model.get_density()
        density_new = gaussian_model.density_inverse_activation(torch.min(
            density,
            gaussian_model.density_activation(inits.density).to(density.device),
        ))
        new_parameters = utils.replace_tensors_to_properties(tensors={
            "density": density_new,
        }, optimizers=optimizers)
        gaussian_model.update_properties(new_parameters)
    