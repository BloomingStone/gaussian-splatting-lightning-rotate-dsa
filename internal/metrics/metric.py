from typing import Tuple, Dict, Any, Callable, cast
from dataclasses import dataclass, field

import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from lightning import LightningModule
from torch import Tensor

from ..dataparsers.dataparser import BatchT
from ..dataparsers.xray_dataparser.meta import XRayMeta
from .ssim import ssim
from ..models.gaussian import GaussianModel
from ..renderers.renderer import RendererOutputs
from ..instantiate_config import Instantiable

class MetricImpl(torch.nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__()
        self.config = config

    def setup(self, stage: str, pl_module):
        pass

    def get_train_metrics(
        self, 
        pl_module: LightningModule, 
        gaussian_model: GaussianModel, 
        step: int, 
        batch: BatchT, 
        outputs: RendererOutputs
    ) -> Tuple[Dict[str, Tensor|float], Dict[str, bool]]:
        """
        :return:
            The first dict: contains the metric values.
                The `backward()` only will be invoked for the one with key `loss`.
                All other values are only for logging.
            The second dict: indicates whether the metric value should be shown on progress bar
        """

        return self.get_validate_metrics(
            pl_module=pl_module,
            gaussian_model=gaussian_model,
            batch=batch,
            outputs=outputs,
        )

    def training_setup(self, pl_module) -> tuple[list[Optimizer]|None, list[LRScheduler]|None]:
        return [], []

    def get_validate_metrics(self, pl_module, gaussian_model, batch: BatchT, outputs) -> Tuple[Dict[str, Tensor|float], Dict[str, bool]]:
        raise NotImplementedError

    def on_parameter_move(self, *args, **kwargs):
        raise NotImplementedError
    
    def on_validation_epoch_start(self, pl_module):
        pass


@dataclass
class metric3DConfig:
    # --- 3D segmentation metric configuration ---
    thresholds_absolute: tuple[float, ...] = (0.0344,)
    """Fixed absolute thresholds for volume binarization."""

    thresholds_percentile: tuple[float, ...] = (0.99, 0.995, 0.998)

    closing_radius_vox: int = 1
    """Radius (voxels) for morphological closing after thresholding."""

    connectivity: int = 1
    """3D connectivity for connected-component analysis (1, 2, or 3)."""

    min_component_size_vox: int = 0
    """Minimum component size in voxels (0 = no filtering)."""


class CommonImageMetricImpl(MetricImpl):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.no_state_dict_models: Dict[str, torch.nn.Module] = {}
        self.metric3d: dict[str, torch.Tensor] = {}

    @staticmethod
    def _create_fused_ssim_adapter() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        from fused_ssim import fused_ssim

        def adapter(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
            return fused_ssim(pred, gt)

        return adapter

    def setup(self, stage: str, pl_module):
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.no_state_dict_models["lpips"] = LearnedPerceptualImagePatchSimilarity(
            normalize=True,
            net_type=self.config.lpips_net_type,
        )

        self.rgb_diff_loss_fn = self._l1_loss
        if self.config.rgb_diff_loss == "l2":
            print("Use L2 loss")
            self.rgb_diff_loss_fn = self._l2_loss

        self.ssim = ssim
        if self.config.fused_ssim:
            print("Fused SSIM enabled")
            self.ssim = self._create_fused_ssim_adapter()

    @staticmethod
    def _ensure_gray_nchw(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        elif image.dim() != 4:
            raise ValueError(f"Unsupported image dim: {image.dim()}")

        if image.shape[1] != 1:
            image = image[:, :1]

        return image

    def add_image_validation_metrics(
        self,
        metrics: Dict[str, Any],
        prog_bar: Dict[str, bool],
        pred_gray: torch.Tensor,
        gt_gray: torch.Tensor,
    ) -> None:
        pred_gray = self._ensure_gray_nchw(pred_gray)
        gt_gray = self._ensure_gray_nchw(gt_gray)

        metrics["psnr"] = self.psnr(pred_gray, gt_gray)
        prog_bar["psnr"] = True

        pred_rgb = pred_gray.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        gt_rgb = gt_gray.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        metrics["lpips"] = self.no_state_dict_models["lpips"](pred_rgb, gt_rgb)
        prog_bar["lpips"] = True

    def add_weighted_validation_metrics(
        self,
        metrics: Dict[str, Any],
        prog_bar: Dict[str, bool],
        pred_gray: torch.Tensor,
        gt_gray: torch.Tensor,
        weight: torch.Tensor | None,
    ) -> None:
        """Compute validation metrics weighted by *weight* (H, W) — 1 inside GT mask,
        Gaussian falloff outside.  When *weight* is None this is a no-op."""
        if weight is None:
            return

        pred_gray = self._ensure_gray_nchw(pred_gray)   # (1,1,H,W)
        gt_gray = self._ensure_gray_nchw(gt_gray)

        # ── weighted L1 ──────────────────────────────────────────────
        w = weight.to(dtype=pred_gray.dtype, device=pred_gray.device).unsqueeze(0).unsqueeze(0)
        w_sum = w.sum().clamp_min(1e-8)
        weighted_l1 = (torch.abs(pred_gray - gt_gray) * w).sum() / w_sum
        metrics["w_l1"] = weighted_l1
        prog_bar["w_l1"] = True

        # ── weighted SSIM (per-pixel ssim_map, then weighted average) ─
        ssim_map = ssim(pred_gray, gt_gray, size_average=False)  # (1,1,H,W)
        weighted_ssim = (ssim_map * w).sum() / w_sum
        metrics["w_ssim"] = weighted_ssim
        metrics["w_ssim_loss"] = 1.0 - weighted_ssim
        prog_bar["w_ssim"] = True
        
        # ── combined psnr + lpips loss with the same weight ──────────────────────────────
        metrics["w_psnr"] = self.psnr(pred_gray, gt_gray)
        prog_bar["w_psnr"] = True

        pred_rgb = pred_gray.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        gt_rgb = gt_gray.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        metrics["w_lpips"] = self.no_state_dict_models["lpips"](pred_rgb, gt_rgb)
        prog_bar["w_lpips"] = True

    def on_parameter_move(self, *args, **kwargs):
        if "lpips" in self.no_state_dict_models:
            self.no_state_dict_models["lpips"] = self.no_state_dict_models["lpips"].to(*args, **kwargs)

    @staticmethod
    def _l1_loss(predict: torch.Tensor, gt: torch.Tensor):
        return torch.abs(predict - gt).mean()

    @staticmethod
    def _l2_loss(predict: torch.Tensor, gt: torch.Tensor):
        return torch.mean((predict - gt) ** 2)
    
    def _compute_3d_metrics(
        self,
        pl_module,
        uniformed_time: float = 0.5,    # time 
        cardiac_phase: float = 0.,
    ) -> dict[str, torch.Tensor]:
        r"""Compute 3D segmentation metrics (Dice, HD95, clDice, etc.).

        Steps:
        1. Get GT 3D label from dataparser (loaded at init time).
        2. Deform gaussians to given cardiac phase.
        3. Rasterize deformed gaussians into a 3D volume.
        4. For each threshold (absolute + percentile), segment & compute metrics.
        
        Args:
            pl_module: the LightningModule, used to access dataparser, gaussian model, etc
            uniformed_time: the time point to deform the gaussians to (if deformable). \in [0,1], 
                0 is the start time of all frames, 1 is the end time. Default 0.5 where idodine contrast 
                is usually filled the most.
            cardiac_phase: the cardiac phase to deform the gaussians to (if deformable). \in [0,1], 0 is 
                the start of cardiac cycle, 1 is the end of cardiac cycle. Default 0, which is the same as
                generated reference 3D label.

        Returns:
            dict with keys like ``metric3D/thd-0.0344/dice``, ``metric3D/thd-90%/hd95``, etc.
            Empty dict on any error (logs a warning).
        """
        try:
            # --- 1. Get GT label & meta ---
            datamodule = pl_module.get_datamodule()
            meta = cast(XRayMeta, datamodule.dataparser_outputs.meta)
            if meta.label_3d_info is None:
                return {}  # no GT label available, skip 3D metrics

            label_info = meta.label_3d_info
            gt_label = label_info.data                     # (D, H, W) bool  (numpy)
            if gt_label is None:
                return {}
            gt_label = torch.from_numpy(gt_label).to(device=pl_module.device, dtype=torch.bool)  # (D, H, W) bool tensor on the same device as model
            aabb_roi_np = label_info.aabb                 # numpy bool mask

            device = pl_module.device
            aabb_roi = torch.from_numpy(aabb_roi_np).to(device=device, dtype=torch.bool)

            volume_shape = tuple(int(x) for x in meta.volume_size)
            coronary_affine = meta.centering_affine
            spacing = np.diag(coronary_affine)[:3]

            # --- 2. Deform gaussians to the requested cardiac phase ---
            gaussian_model = pl_module.gaussian_model
            means3D = gaussian_model.get_means().detach()
            scales = gaussian_model.get_scales().detach()
            rotation = gaussian_model.get_rotations().detach()
            density = gaussian_model.get_density().detach()

            from ..deform_models.deform_model import GSParam
            renderer = pl_module.renderer
            if hasattr(renderer, 'deform_model'):
                deform_model = renderer.deform_model
                with torch.no_grad():
                    deforms = deform_model(
                        xyz=means3D,
                        t=torch.full((means3D.shape[0], 1), uniformed_time, device=device),
                        phase=torch.full((means3D.shape[0], 1), cardiac_phase, device=device),
                    )
                    new_gsparam = deform_model.deform(
                        GSParam(xyz=means3D, scaling=scales, rotation=rotation, density=density),
                        deforms,
                    )
            else:
                new_gsparam = GSParam(xyz=means3D, scaling=scales, rotation=rotation, density=density)

            # --- 3. Rasterize to CUDA volume ---
            from ..savers.x_ray_saver import gaussians_to_volume_by_Rasterizer

            with torch.no_grad():
                vol_pred = gaussians_to_volume_by_Rasterizer(
                    means3D=new_gsparam.xyz,
                    scales=new_gsparam.scaling,
                    rotation=new_gsparam.rotation,
                    density=new_gsparam.density,
                    shape=volume_shape,
                    affine=coronary_affine,
                    to_cpu=False,
                )
                assert isinstance(vol_pred, torch.Tensor)  # still on CUDA

            # --- 4. Segmentation & metrics for each threshold ---
            from .metric_3d_utils import (
                compute_all_metrics,
                compute_density_based_metrics,
            )

            thresholds: list[tuple[str, float]] = []
            cfg = cast(metric3DConfig, self.config.metric3d_cfg)
            for thr in cfg.thresholds_absolute:
                thresholds.append((f"thd-{thr:.4f}", float(thr)))

            if cfg.thresholds_percentile:
                vol_roi = vol_pred[aabb_roi]               # GPU indexing
                if vol_roi.numel() > 0:
                    for pct in cfg.thresholds_percentile:
                        thr_val = float(torch.quantile(vol_roi, pct))
                        thresholds.append((f"thd-{pct * 100:.2f}%", thr_val))

            result: dict[str, torch.Tensor] = {}
            for thr_key, thr_val in thresholds:
                pred = ((vol_pred > thr_val) & aabb_roi).to(device=pl_module.device, dtype=torch.bool)       # CUDA bool tensor
                
                metrics = compute_all_metrics(pred, gt_label, tuple(spacing))
                for metric_name, metric_val in metrics.items():
                    result[f"metric3D/{thr_key}/{metric_name}"] = torch.tensor(
                        metric_val, dtype=torch.float32, device=device,
                    )

            # --- 5. Density-based metrics (threshold-free) ---
            density_metrics = compute_density_based_metrics(
                density=vol_pred,
                gt=gt_label,
                roi=aabb_roi,
            )
            for metric_name, metric_val in density_metrics.items():
                result[f"metric3D/density/{metric_name}"] = torch.tensor(
                    metric_val, dtype=torch.float32, device=device,
                )

            return result
        except Exception as e:
            import warnings
            warnings.warn(f"Error computing 3D metrics: {e}")
            return {}

@dataclass
class Metric(Instantiable):
    metric3d_cfg: metric3DConfig = field(default_factory=metric3DConfig)
    
    def instantiate(self, *args, **kwargs) -> MetricImpl:
        raise NotImplementedError
