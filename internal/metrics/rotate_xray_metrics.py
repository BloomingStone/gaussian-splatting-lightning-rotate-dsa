from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any

import torch
from pytorch_lightning import LightningModule
import numpy as np

from internal.metrics.metric import Metric, MetricImpl, CommonImageMetricImpl
from internal.renderers.xray_4d_renderer import RenderRes
from internal.dataparsers.dataparser import BatchT
from ..renderers.deformabel_xray_renderer_coronary_props import XrayRendererOuputs


@dataclass
class RotateXrayMetrics(Metric):
    w_gray_loss: float = 1.0
    w_ssim_loss: float = 1.0

    rgb_diff_loss: Literal["l1", "l2"] = "l1"

    lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex"
    """
    the vanilla 3DGS uses 'vgg', but 'alex' is faster
    """

    # 目前如果将 fused_ssim 设为 False 会导致 loss 无法正常下降， 可能是因为 pytorch 实现的ssim处理半精度时有问题
    fused_ssim: bool = True
    
    # --- 3D segmentation metric configuration ---
    thresholds_absolute: tuple[float, ...] = (0.0344,)
    """Fixed absolute thresholds for volume binarization."""

    thresholds_percentile: tuple[float, ...] = ()
    """Percentile-based thresholds, e.g. (0.9, 0.8, 0.7) → 90th, 80th, 70th percentile."""

    closing_radius_vox: int = 1
    """Radius (voxels) for morphological closing after thresholding."""

    connectivity: int = 1
    """3D connectivity for connected-component analysis (1, 2, or 3)."""

    min_component_size_vox: int = 0
    """Minimum component size in voxels (0 = no filtering)."""

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return RotateXrayMetricsImpl(self)


class RotateXrayMetricsImpl(CommonImageMetricImpl):
    config:  RotateXrayMetrics

    def _get_basic_metrics(
        self, 
        pl_module: LightningModule, 
        gaussian_model, 
        batch: BatchT, 
        outputs: XrayRendererOuputs
    ):
        _, image_info, _ = batch   # load depth_map as extra_data in internal/dataparsers/rotated_xray_dataparser.py
        _, gt_image, _ = image_info
        gt_image = self._ensure_gray_nchw(gt_image)
        pred_gray = self._ensure_gray_nchw(outputs.gray_image)
        
        gray_loss = self.rgb_diff_loss_fn(pred_gray, gt_image)
        
        ssim_metric = self.ssim(pred_gray, gt_image)
        ssim_loss = 1.0 - ssim_metric

        loss = (
            # image loss
            self.config.w_gray_loss * gray_loss +
            self.config.w_ssim_loss * ssim_loss
        )


        assert not torch.isnan(loss), "Loss is NaN!"
        
        metrics = {
            "loss": loss,
            "gray_loss": gray_loss,
            "ssim_loss": ssim_loss,
        }


        prog_bar = {
            "loss": True,
            "gray_loss": True,
            "ssim_loss": True,
        }
        return metrics, prog_bar
    

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch: BatchT, outputs: XrayRendererOuputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:    # type: ignore
        return self._get_basic_metrics(
            pl_module=pl_module,
            gaussian_model=gaussian_model,
            batch=batch,
            outputs=outputs
        )
    
    def get_validate_metrics(self, pl_module, gaussian_model, batch: BatchT, outputs: XrayRendererOuputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, prog_bar = self._get_basic_metrics(
            pl_module,
            gaussian_model,
            batch,
            outputs
        )

        _, image_info, _ = batch   # load depth_map as extra_data in internal/dataparsers/rotated_xray_dataparser.py
        _, gt_image, _ = image_info
        gt_image = self._ensure_gray_nchw(gt_image)

        self.add_image_validation_metrics(metrics, prog_bar, outputs.gray_image, gt_image)

        return metrics, prog_bar
    
    def on_validation_epoch_start(self, pl_module):
        return super().on_validation_epoch_start(pl_module)
        # self._compute_3d_metrics(
        #     pl_module=pl_module,
        #     gaussian_model=gaussian_model
        # )
    
    def _compute_3d_metrics(
        self, 
        pl_module,
        cardiac_phase: float = 0.,  # (1,) in [0, 1]
    ) -> dict[str, torch.Tensor]:
        """Compute 3D segmentation metrics (Dice, HD95, ASSD, clDice, etc.).
        
        Steps:
        1. Get GT 3D label from dataparser (loaded at init time).
        2. Deform gaussians to given cardiac phase.
        3. Rasterize deformed gaussians into a 3D volume.
        4. For each threshold (absolute + percentile), segment & compute metrics.
        
        Returns:
            dict with keys like ``metric3D/thd-0.0344/dice``, ``metric3D/thd-90%/hd95``, etc.
            Empty dict on any error (logs a warning).
        """
        import logging
        _logger = logging.getLogger(__name__)
        
        try:
            # --- 1. Get GT label from dataparser ---
            datamodule = pl_module.get_datamodule()
            dataparser = datamodule.dataparser
            
            gt_label = dataparser.label_3d          # bool ndarray (D, H, W)
            gt_spacing = dataparser.label_3d_spacing  # (sx, sy, sz) mm
            
            volume_shape = dataparser.volume_shape   # (D, H, W)
            coronary_affine = dataparser.coronary_affine  # (4, 4)
            
            _logger.info(
                "Computing 3D metrics: gt_shape=%s, gt_spacing=%s, volume_shape=%s",
                gt_label.shape, gt_spacing, volume_shape,
            )
            
            # --- 2. Deform gaussians to the current batch phase ---
            gaussian_model = pl_module.gaussian_model
            means3D = gaussian_model.get_means().detach()         # (N, 3)
            scales = gaussian_model.get_scales().detach()          # (N, 3)
            rotation = gaussian_model.get_rotations().detach()     # (N, 4)  quaternion
            density = gaussian_model.get_density().detach()        # (N, 1)
            
            # Access the deform model from the renderer
            renderer = pl_module.renderer
            if not hasattr(renderer, 'deform_model'):
                _logger.warning("Renderer has no deform_model; skipping 3D metrics.")
                return {}
            deform_model = renderer.deform_model
            
            from ..deform_models.deform_model import DeformModel, GSParam
            
            with torch.no_grad():
                deforms = deform_model(
                    xyz=means3D, 
                    time=torch.full((means3D.shape[0],), cardiac_phase, device=means3D.device),
                    phase=torch.full((means3D.shape[0],), cardiac_phase, device=means3D.device)
                )
                new_gsparam = DeformModel.deform(
                    GSParam(xyz=means3D, scaling=scales, rotation=rotation, density=density),
                    deforms
                )
            
            # --- 3. Rasterize to volume ---
            from ..savers.x_ray_saver import gaussians_to_volume_by_Rasterizer
            
            with torch.no_grad():
                vol_pred = gaussians_to_volume_by_Rasterizer(
                    means3D=new_gsparam.xyz,
                    scales=new_gsparam.scaling,
                    rotation=new_gsparam.rotation,
                    density=new_gsparam.density,
                    shape=volume_shape,
                    affine=coronary_affine,
                )
            # vol_pred: np.ndarray (D, H, W) float32
            
            _logger.info(
                "Rasterized volume: shape=%s, min=%.4f, max=%.4f",
                vol_pred.shape, float(vol_pred.min()), float(vol_pred.max()),
            )
            
            # --- 4. Import metric utilities ---
            from .metric_3d_utils import (
                get_aabb_roi,
                segment_volume_with_roi,
                compute_all_metrics,
            )
            
            # Compute AABB ROI from GT label (replaces dilation)
            aabb_roi = get_aabb_roi(gt_label)
            _logger.info("AABB ROI voxels: %d", int(aabb_roi.sum()))
            
            # Build threshold list: absolute + percentile-derived
            thresholds: list[tuple[str, float]] = []
            
            # Absolute thresholds
            for thr in self.config.thresholds_absolute:
                key = f"thd-{thr:.4f}"
                thresholds.append((key, float(thr)))
            
            # Percentile thresholds
            if self.config.thresholds_percentile:
                # Compute percentiles only within the AABB ROI for relevance
                vol_roi = vol_pred[aabb_roi]
                if vol_roi.size > 0:
                    for pct in self.config.thresholds_percentile:
                        thr_val = float(np.percentile(vol_roi, pct * 100.0))
                        key = f"thd-{int(pct * 100):d}%"
                        thresholds.append((key, thr_val))
                        _logger.info(
                            "Percentile %.0f%% threshold = %.6f", pct * 100, thr_val,
                        )
                else:
                    _logger.warning("AABB ROI is empty; skipping percentile thresholds.")
            
            # Compute metrics for each threshold
            cfg = self.config
            result: dict[str, torch.Tensor] = {}
            
            for thr_key, thr_val in thresholds:
                pred_mask = segment_volume_with_roi(
                    volume=vol_pred,
                    threshold=thr_val,
                    aabb_roi=aabb_roi,
                    connectivity=cfg.connectivity,
                    closing_radius_vox=cfg.closing_radius_vox,
                    min_component_size_vox=cfg.min_component_size_vox,
                )
                
                metrics = compute_all_metrics(pred_mask, gt_label, gt_spacing)
                
                for metric_name, metric_val in metrics.items():
                    result[f"metric3D/{thr_key}/{metric_name}"] = torch.tensor(
                        metric_val, dtype=torch.float32, device=pl_module.device,
                    )
                
                _logger.info(
                    "3D metrics [%s]: dice=%.4f, hd95=%.2f, cldice=%.4f",
                    thr_key, metrics["dice"], metrics["hd95"], metrics["cldice"],
                )
            
            return result
        
        except Exception as e:
            _logger.exception("Failed to compute 3D metrics; returning empty dict. Error: %s", e)
            return {}