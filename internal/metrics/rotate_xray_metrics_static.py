from __future__ import annotations

from typing import cast

import torch
import numpy as np
from ..dataparsers.xray_dataparser.meta import XRayMeta
from .metric import metric3DConfig
from .rotate_xray_metrics import RotateXrayMetrics, RotateXrayMetricsImpl

class StaticXrayMetrics(RotateXrayMetrics):
    
    def instantiate(self, *args, **kwargs) -> StaticXrayMetricsImpl:
        return StaticXrayMetricsImpl(self)
    

class StaticXrayMetricsImpl(RotateXrayMetricsImpl):
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
            # --- 1. Get GT label & meta (lazy‑init computer on first call) ---
            if self._metric3d_computer is None:
                datamodule = pl_module.get_datamodule()
                meta = cast(XRayMeta, datamodule.dataparser_outputs.meta)
                if meta.label_3d_info is None:
                    return {}  # no GT label available, skip 3D metrics

                label_info = meta.label_3d_info
                gt_label_np = label_info.data              # (D, H, W) bool  (numpy)
                if gt_label_np is None:
                    return {}
                aabb_roi_np = label_info.aabb              # numpy bool mask
                coronary_affine = meta.centering_affine
                spacing = np.diag(coronary_affine)[:3]

                from .metric_3d_utils import SegmentationMetricsComputer
                self._metric3d_computer = SegmentationMetricsComputer(
                    gt=gt_label_np,
                    aabb_roi=aabb_roi_np,
                    spacing=tuple(spacing),
                )

            # --- Retrieve cached metadata ---
            datamodule = pl_module.get_datamodule()
            meta = cast(XRayMeta, datamodule.dataparser_outputs.meta)
            label_info = meta.label_3d_info
            assert label_info is not None
            coronary_affine = meta.centering_affine
            aabb_roi_np = label_info.aabb
            device = pl_module.device
            aabb_roi = torch.from_numpy(aabb_roi_np).to(device=device, dtype=torch.bool)
            volume_shape = tuple(int(x) for x in meta.volume_size)

            # --- 2. Deform gaussians to the requested cardiac phase ---
            gaussian_model = pl_module.gaussian_model
            means3D = gaussian_model.get_means().detach()
            scales = gaussian_model.get_scales().detach()
            rotation = gaussian_model.get_rotations().detach()
            density = gaussian_model.get_density().detach()

            from ..deform_models.deform_model import GSParam
            gsparam = GSParam(xyz=means3D, scaling=scales, rotation=rotation, density=density)

            # --- 3. Rasterize to CUDA volume ---
            from ..savers.x_ray_saver import gaussians_to_volume_by_Rasterizer

            with torch.no_grad():
                vol_pred = gaussians_to_volume_by_Rasterizer(
                    means3D=gsparam.xyz,
                    scales=gsparam.scaling,
                    rotation=gsparam.rotation,
                    density=gsparam.density,
                    shape=volume_shape,
                    affine=coronary_affine,
                    to_cpu=False,
                )
                assert isinstance(vol_pred, torch.Tensor)  # still on CUDA

            # --- 4. Segmentation & metrics for each threshold ---
            thresholds: list[tuple[str, float]] = []
            cfg = cast(metric3DConfig, self.config.metric3d_cfg)
            for thr in cfg.thresholds_absolute:
                thresholds.append((f"thd-{thr:.4f}", float(thr)))

            if cfg.thresholds_percentile:
                vol_roi = vol_pred[aabb_roi]               # GPU indexing
                if vol_roi.numel() > 0:
                    for pct in cfg.thresholds_percentile:
                        thr_val = float(torch.quantile(vol_roi.cpu(), pct))
                        thresholds.append((f"thd-{pct * 100:.2f}%", thr_val))

            result: dict[str, torch.Tensor] = {}
            for thr_key, thr_val in thresholds:
                pred = ((vol_pred > thr_val) & aabb_roi).to(device=pl_module.device, dtype=torch.bool)       # CUDA bool tensor
                
                metrics = self._metric3d_computer.compute(pred)                                             # type: ignore[union-attr]
                for metric_name, metric_val in metrics.items():
                    result[f"metric3D/{thr_key}/{metric_name}"] = torch.tensor(
                        metric_val, dtype=torch.float32, device=device,
                    )

            # --- 5. Density-based metrics (threshold-free) ---
            density_metrics = self._metric3d_computer.compute_density(vol_pred)                             # type: ignore[union-attr]
            for metric_name, metric_val in density_metrics.items():
                result[f"metric3D/density/{metric_name}"] = torch.tensor(
                    metric_val, dtype=torch.float32, device=device,
                )

            return result
        except Exception as e:
            import warnings
            warnings.warn(f"Error computing 3D metrics: {e}")
            print(f"Error computing 3D metrics: {e}")
            return {}