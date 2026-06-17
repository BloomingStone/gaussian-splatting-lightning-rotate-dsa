from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any, cast

import torch
import numpy as np
from lightning import LightningModule

from internal.metrics.metric import Metric, MetricImpl, CommonImageMetricImpl
from internal.renderers.xray_4d_renderer import XrayRendererOuputs
from internal.models.xray_4d_gaussian import Xray4DGaussianModel
from internal.dataparsers.dataparser import BatchT
from ..dataparsers.xray_dataparser.meta import XRayMeta
from .metric import metric3DConfig

@dataclass
class Xray4DMetrics(Metric):
    w_gray_loss: float = 1.0
    w_ssim_loss: float = 1.0
    
    w_density_var: float = 0.0
    w_xyz_var: float = 0.0

    rgb_diff_loss: Literal["l1", "l2"] = "l1"

    lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex"
    """
    the vanilla 3DGS uses 'vgg', but 'alex' is faster
    """

    fused_ssim: bool = False

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return Xray4DMetricsImpl(self)


class Xray4DMetricsImpl(CommonImageMetricImpl):
    config:  Xray4DMetrics

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

        d_xyz_var = outputs.deforms_var["d_xyz"][outputs.mask].mean()
        
        gaussian_model = cast(Xray4DGaussianModel, gaussian_model)
        density_var_mean = gaussian_model.get_density_res_energy()[outputs.mask].mean()

        loss = (
            # image loss
            self.config.w_gray_loss * gray_loss +
            self.config.w_ssim_loss * ssim_loss + 
            
            # regularization loss
            self.config.w_density_var * density_var_mean +
            self.config.w_xyz_var * d_xyz_var
        )


        # assert not torch.isnan(loss), "Loss is NaN!"
        
        metrics = {
            "loss": loss,
            "gray_loss": gray_loss,
            "ssim_loss": ssim_loss,
            "density_var": density_var_mean,
            "xyz_var": d_xyz_var
        }


        prog_bar = {
            "loss": True,
            "gray_loss": True,
            "ssim_loss": True,
            "density_var": True,
            "xyz_var": True,
        }
        return metrics, prog_bar
    

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs: XrayRendererOuputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        return self._get_basic_metrics(
            pl_module=pl_module,    
            gaussian_model=gaussian_model,
            batch=batch,
            outputs=outputs
        )
    
    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs: XrayRendererOuputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, prog_bar = self._get_basic_metrics(
            pl_module,
            gaussian_model,
            batch,
            outputs
        )

        _, image_info, extra_data = batch
        _, gt_image, _ = image_info
        gt_image = self._ensure_gray_nchw(gt_image)

        self.add_image_validation_metrics(metrics, prog_bar, outputs.gray_image, gt_image)
        
        weight = extra_data.get("soft_mask_weight") if extra_data is not None else None
        if weight is not None:
            self.add_weighted_validation_metrics(metrics, prog_bar, outputs.gray_image, gt_image, weight)

        metrics.update(self.metric3d)
        
        return metrics, prog_bar
    
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
            gaussian_model = cast(Xray4DGaussianModel, pl_module.gaussian_model)
            means3D = gaussian_model.get_means().detach()
            scales = gaussian_model.get_scales().detach()
            rotation = gaussian_model.get_rotations().detach()
            density = gaussian_model.get_density(uniformed_time).detach()

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


    def on_validation_epoch_start(self, pl_module):
        self.metric3d = self._compute_3d_metrics(pl_module=pl_module)
        return super().on_validation_epoch_start(pl_module)