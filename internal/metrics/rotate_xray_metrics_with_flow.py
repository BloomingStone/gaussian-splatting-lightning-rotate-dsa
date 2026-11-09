from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any

import torch
from lightning import LightningModule

from internal.metrics.metric import Metric, MetricImpl, CommonImageMetricImpl
from internal.renderers.deformabel_xray_renderer import  XrayRendererOuputs
from internal.dataparsers.dataparser import BatchT


@dataclass
class RotateXrayMetrics(Metric):
    w_gray_loss: float = 1.0
    w_ssim_loss: float = 1.0
    
    w_d_density: float = 0.0
    w_xyz_var: float = 0.0

    # soft gate: confine flow (d_density) to high-density regions
    w_flow_gate: float = 0.0
    flow_gate_threshold: float | None = None   # None -> use flow_gate_quantile of base density
    flow_gate_quantile: float = 0.5
    flow_gate_sharpness: float = 10.0

    # field regularizers exposed by the deform model (BSplineDeformModel)
    w_manifold: float = 0.0
    w_field_smooth: float = 0.0

    rgb_diff_loss: Literal["l1", "l2"] = "l1"

    lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex"
    """
    the vanilla 3DGS uses 'vgg', but 'alex' is faster
    """

    fused_ssim: bool = False

    compute_3d_metrics: bool = True
    """Whether to compute 3D segmentation metrics at each validation epoch
    (expensive; disable to speed up validation)."""

    compute_artifact_metrics: bool = True
    """Whether to compute artifact-sensitive 2D metrics (spike_score, vessel
    SSIM, edge sharpness, ...) at each validation epoch. Requires the val
    dataset to expose 2D vessel labels (masks_paths)."""

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
        
        # regularization losses on deforms variance
        d_xyz_var = outputs.deforms_var["d_xyz"]
        d_density_std = outputs.deforms_var["d_density"].sqrt()
        d_density_mean = outputs.deforms_mean["d_density"].abs()
        d_xyz_var_avg = d_xyz_var.mean()
        
        # use mean + 2*std as the uncertainty-aware density, which is roughly the upper bound of density with 95% confidence if 
        # we assume the deforms follow a normal distribution. 
        # This is to avoid under-estimate the density for points with high uncertainty.
        d_density_upper= d_density_mean + 2 * d_density_std
        d_density_upper_avg = d_density_upper.mean()

        # soft gate: penalize |d_density| where the base density is low, so that
        # the learned flow is confined to high-density (vessel) regions.
        flow_gate_loss = torch.zeros((), device=gray_loss.device)
        d_density = getattr(outputs.deforms, "d_density", None)
        if self.config.w_flow_gate > 0 and d_density is not None:
            base_density = gaussian_model.get_density()
            threshold = self.config.flow_gate_threshold
            if threshold is None:
                threshold = torch.quantile(base_density.detach(), self.config.flow_gate_quantile)
            gate = torch.sigmoid(-self.config.flow_gate_sharpness * (base_density.detach() - threshold))
            flow_gate_loss = (d_density.abs() * gate).mean()

        # manifold / field-smoothness regularizers from the deform model
        reg = outputs.meta.get(XrayRendererOuputs.MetaTypes.REGULARIZATION) if outputs.meta is not None else None
        manifold_loss = reg["manifold"] if reg else torch.zeros((), device=gray_loss.device)
        field_smooth_loss = reg["smoothness"] if reg else torch.zeros((), device=gray_loss.device)

        loss = (
            # image loss
            self.config.w_gray_loss * gray_loss +
            self.config.w_ssim_loss * ssim_loss + 
            
            # regularization loss
            self.config.w_d_density * d_density_upper_avg + 
            self.config.w_xyz_var * d_xyz_var_avg +
            self.config.w_flow_gate * flow_gate_loss +
            self.config.w_manifold * manifold_loss +
            self.config.w_field_smooth * field_smooth_loss
        )


        # assert not torch.isnan(loss), "Loss is NaN!"
        
        metrics = {
            "loss": loss,
            "gray_loss": gray_loss,
            "ssim_loss": ssim_loss,
            "d_density_upper": d_density_upper_avg,
            "d_density_mean": d_density_mean.mean(),
            "d_density_std": d_density_std.mean(),
            "xyz_var": d_xyz_var_avg,
            "flow_gate_loss": flow_gate_loss,
            "manifold_loss": manifold_loss,
            "field_smooth_loss": field_smooth_loss,
        }


        prog_bar = {
            "loss": True,
            "gray_loss": True,
            "ssim_loss": True,
            "d_density_upper": True,
            "d_density_mean": False,
            "d_density_std": False,
            "xyz_var": True,
            "flow_gate_loss": True,
            "manifold_loss": False,
            "field_smooth_loss": False,
        }
        return metrics, prog_bar
    

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        return self._get_basic_metrics(
            pl_module=pl_module,    # type: ignore
            gaussian_model=gaussian_model,
            batch=batch,
            outputs=outputs     # type: ignore
        )
    
    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, prog_bar = self._get_basic_metrics(
            pl_module,
            gaussian_model,
            batch,
            outputs
        )

        _, image_info, extra_data = batch
        _, gt_image, mask = image_info
        gt_image = self._ensure_gray_nchw(gt_image)

        self.add_image_validation_metrics(metrics, prog_bar, outputs.gray_image, gt_image)

        weight = extra_data.get("soft_mask_weight") if extra_data is not None else None
        if weight is not None:
            self.add_weighted_validation_metrics(metrics, prog_bar, outputs.gray_image, gt_image, weight)

        if self.config.compute_artifact_metrics:
            vessel_mask = mask[0] if mask is not None else None  # (H, W) bool
            self._add_artifact_metrics(metrics, outputs.gray_image, gt_image, vessel_mask)

        metrics.update(self.metric3d)
        
        return metrics, prog_bar

    @torch.no_grad()
    def _add_artifact_metrics(
        self,
        metrics: Dict[str, Any],
        pred_gray: torch.Tensor,
        gt_gray: torch.Tensor,
        vessel_mask: torch.Tensor | None,
    ) -> None:
        """Artifact-sensitive metrics (numpy, same definitions as
        scripts/compute_artifact_metrics.py):

        * artifact/spike_score    : background high-frequency energy ratio
                                    (needle / spike artifacts)
        * artifact/v_psnr, v_ssim, v_l1 : vessel-region (2D label) metrics
                                    (misalignment / ghosting)
        * artifact/edge_sharpness : |grad(pred)| / |grad(gt)| on GT vessel edges
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter, sobel
        from skimage.metrics import structural_similarity

        device = pred_gray.device
        pred = np.squeeze(pred_gray.detach().cpu().float().numpy())
        gt = np.squeeze(gt_gray.detach().cpu().float().numpy())

        # --- spike score (background high-frequency energy ratio) ---
        hp_gt = gt - gaussian_filter(gt, sigma=1.0)
        hp_pred = pred - gaussian_filter(pred, sigma=1.0)
        if vessel_mask is not None:
            vessel = vessel_mask.detach().cpu().numpy().astype(bool)
            bg = ~vessel
            e_gt = float((hp_gt[bg] ** 2).mean())
            e_pred = float((hp_pred[bg] ** 2).mean())
        else:
            e_gt = float((hp_gt ** 2).mean())
            e_pred = float((hp_pred ** 2).mean())
        metrics["artifact/spike_score"] = torch.tensor(e_pred / (e_gt + 1e-8), device=device)

        if vessel_mask is None or vessel.sum() == 0:
            return

        # --- vessel-region metrics ---
        diff = (pred - gt) ** 2
        v_mse = float(diff[vessel].mean())
        metrics["artifact/v_psnr"] = torch.tensor(10.0 * np.log10(1.0 / (v_mse + 1e-8)), device=device)
        metrics["artifact/v_l1"] = torch.tensor(float(np.abs(pred - gt)[vessel].mean()), device=device)
        _, ssim_map = structural_similarity(gt, pred, data_range=1.0, full=True)
        metrics["artifact/v_ssim"] = torch.tensor(float(ssim_map[vessel].mean()), device=device)

        # --- edge sharpness ---
        g_gt = np.hypot(sobel(gt, axis=0), sobel(gt, axis=1))
        g_pred = np.hypot(sobel(pred, axis=0), sobel(pred, axis=1))
        thr = np.quantile(g_gt, 0.90)
        edge = (g_gt > thr) & vessel
        if edge.sum() > 0:
            metrics["artifact/edge_sharpness"] = torch.tensor(
                float(g_pred[edge].mean() / (g_gt[edge].mean() + 1e-8)), device=device
            )

        # --- ghosting / edge-widening score (normal-direction gradient profile
        #     second moment ratio: pred / gt) ---
        gx_gt = sobel(gt, axis=1)
        gy_gt = sobel(gt, axis=0)
        gmag_gt = np.hypot(gx_gt, gy_gt)
        gx_p = sobel(pred, axis=1)
        gy_p = sobel(pred, axis=0)
        gmag_p = np.hypot(gx_p, gy_p)

        thr = np.quantile(gmag_gt, 0.90)
        edge = (gmag_gt > thr) & vessel
        ys, xs = np.nonzero(edge)
        if len(xs) > 0:
            step = max(1, len(xs) // 800)
            H, W = gt.shape
            rs = np.arange(-10, 11)

            def profile_width(img_gmag):
                widths = []
                for k in range(0, len(xs), step):
                    y, x = ys[k], xs[k]
                    gx0, gy0 = gx_gt[y, x], gy_gt[y, x]
                    nm = np.hypot(gx0, gy0) + 1e-8
                    nx, ny = gx0 / nm, gy0 / nm
                    prof = []
                    for r in rs:
                        xx = int(round(x + nx * r))
                        yy = int(round(y + ny * r))
                        prof.append(img_gmag[yy, xx] if 0 <= yy < H and 0 <= xx < W else 0.0)
                    prof = np.asarray(prof)
                    s = prof.sum() + 1e-8
                    c = float((rs * prof).sum() / s)
                    widths.append(float(np.sqrt(((rs - c) ** 2 * prof).sum() / s)))
                return float(np.mean(widths)) if widths else 1.0

            w_gt = profile_width(gmag_gt)
            w_pred = profile_width(gmag_p)
            metrics["artifact/ghost_score"] = torch.tensor(w_pred / (w_gt + 1e-8), device=device)
    
    def on_validation_epoch_start(self, pl_module):
        if self.config.compute_3d_metrics:
            self.metric3d = self._compute_3d_metrics(pl_module=pl_module)
        else:
            self.metric3d = {}
        return super().on_validation_epoch_start(pl_module)