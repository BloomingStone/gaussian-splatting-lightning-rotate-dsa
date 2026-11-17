from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any

import torch
import torch.nn.functional as F
from lightning import LightningModule

from internal.metrics.metric import Metric, MetricImpl, CommonImageMetricImpl
from internal.renderers.deformabel_xray_renderer import  XrayRendererOuputs
from internal.dataparsers.dataparser import BatchT


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """q: [..., 4] wxyz -> R: [..., 3, 3]"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    shape = q.shape[:-1] + (3, 3)
    R = torch.empty(shape, device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


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

    # Gaussian shape / compactness regularizers.
    #   w_aniso > 0  penalises needle-like gaussians (max/min scale > aniso_tau),
    #                the geometric source of the 3D "spikes".
    #   w_overlap > 0 penalises gaussian centres falling inside a neighbour's
    #                ellipsoid (soft cover), so points spread out instead of
    #                stacking.
    w_aniso: float = 0.0
    aniso_tau: float = 5.0
    w_overlap: float = 0.0
    overlap_sigma: float = 1.0
    overlap_k: int = 4
    overlap_sample: int = 1024
    overlap_temp: float = 0.2

    # Depth-concentration regularizers (against the depth ambiguity of the
    # transmission model).  Two complementary sources of depth spreading:
    #   w_depth_conc  penalises pairs of gaussians that project onto the same
    #                 pixel region (2D neighbours) but lie far apart in depth:
    #                 sum_{2D-near} d_i d_j (z_i - z_j)^2.
    #   w_depth_scale penalises a single gaussian whose ellipsoid extends too far
    #                 along the ray direction: relu(sigma_depth - tau), where
    #                 sigma_depth = ||S R^T v|| and v is the unit ray direction.
    w_depth_conc: float = 0.0
    depth_conc_k: int = 4
    depth_conc_sample: int = 1024
    w_depth_scale: float = 0.0
    depth_scale_tau: float = 2.0

    # Frangi (coronary-prior) weighting of the image losses.  When > 0, the L1
    # and SSIM losses are reweighted by the Frangi soft mask (vessel = 1,
    # Gaussian falloff into the background) so a simplified deform field must
    # still prioritise vessel reconstruction instead of averaging over the
    # (usually dominant) background.  Requires a dataset that exposes
    # `frangi_soft_mask` (FrangiImagesDataset).
    w_frangi: float = 0.0

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
        _, image_info, extra_data = batch   # load depth_map as extra_data in internal/dataparsers/rotated_xray_dataparser.py
        _, gt_image, _ = image_info
        gt_image = self._ensure_gray_nchw(gt_image)
        pred_gray = self._ensure_gray_nchw(outputs.gray_image)

        # Frangi (coronary-prior) weighting of the image losses: forces the
        # simplified deform field to prioritise vessel regions instead of
        # averaging over the dominant background.
        frangi_w = None
        if extra_data is not None:
            frangi_w = extra_data.get("frangi_soft_mask")
            if frangi_w is None:
                frangi_w = extra_data.get("soft_mask_weight")

        if frangi_w is not None and self.config.w_frangi > 0:
            w = frangi_w.to(dtype=pred_gray.dtype, device=pred_gray.device)
            if w.dim() == 2:
                w = w.unsqueeze(0).unsqueeze(0)
            elif w.dim() == 3:
                w = w.unsqueeze(0)
            # keep background at weight 1 and boost vessels to 1 + w_frangi,
            # so the background is still constrained while vessels dominate.
            w = 1.0 + self.config.w_frangi * w
            w_sum = w.sum().clamp_min(1e-8)
            # weighted L1
            gray_loss = (torch.abs(pred_gray - gt_image) * w).sum() / w_sum
            # weighted SSIM (per-pixel ssim map, weighted by the Frangi mask).
            # NOTE: compute the per-pixel map in fp32 *and* outside the autocast
            # context — under 16-mixed autocast the standard SSIM map overflows
            # (values > 1, gradients ~1e5x too large), which blows up the deform
            # field.
            from internal.metrics.ssim import ssim as _std_ssim
            with torch.autocast(device_type=pred_gray.device.type, enabled=False):
                ssim_map = _std_ssim(pred_gray.float(), gt_image.float(), size_average=False)  # (B, C, H, W)
            ssim_metric = (ssim_map * w).sum() / w_sum
            ssim_loss = 1.0 - ssim_metric
        else:
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

        # Gaussian shape / compactness regularizers (anti-spike / anti-overlap)
        device = gray_loss.device
        aniso_loss = torch.zeros((), device=device)
        if self.config.w_aniso > 0:
            gs_scales = gaussian_model.get_scales()  # [N, 3] activated
            s_max = gs_scales.max(dim=-1).values
            s_min = gs_scales.min(dim=-1).values
            aniso = s_max / (s_min + 1e-8)
            aniso_loss = F.relu(aniso - self.config.aniso_tau).mean()

        overlap_loss = torch.zeros((), device=device)
        if self.config.w_overlap > 0:
            gs_means = gaussian_model.get_means()      # [N, 3]
            gs_scales = gaussian_model.get_scales()    # [N, 3]
            gs_rot = gaussian_model.get_rotations()    # [N, 4] normalized quat
            n = gs_means.shape[0]
            m = min(self.config.overlap_sample, n)
            perm = torch.randperm(n, device=device)[:m]
            m_means = gs_means[perm]  # [m, 3]

            # euclidean squared distance to every gaussian, exclude self
            d2 = ((m_means[:, None, :] - gs_means[None, :, :]) ** 2).sum(-1)  # [m, n]
            d2.scatter_(1, perm[:, None], float("inf"))
            k = min(self.config.overlap_k, n - 1)
            _, nn_idx = d2.topk(k, dim=1, largest=False)  # [m, k]

            nn_scales = gs_scales[nn_idx]  # [m, k, 3]
            nn_rot = gs_rot[nn_idx]        # [m, k, 4]
            R = _quat_to_matrix(nn_rot)    # [m, k, 3, 3]
            diff = m_means[:, None, :] - gs_means[nn_idx]  # [m, k, 3]

            # Mahalanobis distance of each sampled centre into the neighbour's
            # ellipsoid: y = R_j^T (mu_i - mu_j), d2 = ||y / s_j||^2
            y = torch.einsum("mkd,mkde->mke", diff, R)  # [m, k, 3]
            d2_maha = ((y / (nn_scales + 1e-8)) ** 2).sum(-1)  # [m, k]

            # soft cover: sigmoid((sigma^2 - d2) / temp), high when centre is inside
            soft_cover = torch.sigmoid(
                (self.config.overlap_sigma ** 2 - d2_maha) / self.config.overlap_temp
            )
            overlap_loss = soft_cover.max(dim=-1).values.mean()

        # Depth-concentration regularizers: penalise density spreading along the
        # ray direction (the transmission model integrates depth away, so without
        # a prior the density can smear uniformly or concentrate at a point).
        depth_conc_loss = torch.zeros((), device=device)
        depth_scale_loss = torch.zeros((), device=device)
        if self.config.w_depth_conc > 0 or self.config.w_depth_scale > 0:
            gs_means = gaussian_model.get_means()          # [N, 3]
            gs_density = gaussian_model.get_density().squeeze(-1)  # [N]
            camera = batch[0]
            w2c = camera.world_to_camera                   # [4, 4] (transposed layout)
            campos = camera.camera_center                  # [3]

            # world -> camera coords. w2c is stored transposed:
            # [[R^T, 0], [T, 1]], so R_view = w2c[:3,:3].T, T_view = w2c[3,:3].
            R_view = w2c[:3, :3].T
            T_view = w2c[3, :3]
            p_cam = gs_means @ R_view.T + T_view           # [N, 3]

            if self.config.w_depth_scale > 0:
                # sigma_depth = || S R_g^T v || with v the unit ray direction
                v = gs_means - campos
                v = v / (v.norm(dim=-1, keepdim=True).clamp_min(1e-8))
                gs_scales = gaussian_model.get_scales()
                gs_rot = gaussian_model.get_rotations()
                Rg = _quat_to_matrix(gs_rot)               # [N, 3, 3] local -> world
                Rtv = torch.einsum("nd,nde->ne", v, Rg)    # [N, 3] = R_g^T v
                sigma_depth = (gs_scales * Rtv).norm(dim=-1)  # [N]
                depth_scale_loss = (
                    gs_density * F.relu(sigma_depth - self.config.depth_scale_tau)
                ).mean()

            if self.config.w_depth_conc > 0:
                n = gs_means.shape[0]
                m = min(self.config.depth_conc_sample, n)
                perm = torch.randperm(n, device=device)[:m]
                xy = p_cam[:, :2] / p_cam[:, 2:3].clamp_min(1e-6)  # [N, 2] normalised 2D
                z = p_cam[:, 2]                          # [N] depth
                xy_s = xy[perm]                          # [m, 2]
                d2_2d = ((xy_s[:, None, :] - xy[None, :, :]) ** 2).sum(-1)  # [m, n]
                d2_2d.scatter_(1, perm[:, None], float("inf"))
                k = min(self.config.depth_conc_k, n - 1)
                _, nn_idx = d2_2d.topk(k, dim=1, largest=False)  # [m, k]
                dz = z[perm][:, None] - z[nn_idx]       # [m, k] depth difference
                w_ij = gs_density[perm][:, None] * gs_density[nn_idx]  # [m, k] = d_i d_j
                depth_conc_loss = (w_ij * (dz ** 2)).mean()

        loss = (
            # image loss
            self.config.w_gray_loss * gray_loss +
            self.config.w_ssim_loss * ssim_loss + 
            
            # regularization loss
            self.config.w_d_density * d_density_upper_avg + 
            self.config.w_xyz_var * d_xyz_var_avg +
            self.config.w_flow_gate * flow_gate_loss +
            self.config.w_manifold * manifold_loss +
            self.config.w_field_smooth * field_smooth_loss +
            self.config.w_aniso * aniso_loss +
            self.config.w_overlap * overlap_loss +
            self.config.w_depth_conc * depth_conc_loss +
            self.config.w_depth_scale * depth_scale_loss
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
            "aniso_loss": aniso_loss,
            "overlap_loss": overlap_loss,
            "depth_conc_loss": depth_conc_loss,
            "depth_scale_loss": depth_scale_loss,
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
            "aniso_loss": False,
            "overlap_loss": False,
            "depth_conc_loss": False,
            "depth_scale_loss": False,
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

        # --- ghosting / double-edge score (fraction of GT vessel-edge pixels
        #     whose normal-direction intensity profile shows >=2 dark valleys) ---
        gx_gt = sobel(gt, axis=1)
        gy_gt = sobel(gt, axis=0)
        gmag_gt = np.hypot(gx_gt, gy_gt)
        thr = np.quantile(gmag_gt, 0.90)
        edge = (gmag_gt > thr) & vessel
        ys, xs = np.nonzero(edge)
        if len(xs) > 0:
            step = max(1, len(xs) // 800)
            H, W = gt.shape
            dbl = 0
            tot = 0
            for k in range(0, len(xs), step):
                y, x = ys[k], xs[k]
                gx0, gy0 = gx_gt[y, x], gy_gt[y, x]
                nm = np.hypot(gx0, gy0) + 1e-8
                nx, ny = gx0 / nm, gy0 / nm
                prof = []
                for r in range(-8, 9):
                    xx = int(round(x + nx * r))
                    yy = int(round(y + ny * r))
                    prof.append(pred[yy, xx] if 0 <= yy < H and 0 <= xx < W else 0.0)
                prof = np.asarray(prof)
                prof = gaussian_filter(prof, sigma=0.5)
                m = prof.mean()
                s = prof.std() + 1e-8
                valleys = sum(
                    1 for j in range(1, len(prof) - 1)
                    if prof[j] < prof[j - 1] and prof[j] <= prof[j + 1]
                    and prof[j] < m - 0.5 * s
                )
                if valleys >= 2:
                    dbl += 1
                tot += 1
            metrics["artifact/ghost_score"] = torch.tensor(dbl / tot, device=device)

            # --- edge contrast ratio (under-reconstruction blur) ---
            def _mean_range(img):
                ranges = []
                for k in range(0, len(xs), step):
                    y, x = ys[k], xs[k]
                    gx0, gy0 = gx_gt[y, x], gy_gt[y, x]
                    nm = np.hypot(gx0, gy0) + 1e-8
                    nx, ny = gx0 / nm, gy0 / nm
                    prof = []
                    for r in range(-6, 7):
                        xx = int(round(x + nx * r))
                        yy = int(round(y + ny * r))
                        prof.append(img[yy, xx] if 0 <= yy < H and 0 <= xx < W else 0.0)
                    ranges.append(float(np.max(prof) - np.min(prof)))
                return float(np.mean(ranges)) if ranges else 1.0

            metrics["artifact/edge_contrast"] = torch.tensor(
                _mean_range(pred) / (_mean_range(gt) + 1e-8), device=device
            )
    
    def on_validation_epoch_start(self, pl_module):
        if self.config.compute_3d_metrics:
            self.metric3d = self._compute_3d_metrics(pl_module=pl_module)
        else:
            self.metric3d = {}
        return super().on_validation_epoch_start(pl_module)