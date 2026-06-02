from dataclasses import dataclass
from typing import Dict, Literal, Any, Tuple

import torch
from pytorch_lightning import LightningModule

from internal.dataparsers.dataparser import BatchT
from .metric import Metric, MetricImpl, CommonImageMetricImpl
from ..renderers.deformabel_xray_renderer_coronary_props import XrayRendererOuputs


@dataclass
class RotateXrayMetricsWeightPatch(Metric):
    w_gray_loss: float = 1.0
    w_ssim_loss: float = 1.0
    w_patch_loss: float = 0.1

    rgb_diff_loss: Literal["l1", "l2"] = "l1"

    lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex"
    fused_ssim: bool = True

    # Patch sampling
    num_patches: int = 10
    patch_divisor: int = 8  # patch_side = image_min_side / patch_divisor
    weight_power: float = 2.0  # sharpen weight distribution before sampling

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return RotateXrayMetricsWeightPatchImpl(self)


class RotateXrayMetricsWeightPatchImpl(CommonImageMetricImpl):
    config: RotateXrayMetricsWeightPatch

    # ------------------------------------------------------------------
    # Patch sampling helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_patches(
        weight_map: torch.Tensor,
        patch_size: int,
        num_patches: int,
        weight_power: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cy, cx) centre coordinates sampled from *weight_map*.

        *weight_power* sharpens (>1) or flattens (<1) the distribution.
        """
        H, W = weight_map.shape
        half = patch_size // 2
        if H <= 2 * half or W <= 2 * half:
            return torch.empty(0, dtype=torch.long, device=weight_map.device), \
                   torch.empty(0, dtype=torch.long, device=weight_map.device)

        # valid centre region
        valid = weight_map[half: H - half, half: W - half]
        # sharpen / flatten the distribution
        valid = valid ** weight_power
        eps = valid.max() * 1e-6
        probs = (valid + eps) / (valid.sum() + eps * valid.numel())

        indices = torch.multinomial(probs.flatten(), num_patches, replacement=True)
        cy = indices // (W - 2 * half) + half
        cx = indices % (W - 2 * half) + half
        return cy, cx

    # ------------------------------------------------------------------
    # Core metric computation
    # ------------------------------------------------------------------
    def _weight_patch_metrics(
        self,
        batch: BatchT,
        outputs: XrayRendererOuputs,
    ) -> tuple[dict[str, torch.Tensor], dict[str, bool]]:
        _, image_info, extra_data = batch
        _, gt_image, mask = image_info

        gt_image = self._ensure_gray_nchw(gt_image)
        pred_gray = self._ensure_gray_nchw(outputs.gray_image)

        # ── whole-image basic loss ────────────────────────────────────
        gray_loss = self.rgb_diff_loss_fn(pred_gray, gt_image)
        ssim_val = self.ssim(pred_gray, gt_image)
        ssim_loss = 1.0 - ssim_val

        # ── weight-map guided patch losses ────────────────────────────
        patch_gray_loss = gt_image.new_zeros(())
        patch_ssim_loss = gt_image.new_zeros(())
        num_valid = 0

        if extra_data is not None and "weight_map" in extra_data:
            wmap = extra_data["weight_map"].to(device=pred_gray.device, dtype=pred_gray.dtype)
            patch_side = max(gt_image.shape[-1], gt_image.shape[-2]) // self.config.patch_divisor
            patch_side = max(patch_side, 7)  # keep SSIM window happy

            cy, cx = self._sample_patches(
                wmap, patch_side, self.config.num_patches,
                weight_power=self.config.weight_power,
            )
            if len(cy) > 0:
                half = patch_side // 2
                pg_list, ps_list = [], []
                for i in range(len(cy)):
                    y1, y2 = cy[i] - half, cy[i] + half
                    x1, x2 = cx[i] - half, cx[i] + half
                    p_pred = pred_gray[..., y1:y2, x1:x2]
                    p_gt = gt_image[..., y1:y2, x1:x2]

                    pg = self.rgb_diff_loss_fn(p_pred, p_gt)
                    ps = 1.0 - self.ssim(p_pred, p_gt)
                    pg_list.append(pg)
                    ps_list.append(ps)

                patch_gray_loss = torch.stack(pg_list).mean()
                patch_ssim_loss = torch.stack(ps_list).mean()
                num_valid = len(cy)

        # ── combine ───────────────────────────────────────────────────
        loss = (
            self.config.w_gray_loss * gray_loss
            + self.config.w_ssim_loss * ssim_loss
            + self.config.w_patch_loss * (
                self.config.w_gray_loss * patch_gray_loss
                + self.config.w_ssim_loss * patch_ssim_loss
            )
        )

        assert not torch.isnan(loss), "Loss is NaN!"

        metrics: dict[str, torch.Tensor] = {
            "loss": loss,
            "gray_loss": gray_loss,
            "ssim_loss": ssim_loss,
            "patch_gray_loss": patch_gray_loss,
            "patch_ssim_loss": patch_ssim_loss,
            "num_patches": gt_image.new_tensor(num_valid, dtype=torch.float32),
        }
        prog_bar: dict[str, bool] = {
            "loss": True,
            "gray_loss": True,
            "ssim_loss": True,
            "patch_gray_loss": True,
            "patch_ssim_loss": True,
            "num_patches": False,
        }
        return metrics, prog_bar

    # ------------------------------------------------------------------
    # Lightning entry points
    # ------------------------------------------------------------------
    def get_train_metrics(
        self,
        pl_module: LightningModule,
        gaussian_model,
        step: int,
        batch: BatchT,
        outputs: XrayRendererOuputs,
    ) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        del pl_module, gaussian_model, step
        return self._weight_patch_metrics(batch, outputs)

    def get_validate_metrics(
        self,
        pl_module: LightningModule,
        gaussian_model,
        batch: BatchT,
        outputs: XrayRendererOuputs,
    ) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        del pl_module, gaussian_model
        metrics, prog_bar = self._weight_patch_metrics(batch, outputs)

        _, image_info, _ = batch
        _, gt_image, _ = image_info
        gt_image = self._ensure_gray_nchw(gt_image)

        self.add_image_validation_metrics(metrics, prog_bar, outputs.gray_image, gt_image)
        return metrics, prog_bar
