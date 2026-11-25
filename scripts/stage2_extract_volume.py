"""在已训练的 stage2 模型上提取指定 (time_s, phase) 的 3D 密度体积。

stage2 的 density = base_density + d_density(x, t)。在 time_s=2s（归一化约 0.67）
造影剂完全充盈，d_density 补全了 stage1 静态重建遗漏的末梢分支。本脚本用
``DeformWithFlowModel.deform`` 把 d_density(t, phase) 加到 density 上，再光栅化
成 volume 保存 nii，供 ``scripts/compute_shape_metrics.py`` 计算 3D 指标
（与 stage1 口径完全一致）。

用法：
    EXTRACT_OUT=outputs/stage2_volume_extract/stage2-full \
    EXTRACT_TIME_S=2.0 EXTRACT_PHASE=0.0 \
    pixi run python scripts/stage2_extract_volume.py validate \
        --config configs/gen_4d_output_all/flow-mlp-stage2-flow.yaml \
        --data.path data/example_grids --ckpt_path <stage2.ckpt>

输出：<EXTRACT_OUT>/density_time{time_s}s_phase{phase}.nii.gz
"""
import os
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from internal.cli import CLI
from internal.gaussian_splatting import GaussianSplatting
from internal.datamodule import DataModule


class ExtractVolumeCallback(Callback):
    def __init__(self, out_dir: str, time_s: float = 2.0, phase: float = 0.0):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.time_s = time_s
        self.phase = phase

    def on_validation_start(self, trainer, pl_module):
        import nibabel as nib

        from internal.savers.x_ray_saver import gaussians_to_volume_by_Rasterizer
        from internal.deform_models.deform_model import GSParam

        pc = pl_module.gaussian_model
        renderer = pl_module.renderer
        deform_model = renderer.deform_model

        means3D = pc.get_means().detach()
        density = pc.get_density().detach()
        rotation = pc.get_rotations().detach()
        scales = pc.get_scales().detach()

        meta = pl_module.get_datamodule().dataparser.meta
        t_max = float(meta.time_array.max())
        t_norm = self.time_s / t_max
        print(f"[ExtractVolume] time_s={self.time_s} -> t_norm={t_norm:.4f} "
              f"(t_max={t_max:.4f})  phase={self.phase}")

        with torch.no_grad():
            deforms = deform_model(
                xyz=means3D,
                t=torch.full((means3D.shape[0], 1), t_norm, device=means3D.device),
                phase=torch.full((means3D.shape[0], 1), self.phase, device=means3D.device),
            )
            source_deformed = deform_model.deform(
                GSParam(xyz=means3D, rotation=rotation, scaling=scales, density=density),
                deforms,
            )
            # density 已含 d_density（DeformWithFlowModel.deform 会做 density + d_density）

        volume_shape = tuple(int(x) for x in meta.volume_size)
        coronary_affine = meta.centering_affine

        vol = gaussians_to_volume_by_Rasterizer(
            source_deformed.xyz,
            source_deformed.scaling,
            source_deformed.rotation,
            source_deformed.density,
            volume_shape,
            coronary_affine,
        )
        assert isinstance(vol, np.ndarray)
        torch.cuda.empty_cache()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.out_dir / f"density_time{self.time_s}s_phase{self.phase}.nii.gz"
        nib.save(nib.Nifti1Image(vol, affine=coronary_affine), out_path)
        print(f"[ExtractVolume] saved -> {out_path}  "
              f"shape={vol.shape}  max={vol.max():.4f}  mean={vol.mean():.6f}")

        # 关键 3D 分割/密度指标（与 compute_shape_metrics.py 同一套计算机，便于快速查看）
        try:
            from internal.metrics.metric_3d_utils import SegmentationMetricsComputer
            label_info = meta.label_3d_info
            if label_info is not None:
                gt_np = label_info.data.astype(np.uint8)
                # crop 到 GT 的 AABB（与 compute_shape_metrics 一致）
                aabb = label_info.aabb
                coords = np.argwhere(aabb)
                mn, mx = coords.min(axis=0), coords.max(axis=0)
                bbox = tuple(slice(mn[i], mx[i] + 1) for i in range(3))
                gt_c = np.ascontiguousarray(gt_np[bbox])
                spacing = tuple(np.abs(np.diag(np.array(coronary_affine))[:3]))
                computer = SegmentationMetricsComputer(
                    gt=gt_c, aabb_roi=np.ones_like(gt_c, dtype=bool), spacing=spacing,
                )
                pred_c = np.ascontiguousarray(((vol > 0.0344) & aabb)[bbox].astype(bool))
                seg = computer.compute(torch.from_numpy(pred_c))
                vol_c = np.ascontiguousarray(vol[bbox])
                dens = computer.compute_density(torch.from_numpy(vol_c))
                print("[ExtractVolume] dice=%.4f cldice=%.4f dist_avg_cl=%.4f "
                      "pr_auc=%.4f" % (
                        seg["dice"], seg["cldice"], seg["dist_avg_cl"], dens["pr_auc"]))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ExtractVolume] 3D metric preview failed: {e}")


class ExtractVolumeCLI(CLI):
    def before_validate(self):
        out_dir = os.environ.get("EXTRACT_OUT", "outputs/stage2_volume_extract/default")
        time_s = float(os.environ.get("EXTRACT_TIME_S", "2.0"))
        phase = float(os.environ.get("EXTRACT_PHASE", "0.0"))
        self.trainer.callbacks.append(
            ExtractVolumeCallback(out_dir, time_s=time_s, phase=phase)
        )
        print(f"[ExtractVolumeCLI] out={out_dir} time_s={time_s} phase={phase}")


def main():
    ExtractVolumeCLI(
        GaussianSplatting,
        DataModule,
        seed_everything_default=42,
        auto_configure_optimizers=False,
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": "auto",
            "devices": 1,
            "num_sanity_val_steps": 0,
            "max_steps": 30_000,
            "use_distributed_sampler": False,
            "enable_checkpointing": False,
        },
    )


if __name__ == "__main__":
    main()
