"""导出形变后的密度体积(NIfTI)并计算 3D 分割指标（Dice/HD95/clDice/roc_auc...）。

复用 metric._compute_3d_metrics 的指标逻辑，并额外把 deform 到指定 phase 的
密度体积栅格化后保存为 NIfTI。

用法（从 GS-contrast-flow 根目录）：
    VOL3D_OUT=/path/to/out \
    pixi run python scripts/export_volume_3d_metrics.py validate \
        --config <config.yaml> --data.path data/example_grids \
        [extra model args...] \
        --ckpt_path <ckpt.ckpt>

环境变量：
    VOL3D_OUT      输出目录（默认 outputs/vol3d_export）
    VOL3D_PHASE    形变目标 cardiac phase（默认 0.0）
    VOL3D_TIME     形变目标 time（默认 0.5）
"""
import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from lightning.pytorch.callbacks import Callback

from internal.cli import CLI
from internal.gaussian_splatting import GaussianSplatting
from internal.datamodule import DataModule
from internal.deform_models.deform_model import GSParam
from internal.savers.x_ray_saver import gaussians_to_volume_by_Rasterizer


class Volume3DExporter(Callback):
    def __init__(self, out_dir: str, phase: float, time: float):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.phase = phase
        self.time = time

    def on_validation_start(self, trainer, pl_module):
        # ---- 3D metrics (reuse existing implementation) ----
        try:
            metrics = pl_module.metric._compute_3d_metrics(
                pl_module,
                uniformed_time=self.time,
                cardiac_phase=self.phase,
            )
            metrics_plain = {k: float(v) for k, v in metrics.items()}
            print("\n[Volume3DExporter] 3D metrics:")
            for k, v in metrics_plain.items():
                print(f"  {k}: {v:.4f}")
            with open(self.out_dir / "metrics3d.json", "w") as f:
                json.dump(metrics_plain, f, indent=2)
        except Exception:
            print("[Volume3DExporter] 3D metrics failed:")
            traceback.print_exc()

        # ---- export volume ----
        try:
            self._export_volume(pl_module)
        except Exception:
            print("[Volume3DExporter] volume export failed:")
            traceback.print_exc()

        # We only need the 3D export — skip the 90-frame render pass entirely.
        # It wastes ~50s/trial and raises peak VRAM (renders peaked higher than
        # the volume export itself).  Hard-exit after flushing so logs survive.
        if torch.cuda.is_available():
            print(
                f"[Volume3DExporter] done, peak VRAM = "
                f"{torch.cuda.max_memory_allocated()/1024**2:.1f} MB",
                flush=True,
            )
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    def _export_volume(self, pl_module):
        datamodule = pl_module.get_datamodule()
        meta = datamodule.dataparser_outputs.meta
        label_info = meta.label_3d_info
        assert label_info is not None, "no 3D label available"
        coronary_affine = meta.centering_affine
        volume_shape = tuple(int(x) for x in meta.volume_size)
        device = pl_module.device

        gaussian_model = pl_module.gaussian_model
        means3D = gaussian_model.get_means().detach()
        scales = gaussian_model.get_scales().detach()
        rotation = gaussian_model.get_rotations().detach()
        density = gaussian_model.get_density().detach()

        renderer = pl_module.renderer
        deform_model = renderer.deform_model
        n = means3D.shape[0]
        with torch.no_grad():
            deforms = deform_model(
                xyz=means3D,
                t=torch.full((n, 1), self.time, device=device),
                phase=torch.full((n, 1), self.phase, device=device),
            )
            new_gsparam = deform_model.deform(
                GSParam(xyz=means3D, scaling=scales, rotation=rotation, density=density),
                deforms,
            )

        with torch.no_grad():
            vol = gaussians_to_volume_by_Rasterizer(
                means3D=new_gsparam.xyz,
                scales=new_gsparam.scaling,
                rotation=new_gsparam.rotation,
                density=new_gsparam.density,
                shape=volume_shape,
                affine=coronary_affine,
                to_cpu=True,
            )
        vol_np = vol.cpu().numpy() if isinstance(vol, torch.Tensor) else np.asarray(vol)
        vol_np = vol_np.astype(np.float32)
        out_path = self.out_dir / f"density_phase{self.phase:.3f}.nii.gz"
        nib.save(Nifti1Image(vol_np, affine=coronary_affine), out_path)
        print(f"[Volume3DExporter] saved volume -> {out_path} "
              f"(shape {vol_np.shape}, min {vol_np.min():.5f}, max {vol_np.max():.5f})")

        # 预测冠脉分割标签（density 阈值化，affine 与 GT label 一致，可直接对比）
        thr = float(os.environ.get("VOL3D_THRESHOLD", "0.0344"))
        pred_label = (vol_np > thr).astype(np.uint8)
        label_path = self.out_dir / f"coronary_pred_phase{self.phase:.3f}.nii.gz"
        nib.save(Nifti1Image(pred_label, affine=coronary_affine), label_path)
        print(f"[Volume3DExporter] saved predicted coronary label -> {label_path} "
              f"(threshold {thr}, voxels {int(pred_label.sum())})")


class Export3DCLI(CLI):
    def before_validate(self):
        out_dir = os.environ.get("VOL3D_OUT", "outputs/vol3d_export")
        phase = float(os.environ.get("VOL3D_PHASE", "0.0"))
        time = float(os.environ.get("VOL3D_TIME", "0.5"))
        self.trainer.callbacks.append(Volume3DExporter(out_dir, phase, time))
        print(f"[Export3DCLI] exporting volume + 3D metrics to {out_dir}")


def main():
    Export3DCLI(
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
