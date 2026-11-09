"""Analyse the stage1 deform field to locate the cause of needle/spike artifacts.

Hypothesis: the deform model's d_scaling compresses some Gaussians to a tiny
scale (d_scaling -> -1) at certain phases, producing needle-like projections.

Run in the GS-contrast-flow environment via Lightning (tcnn weights need the
renderer.initialize_from_checkpoint path).
"""
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from internal.cli import CLI
from internal.gaussian_splatting import GaussianSplatting
from internal.datamodule import DataModule


class AnalyzeDeform(Callback):
    def on_validation_start(self, trainer, pl_module):
        renderer = pl_module.renderer
        deform = renderer.deform_model
        gs = pl_module.gaussian_model
        xyz = gs.get_means().detach()
        scales = gs.get_scales().detach()
        density = gs.get_density().detach().squeeze(-1)  # (N,)
        dev = xyz.device

        print("\n=== stage1 deform-field analysis ===")
        print(f"n_points={xyz.shape[0]}")
        for ph in [0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0]:
            t = torch.zeros(xyz.shape[0], 1, device=dev) + 0.5
            p = torch.zeros(xyz.shape[0], 1, device=dev) + ph
            with torch.no_grad():
                deforms = deform(xyz.detach(), t, p)
            d_scaling = deforms.d_scaling
            new_scales = scales * (1 + d_scaling)
            new_smin = new_scales.min(dim=1).values

            d_min = d_scaling.min().item()
            d_max = d_scaling.max().item()
            ns_min = new_smin.min().item()
            ns_p1 = float(np.percentile(new_smin.cpu().numpy(), 1))
            # scale compressed >90% (d_scaling < -0.9) & density non-trivial
            n_compress = int((d_scaling.min(dim=1).values < -0.9).sum())
            n_tiny = int((new_smin < 0.003).sum())
            print(f"phase {ph:.3f}: d_scaling[{d_min:.3f},{d_max:.3f}] "
                  f"new_scale_min: min={ns_min:.4f} p1={ns_p1:.4f} "
                  f"d_scaling<-0.9:{n_compress} new_scale<0.003:{n_tiny}")

        # static scale distribution for reference
        smin = scales.min(dim=1).values
        print(f"\nstatic scale_min: min={smin.min():.4f} p1={float(np.percentile(smin.cpu().numpy(),1)):.4f}")
        print(f"static density: p99={float(np.percentile(density.cpu().numpy(),99)):.5f}")


class AnalyzeCLI(CLI):
    def before_validate(self):
        self.trainer.callbacks.append(AnalyzeDeform())


def main():
    AnalyzeCLI(
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
