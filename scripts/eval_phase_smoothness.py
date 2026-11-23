"""评估心脏相位平滑性：相邻 phase 的位移是否相近、phase 0/1 是否一致。

加载训练好的形变模型，固定高斯位置 xyz 和 time，沿 phase ∈ [0,1] 扫描，
统计形变位移 d_xyz 随相位的变化：
  1. 相邻 phase 的位移变化（相邻相位位移应相近）
  2. phase 0 vs phase 1 的位移差异（循环相位边界应一致）
  3. 全相位范围内位移幅度（作为归一化参考）

用法：
    pixi run python scripts/eval_phase_smoothness.py validate \
        --config <config.yaml> --data.path data/example_grids \
        [extra model args...] --ckpt_path <ckpt.ckpt>
"""
import os
import sys

import torch
from lightning.pytorch.callbacks import Callback

from internal.cli import CLI
from internal.gaussian_splatting import GaussianSplatting
from internal.datamodule import DataModule


class PhaseSmoothnessEval(Callback):
    def __init__(self, n_phases: int = 41, time: float = 0.5):
        super().__init__()
        self.n_phases = n_phases
        self.time = time

    def on_validation_start(self, trainer, pl_module):
        gm = pl_module.gaussian_model
        deform_model = pl_module.renderer.deform_model
        device = pl_module.device
        xyz = gm.get_means().detach()
        n = xyz.shape[0]

        phases = torch.linspace(0.0, 1.0, self.n_phases, device=device)
        all_dxyz = []
        with torch.no_grad():
            for p in phases:
                deforms = deform_model(
                    xyz=xyz,
                    t=torch.full((n, 1), self.time, device=device),
                    phase=torch.full((n, 1), float(p), device=device),
                )
                all_dxyz.append(deforms.d_xyz.detach().float())
        dxyz = torch.stack(all_dxyz, dim=0)  # [P, N, 3] fp32

        dp = 1.0 / (self.n_phases - 1)
        adj = (dxyz[1:] - dxyz[:-1]).norm(dim=-1)                      # [P-1, N]
        boundary = (dxyz[0] - dxyz[-1]).norm(dim=-1)                   # [N]
        total_var = (dxyz.max(0).values - dxyz.min(0).values).norm(dim=-1)  # [N]
        ratio = adj.mean(dim=0) / (total_var + 1e-8)                   # [N]

        print(f"\n=== 相位平滑性评估 (n_gaussians={n}, phases={self.n_phases}, dp={dp:.4f}) ===")
        print(f"相邻 phase 位移变化 (L2, mm): "
              f"mean={adj.mean():.4f} median={adj.median():.4f} "
              f"p90={adj.quantile(0.9):.4f} max={adj.max():.4f}")
        print(f"phase 0 vs 1 位移差异  (L2, mm): "
              f"mean={boundary.mean():.4f} median={boundary.median():.4f} "
              f"p90={boundary.quantile(0.9):.4f} max={boundary.max():.4f}")
        print(f"全相位位移幅度        (L2, mm): "
              f"mean={total_var.mean():.4f} median={total_var.median():.4f} "
              f"p90={total_var.quantile(0.9):.4f}")
        print(f"相邻变化/总幅度 比率: "
              f"mean={ratio.mean():.4f} median={ratio.median():.4f} "
              f"p90={ratio.quantile(0.9):.4f}")

        sys.stdout.flush()
        os._exit(0)


class EvalPhaseCLI(CLI):
    def before_validate(self):
        self.trainer.callbacks.append(PhaseSmoothnessEval())
        print("[EvalPhaseCLI] evaluating phase smoothness")


def main():
    EvalPhaseCLI(
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
