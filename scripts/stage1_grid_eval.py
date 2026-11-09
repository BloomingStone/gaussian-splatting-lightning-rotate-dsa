"""Re-run stage1 validation on the `grid/` multi-view set while dumping every
rendered image (and its GT) to disk and computing PSNR / SSIM with the same
definitions used by the r2_gaussian grid eval (J-domain, standard 3DGS SSIM).

Usage (from GS-contrast-flow root, pixi env):
    GRID_EVAL_OUT=/path/to/out \
    pixi run python scripts/stage1_grid_eval.py validate \
        --config configs/gen_4d_output_all/flow-mlp-stage1-gridval.yaml \
        --ckpt_path outputs/stage1_examplegrids_gridval/checkpoints/epoch=143-step=20000.ckpt

The callback saves images named by `batch_idx` (unique per grid image), so the
`image_name` collision that normally overwrites saved val images is avoided.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from PIL import Image

from internal.cli import CLI
from internal.gaussian_splatting import GaussianSplatting
from internal.datamodule import DataModule
from internal.metrics.ssim import ssim as ssim_fn


def psnr_np(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


class GridRenderSaver(Callback):
    def __init__(self, out_dir: str, train_frames=(40, 100, 160), train_start_frame: int = 40):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.renders_dir = self.out_dir / "renders"
        self.gt_dir = self.out_dir / "gt"
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        self.gt_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []
        self.train_frames = train_frames
        self.train_start_frame = train_start_frame

    def _to_np(self, t: torch.Tensor) -> np.ndarray:
        t = t.detach().cpu().float()
        while t.ndim > 2:
            t = t[0]
        return t.numpy()

    def on_validation_start(self, trainer, pl_module):
        """Render the training-set frames with the same phase as frame 100
        (phase 0.667): frames 40 / 100 / 160, all within the training set."""
        try:
            dm = pl_module.get_datamodule()
            train_set = dm.dataparser_outputs.train_set
            for train_frame in self.train_frames:
                pos = train_frame - self.train_start_frame
                item = train_set[pos]
                camera = item.camera.to_device(pl_module.device)
                gt = item.image.gt_image  # (3, H, W) float [0, 1]

                with torch.no_grad():
                    outputs = pl_module(camera)
                pred = self._to_np(outputs.gray_image)
                gt_np = self._to_np(gt)  # (3, H, W) -> (H, W)

                pred = np.clip(pred, 0.0, 1.0)
                gt_np = np.clip(gt_np, 0.0, 1.0)

                p = psnr_np(pred, gt_np)
                train_dir = self.out_dir / "train_frame"
                train_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray((pred * 255).astype(np.uint8)).save(train_dir / f"stage1_frame{train_frame}.png")
                Image.fromarray((gt_np * 255).astype(np.uint8)).save(train_dir / f"gt_frame{train_frame}.png")
                with open(train_dir / f"stage1_frame{train_frame}_psnr.json", "w") as f:
                    json.dump({"frame": train_frame, "psnr": p}, f)
                print(f"[GridRenderSaver] train frame {train_frame} rendered, PSNR = {p:.4f}")
        except Exception as e:
            import traceback
            traceback.print_exc()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        camera, image_info, extra = batch
        image_name, gt_image, _ = image_info
        pred = self._to_np(outputs.gray_image)
        gt = self._to_np(gt_image)

        pred = np.clip(pred, 0.0, 1.0)
        gt = np.clip(gt, 0.0, 1.0)

        p = psnr_np(pred, gt)
        p_t = torch.from_numpy(pred.astype(np.float32))[None, None]
        g_t = torch.from_numpy(gt.astype(np.float32))[None, None]
        s = float(ssim_fn(p_t, g_t).item())

        self.results.append({"idx": batch_idx, "psnr": p, "ssim": s})

        Image.fromarray((pred * 255).astype(np.uint8)).save(
            self.renders_dir / f"val_{batch_idx:03d}.png")
        Image.fromarray((gt * 255).astype(np.uint8)).save(
            self.gt_dir / f"val_{batch_idx:03d}.png")

    def on_validation_epoch_end(self, trainer, pl_module):
        if len(self.results) == 0:
            return
        mean = {
            "psnr": float(np.mean([r["psnr"] for r in self.results if np.isfinite(r["psnr"])])),
            "ssim": float(np.mean([r["ssim"] for r in self.results])),
            "n_images": len(self.results),
        }
        with open(self.out_dir / "metrics.json", "w") as f:
            json.dump({"mean": mean, "per_image": self.results}, f, indent=2)
        print(f"\n[GridRenderSaver] {len(self.results)} images -> {self.out_dir / 'metrics.json'}")
        print(f"[GridRenderSaver] mean PSNR = {mean['psnr']:.4f}  mean SSIM = {mean['ssim']:.4f}")


class GridEvalCLI(CLI):
    def before_validate(self):
        out_dir = os.environ.get("GRID_EVAL_OUT", "outputs/stage1_grid_eval")
        self.trainer.callbacks.append(GridRenderSaver(out_dir))
        print(f"[GridEvalCLI] saving grid renders to {out_dir}")


def main():
    GridEvalCLI(
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
