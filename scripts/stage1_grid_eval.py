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

# reuse the artifact-sensitive / vessel-weighted metric definitions
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_artifact_metrics import compute_metrics as artifact_metrics


def psnr_np(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


class GridRenderSaver(Callback):
    def __init__(self, out_dir: str, train_frames=(40, 100, 160), train_start_frame: int = 40,
                 time_threshold: float | None = None):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.renders_dir = self.out_dir / "renders"
        self.gt_dir = self.out_dir / "gt"
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        self.gt_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []
        self.train_frames = train_frames
        self.train_start_frame = train_start_frame
        self.time_threshold = time_threshold

    def _to_np(self, t: torch.Tensor) -> np.ndarray:
        t = t.detach().cpu().float()
        while t.ndim > 2:
            t = t[0]
        return t.numpy()

    def on_validation_start(self, trainer, pl_module):
        """Render the training-set frames with the same phase as frame 100
        (phase 0.667): frames 40 / 100 / 160, all within the training set."""
        if self.train_frames is None:
            return
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
        image_name, gt_image, mask = image_info
        pred = self._to_np(outputs.gray_image)
        gt = self._to_np(gt_image)

        pred = np.clip(pred, 0.0, 1.0)
        gt = np.clip(gt, 0.0, 1.0)

        p = psnr_np(pred, gt)
        p_t = torch.from_numpy(pred.astype(np.float32))[None, None]
        g_t = torch.from_numpy(gt.astype(np.float32))[None, None]
        s = float(ssim_fn(p_t, g_t).item())

        # record the normalized time so we can split metrics into
        # early frames (< time_threshold) vs all frames.
        t = 0.0
        cam_time = getattr(camera, "time", None)
        if cam_time is not None:
            cam_time = cam_time.detach().cpu().reshape(-1)
            if len(cam_time) > 0:
                t = float(cam_time[0].item())

        entry = {"idx": batch_idx, "psnr": p, "ssim": s, "time": t}

        # vessel-weighted + artifact metrics (using the 2D coronary label mask)
        if mask is not None:
            mask_np = self._to_np(mask) > 0.5  # (H, W) bool vessel mask
            if mask_np.any():
                try:
                    am = artifact_metrics(gt, pred, mask_np)
                    for k, v in am.items():
                        entry[k] = v  # v_psnr/v_ssim/v_l1/edge_sharpness/spike_score/ghost_score/edge_contrast
                except Exception:
                    import traceback
                    traceback.print_exc()

        self.results.append(entry)

        Image.fromarray((pred * 255).astype(np.uint8)).save(
            self.renders_dir / f"val_{batch_idx:03d}.png")
        Image.fromarray((gt * 255).astype(np.uint8)).save(
            self.gt_dir / f"val_{batch_idx:03d}.png")

    def on_validation_epoch_end(self, trainer, pl_module):
        if len(self.results) == 0:
            return
        def _mean(rs):
            out = {"n_images": len(rs)}
            if not rs:
                return out
            for key in rs[0]:
                if key in ("idx", "time"):
                    continue
                vals = [r[key] for r in rs
                        if key in r and isinstance(r[key], (int, float)) and np.isfinite(r[key])]
                if vals:
                    out[key] = float(np.mean(vals))
            return out
        mean = _mean(self.results)
        payload = {"mean": mean, "per_image": self.results}
        if self.time_threshold is not None:
            early = [r for r in self.results if r.get("time", 0.0) < self.time_threshold]
            late = [r for r in self.results if r.get("time", 0.0) >= self.time_threshold]
            payload["mean_early"] = _mean(early) if early else None
            payload["mean_late"] = _mean(late) if late else None
            print(f"[GridRenderSaver] early(<t={self.time_threshold}) PSNR = "
                  f"{payload['mean_early']['psnr']:.4f} SSIM = {payload['mean_early']['ssim']:.4f} "
                  f"({payload['mean_early']['n_images']} img)" if early else "[GridRenderSaver] no early images")
            print(f"[GridRenderSaver] late(>=t={self.time_threshold}) PSNR = "
                  f"{payload['mean_late']['psnr']:.4f} SSIM = {payload['mean_late']['ssim']:.4f} "
                  f"({payload['mean_late']['n_images']} img)" if late else "[GridRenderSaver] no late images")
        with open(self.out_dir / "metrics.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[GridRenderSaver] {len(self.results)} images -> {self.out_dir / 'metrics.json'}")
        print(f"[GridRenderSaver] mean PSNR = {mean['psnr']:.4f}  mean SSIM = {mean['ssim']:.4f}")


class GridEvalCLI(CLI):
    def before_validate(self):
        out_dir = os.environ.get("GRID_EVAL_OUT", "outputs/stage1_grid_eval")
        thr = os.environ.get("GRID_EVAL_TIME_THRESHOLD")
        time_threshold = float(thr) if thr else None
        skip_train = os.environ.get("GRID_EVAL_SKIP_TRAIN_FRAMES", "0") == "1"
        train_frames = None if skip_train else (40, 100, 160)
        self.trainer.callbacks.append(GridRenderSaver(out_dir, train_frames=train_frames, time_threshold=time_threshold))
        print(f"[GridEvalCLI] saving grid renders to {out_dir}")
        if time_threshold is not None:
            print(f"[GridEvalCLI] splitting metrics at normalized time < {time_threshold}")


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
