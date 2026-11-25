"""stage2 grid 评估：冠脉加权 + 造影剂流动保真度指标。

在 grid 多视角验证集上计算：
  - 全局: psnr / ssim
  - 冠脉加权: v_psnr / v_ssim / v_l1 / edge_sharpness / spike_score / ghost_score / edge_contrast
  - 方案B (无 flow): label 内 L2 = mean((pred-gt)^2) 在冠脉 2D label 内
  - 方案A (用 flow): 造影剂假阳性 / 漏检（flow 连续浓度做权重，gt 做同域基线）

flow 数据与 GT 同 frame/phase/视角/文件名对齐，位于 data/example_only_flow_grids/grid/...
（GT 在 data/example_grids/grid/...）。

用法：
    GRID_EVAL_OUT=/path GRID_EVAL_TIME_THRESHOLD=0.22 GRID_EVAL_SKIP_TRAIN_FRAMES=1 \
    pixi run python scripts/stage2_grid_eval.py validate \
        --config configs/gen_4d_output_all/flow-mlp-stage2-flow.yaml \
        --data.path data/example_grids --ckpt_path <ckpt>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from internal.cli import CLI
from internal.gaussian_splatting import GaussianSplatting
from internal.datamodule import DataModule
from internal.metrics.ssim import ssim as ssim_fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_artifact_metrics import compute_metrics as artifact_metrics


def psnr_np(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.float64) / 255.0


class Stage2GridEval(Callback):
    def __init__(self, out_dir: str, time_threshold: float | None = 0.22,
                 flow_data_dir: str = "example_only_flow_grids"):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.renders_dir = self.out_dir / "renders"
        self.gt_dir = self.out_dir / "gt"
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        self.gt_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []
        self.time_threshold = time_threshold
        self.flow_data_dir = flow_data_dir
        self.flow_paths: list[Path] = []

    def _to_np(self, t: torch.Tensor) -> np.ndarray:
        t = t.detach().cpu().float()
        while t.ndim > 2:
            t = t[0]
        return t.numpy()

    def on_validation_start(self, trainer, pl_module):
        # 整图 LPIPS（不叠加 mask），alex 与 vanilla_metrics 默认一致
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type="alex")

        # 建立 batch_idx -> flow 路径映射（与 GT 同 frame/phase/视角，仅换根目录）
        dm = pl_module.get_datamodule()
        val_set = dm.dataparser_outputs.val_set
        image_paths = list(val_set.image_paths)
        self.flow_paths = []
        for p in image_paths:
            sp = str(p).replace("/example_grids/", f"/{self.flow_data_dir}/")
            self.flow_paths.append(Path(sp))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        camera, image_info, extra = batch
        image_name, gt_image, mask = image_info

        pred_raw = outputs.gray_image.detach().cpu().float()
        gt_raw = gt_image.detach().cpu().float()
        while pred_raw.ndim > 2:
            pred_raw = pred_raw[0]
        while gt_raw.ndim > 2:
            gt_raw = gt_raw[0]

        pred = pred_raw.numpy()
        gt = gt_raw.numpy()
        pred = np.clip(pred, 0.0, 1.0)
        gt = np.clip(gt, 0.0, 1.0)

        p = psnr_np(pred, gt)

        # 整图 LPIPS（不叠加 mask）：灰度 -> 3 通道复制
        with torch.no_grad():
            lp = float(self.lpips(
                pred_raw.clamp(0.0, 1.0)[None, None].repeat(1, 3, 1, 1),
                gt_raw.clamp(0.0, 1.0)[None, None].repeat(1, 3, 1, 1),
            ).item())
        p_t = torch.from_numpy(pred.astype(np.float32))[None, None]
        g_t = torch.from_numpy(gt.astype(np.float32))[None, None]
        s = float(ssim_fn(p_t, g_t).item())

        t = 0.0
        cam_time = getattr(camera, "time", None)
        if cam_time is not None:
            cam_time = cam_time.detach().cpu().reshape(-1)
            if len(cam_time) > 0:
                t = float(cam_time[0].item())

        entry = {"idx": batch_idx, "psnr": p, "ssim": s, "lpips": lp, "time": t}

        vessel = None
        if mask is not None:
            vessel = self._to_np(mask) > 0.5
            # 冠脉加权 + artifact 指标
            if vessel.any():
                try:
                    am = artifact_metrics(gt, pred, vessel)
                    for k, v in am.items():
                        entry[k] = v
                except Exception:
                    import traceback
                    traceback.print_exc()
                # 方案B: label 内 L2
                entry["label_l2"] = float(((pred - gt)[vessel] ** 2).mean())

        # 方案A: flow 连续权重 假阳性/漏检
        if vessel is not None and vessel.any() and batch_idx < len(self.flow_paths):
            flow_path = self.flow_paths[batch_idx]
            if flow_path.exists():
                try:
                    flow = load_gray(flow_path)
                    if flow.shape == gt.shape:
                        fc = 1.0 - flow  # 0=无造影剂, 1=高浓度
                        fp = (np.clip(gt - pred, 0.0, None) * (1.0 - fc))[vessel].mean()
                        fn = (np.clip(pred - gt, 0.0, None) * fc)[vessel].mean()
                        entry["flow_fp"] = float(fp)
                        entry["flow_fn"] = float(fn)
                except Exception:
                    import traceback
                    traceback.print_exc()

        self.results.append(entry)

        Image.fromarray((pred * 255).astype(np.uint8)).save(self.renders_dir / f"val_{batch_idx:03d}.png")
        Image.fromarray((gt * 255).astype(np.uint8)).save(self.gt_dir / f"val_{batch_idx:03d}.png")

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
        with open(self.out_dir / "metrics.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[Stage2GridEval] {len(self.results)} images -> {self.out_dir / 'metrics.json'}")
        print(f"[Stage2GridEval] mean PSNR={mean.get('psnr', float('nan')):.4f} "
              f"SSIM={mean.get('ssim', float('nan')):.4f} "
              f"LPIPS={mean.get('lpips', float('nan')):.4f}")
        for k in ("v_psnr", "label_l2", "flow_fp", "flow_fn"):
            if k in mean:
                print(f"[Stage2GridEval] {k} = {mean[k]:.5f}")


class Stage2GridEvalCLI(CLI):
    def before_validate(self):
        out_dir = os.environ.get("GRID_EVAL_OUT", "outputs/stage2_grid_eval")
        thr = os.environ.get("GRID_EVAL_TIME_THRESHOLD")
        time_threshold = float(thr) if thr else None
        flow_dir = os.environ.get("FLOW_DATA_DIR", "example_only_flow_grids")
        self.trainer.callbacks.append(
            Stage2GridEval(out_dir, time_threshold=time_threshold, flow_data_dir=flow_dir)
        )
        print(f"[Stage2GridEvalCLI] saving grid renders to {out_dir}, flow_dir={flow_dir}")
        if time_threshold is not None:
            print(f"[Stage2GridEvalCLI] splitting at time < {time_threshold}")


def main():
    Stage2GridEvalCLI(
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
