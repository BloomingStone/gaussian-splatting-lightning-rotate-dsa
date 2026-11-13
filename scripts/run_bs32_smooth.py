"""bs32 + 平滑正则(w_field_smooth) 消融：验证"高频自由度拟合噪声"假设。

用法（并行）：
    python scripts/run_bs32_smooth.py --gpu 0 --weights 0.005 0.02
    python scripts/run_bs32_smooth.py --gpu 2 --weights 0.1
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/media/F/sj/Code/GS-contrast-flow")
CONFIG = ROOT / "configs/dev_grid_views/bspline-frangi/flow-mlp-stage1-bspline-sweep.yaml"
FIG_DIR = ROOT / "dev_grid_views/bspline-frangi"
R2G_DIR = ROOT.parent / "r2_gaussian/output/lca_grid_eval"


def run(cmd, env_extra=None):
    print(">> " + " ".join(str(c) for c in cmd), flush=True)
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        print(f"!! 失败 returncode={r.returncode}", flush=True)
        return False
    return True


def train_and_eval(gpu: int, w: float, frangi: bool = False, res: int = 32):
    suffix = "-frangi" if frangi else ""
    name = f"bs{res}-smooth-{w}{suffix}"
    out_dir = ROOT / "outputs" / "dev_grid_views" / "bspline-frangi" / name
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    extra = [
        "--model.renderer.init_args.deform_model_config.init_args.grid_resolution",
        f"[{res},{res},{res}]",
        "--model.metric.init_args.w_field_smooth",
        str(w),
    ]
    if frangi:
        extra += ["--model.metric.init_args.w_frangi", "1.0"]

    # train
    cmd = ["pixi", "run", "gs-fit", "--config", str(CONFIG),
           "--data.path", "data/example_grids",
           "-n", f"dev_grid_views/bspline-frangi/{name}",
           "--overwrite_output"] + extra
    if not run(cmd, env):
        return

    ckpt = out_dir / "checkpoints" / "epoch=143-step=20000.ckpt"
    if not ckpt.exists():
        ckpts = sorted((out_dir / "checkpoints").glob("*.ckpt"))
        if not ckpts:
            print("!! 无 ckpt", flush=True)
            return
        ckpt = ckpts[-1]

    # validate render
    cmd = ["pixi", "run", "python", "scripts/stage1_grid_eval.py", "validate",
           "--config", str(CONFIG), "--data.path", "data/example_grids",
           "--ckpt_path", str(ckpt)] + extra
    env["GRID_EVAL_OUT"] = str(out_dir)
    if not run(cmd, env):
        return

    # artifact metrics
    cmd = ["pixi", "run", "python", "scripts/compute_artifact_metrics.py",
           "--stage1_dir", str(out_dir),
           "--r2g_dir", str(R2G_DIR),
           "--out_dir", str(out_dir)]
    if not run(cmd):
        return

    # frame-100 figure
    run(["pixi", "run", "python", "scripts/make_frame100_stage1.py",
         "--stage1_dir", str(out_dir),
         "--out_dir", str(FIG_DIR),
         "--exp_name", name])

    print(f"==== {name} 完成 ====", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--weights", nargs="+", type=float, required=True)
    p.add_argument("--frangi", action="store_true", help="同时加 w_frangi=1.0")
    p.add_argument("--res", type=int, default=32)
    args = p.parse_args()
    for w in args.weights:
        print(f"\n========== bs{args.res} w_field_smooth={w} frangi={args.frangi} (gpu {args.gpu}) ==========", flush=True)
        train_and_eval(args.gpu, w, frangi=args.frangi, res=args.res)


if __name__ == "__main__":
    main()
