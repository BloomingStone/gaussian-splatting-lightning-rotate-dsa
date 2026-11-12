"""批量运行 bspline 控制点 sweep + frangi 加权重建实验。

每个实验：训练 → 验证渲染 → 计算伪影指标 → frame-100 对比图。

用法（可并行启动多个进程，分别指定 --gpu 和 --jobs）：
    python scripts/run_bspline_sweep.py --gpu 2 --jobs frangi-only bs16 bs32
    python scripts/run_bspline_sweep.py --gpu 3 --jobs bs64 bs128 bs16-f bs32-f
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/media/F/sj/Code/GS-contrast-flow")
CONFIG_DIR = ROOT / "configs/dev_grid_views/bspline-frangi"
FIG_DIR = ROOT / "dev_grid_views/bspline-frangi"

FRANGI_ONLY_CFG = CONFIG_DIR / "flow-mlp-stage1-frangi-only.yaml"
BSPLINE_CFG = CONFIG_DIR / "flow-mlp-stage1-bspline-sweep.yaml"

# job name -> (config, extra CLI args)
def _bs(res: int, frangi: bool):
    extra = [
        "--model.renderer.init_args.deform_model_config.init_args.grid_resolution",
        f"[{res},{res},{res}]",
    ]
    if frangi:
        extra += ["--model.metric.init_args.w_frangi", "1.0"]
    return BSPLINE_CFG, extra

JOBS = {}
JOBS["frangi-only"] = (FRANGI_ONLY_CFG, [])
for res in (16, 32, 64, 128):
    JOBS[f"bs{res}"] = _bs(res, False)
    JOBS[f"bs{res}-f"] = _bs(res, True)


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


def train_and_eval(gpu: int, job: str):
    config, extra = JOBS[job]
    name = f"bspline-frangi/{job}"
    out_dir = ROOT / "outputs" / "dev_grid_views" / name
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}

    # ---- train ----
    cmd = ["pixi", "run", "gs-fit", "--config", str(config),
           "--data.path", "data/example_grids", "-n", f"dev_grid_views/{name}"]
    if (out_dir / "checkpoints").exists():
        cmd.append("--overwrite_output")
    cmd += extra
    if not run(cmd, env):
        return

    # ---- find ckpt ----
    ckpt_dir = out_dir / "checkpoints"
    ckpt = ckpt_dir / "epoch=143-step=20000.ckpt"
    if not ckpt.exists():
        ckpts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
        if not ckpts:
            print("!! 无 ckpt", flush=True)
            return
        ckpt = ckpts[-1]

    # ---- validate render ----
    cmd = ["pixi", "run", "python", "scripts/stage1_grid_eval.py", "validate",
           "--config", str(config), "--data.path", "data/example_grids",
           "--ckpt_path", str(ckpt)]
    cmd += extra
    env["GRID_EVAL_OUT"] = str(out_dir)
    if not run(cmd, env):
        return

    # ---- artifact metrics ----
    cmd = ["pixi", "run", "python", "scripts/compute_artifact_metrics.py",
           "--stage1_dir", str(out_dir),
           "--r2g_dir", str(ROOT.parent / "r2_gaussian/output/dev_grid_views/r2g-blur"),
           "--out_dir", str(out_dir)]
    if not run(cmd):
        return

    # ---- frame-100 figure ----
    cmd = ["pixi", "run", "python", "scripts/make_frame100_stage1.py",
           "--stage1_dir", str(out_dir),
           "--out_dir", str(FIG_DIR),
           "--exp_name", job]
    run(cmd)

    print(f"==== {job} 完成 ====", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--jobs", nargs="+", required=True)
    args = p.parse_args()

    for job in args.jobs:
        if job not in JOBS:
            print(f"!! 未知 job: {job}，可选: {list(JOBS)}", flush=True)
            sys.exit(1)

    for job in args.jobs:
        print(f"\n========== {job} (gpu {args.gpu}) ==========", flush=True)
        train_and_eval(args.gpu, job)


if __name__ == "__main__":
    main()
