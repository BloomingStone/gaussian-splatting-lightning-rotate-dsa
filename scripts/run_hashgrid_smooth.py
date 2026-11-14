"""hashgrid + 空间自适应平滑正则(w_field_smooth) 权重扫描。

用法（并行）：
    python scripts/run_hashgrid_smooth.py --gpu 0 --weights 0.5 2.0
    python scripts/run_hashgrid_smooth.py --gpu 2 --weights 8.0
"""
import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path("/media/F/sj/Code/GS-contrast-flow")
CONFIG = ROOT / "configs/dev_grid_views/hashgrid-adaptive-smooth/flow-mlp-stage1-hashgrid-smooth.yaml"
FIG_DIR = ROOT / "dev_grid_views/hashgrid-adaptive-smooth"
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


def train_and_eval(gpu: int, w: float):
    name = f"hs-smooth-{w}"
    out_dir = ROOT / "outputs" / "dev_grid_views" / "hashgrid-adaptive-smooth" / name
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    extra = ["--model.metric.init_args.w_field_smooth", str(w)]

    # train
    cmd = ["pixi", "run", "gs-fit", "--config", str(CONFIG),
           "--data.path", "data/example_grids",
           "-n", f"dev_grid_views/hashgrid-adaptive-smooth/{name}",
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
    args = p.parse_args()
    for w in args.weights:
        print(f"\n========== hashgrid w_field_smooth={w} (gpu {args.gpu}) ==========", flush=True)
        train_and_eval(args.gpu, w)


if __name__ == "__main__":
    main()
