"""hashgrid + 各向异性/重叠正则 权重扫描。

用法：
    python scripts/run_hashgrid_aniso.py --gpu 0 --aniso 0.01 0.05
    python scripts/run_hashgrid_aniso.py --gpu 2 --aniso 0.1 --overlap 0.01
"""
import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path("/media/F/sj/Code/GS-contrast-flow")
CONFIG = ROOT / "configs/dev_grid_views/hashgrid-adaptive-smooth/flow-mlp-stage1-hashgrid-aniso.yaml"
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


def train_and_eval(gpu: int, aniso: float, overlap: float):
    name = f"hs-aniso-{aniso}" + (f"-ov-{overlap}" if overlap > 0 else "")
    out_dir = ROOT / "outputs" / "dev_grid_views" / "hashgrid-adaptive-smooth" / name
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    extra = [
        "--model.metric.init_args.w_aniso", str(aniso),
        "--model.metric.init_args.w_overlap", str(overlap),
    ]

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

    cmd = ["pixi", "run", "python", "scripts/stage1_grid_eval.py", "validate",
           "--config", str(CONFIG), "--data.path", "data/example_grids",
           "--ckpt_path", str(ckpt)] + extra
    env["GRID_EVAL_OUT"] = str(out_dir)
    if not run(cmd, env):
        return

    cmd = ["pixi", "run", "python", "scripts/compute_artifact_metrics.py",
           "--stage1_dir", str(out_dir),
           "--r2g_dir", str(R2G_DIR),
           "--out_dir", str(out_dir)]
    if not run(cmd):
        return

    run(["pixi", "run", "python", "scripts/make_frame100_stage1.py",
         "--stage1_dir", str(out_dir),
         "--out_dir", str(FIG_DIR),
         "--exp_name", name])

    print(f"==== {name} 完成 ====", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--aniso", nargs="+", type=float, required=True)
    p.add_argument("--overlap", type=float, default=0.0)
    args = p.parse_args()
    for a in args.aniso:
        print(f"\n========== hashgrid w_aniso={a} w_overlap={args.overlap} (gpu {args.gpu}) ==========", flush=True)
        train_and_eval(args.gpu, a, args.overlap)


if __name__ == "__main__":
    main()
