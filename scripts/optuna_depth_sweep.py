"""Optuna 超参搜索：深度集中正则的 w_depth_conc / w_depth_scale / depth_scale_tau。

搜索空间（贝叶斯 TPE）：
    w_depth_conc     ∈ [0.1, 2.0]  (log)
    w_depth_scale    ∈ [0.5, 3.0]
    depth_scale_tau  ∈ [1.0, 4.0]

主目标（单目标合成，相对 spike 基线的等权相对改进，越大越好）：
    score = +(NSD1-0.936)/0.936 -(assd-0.69)/0.69 -(dist_avg_cl-1.396)/1.396
            +(pr_auc-0.665)/0.665 -(w1_len-9.83)/9.83
其他指标（dice/cldice/recall/n_jct/n_br/br_len/...）全部记录到 trial.user_attrs。

并发：n_jobs 个 trial 并行，GPU 按 `trial.number % n_gpus` 分配（一张卡可跑多个
worker，因为单实验显存 ~1.2GB / util ~20-30%）。

用法：
    pixi run python scripts/optuna_depth_sweep.py --n-trials 24 --n-jobs 8 --gpus 4
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import optuna

ROOT = Path("/media/F/sj/Code/GS-contrast-flow")
CONFIG = ROOT / "configs/dev_grid_views/hashgrid-adaptive-smooth/flow-mlp-stage1-hashgrid-depth.yaml"
GT = "data/example_grids/coronary_label.nii.gz"
OUT_ROOT = ROOT / "outputs" / "dev_grid_views" / "hashgrid-adaptive-smooth"

# spike 基线（相对改进归一化基准）
BASELINE = {"NSD1": 0.936, "assd": 0.69, "dist_avg_cl": 1.396,
            "pr_auc": 0.665, "w1_len": 9.83}
# 方向：+1 越大越好，-1 越小越好
DIR = {"NSD1": +1, "assd": -1, "dist_avg_cl": -1, "pr_auc": +1, "w1_len": -1}


def _n_gpus() -> int:
    # via env so it survives optuna's worker process spawning
    return int(os.environ.get("OPTUNA_N_GPUS", "4"))


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


def name_of(conc: float, scale: float, tau: float) -> str:
    return f"hs-depth-scale-{scale}-conc-{conc}-tau{tau}"


def train_eval_and_metrics(gpu: int, conc: float, scale: float, tau: float) -> dict | None:
    """一次完整实验：训练 + 导出 3D 体积 + 统一 shape/topology 指标，返回指标 dict。"""
    name = name_of(conc, scale, tau)
    out_dir = OUT_ROOT / name
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    extra = [
        "--model.metric.init_args.w_depth_conc", f"{conc}",
        "--model.metric.init_args.w_depth_scale", f"{scale}",
        "--model.metric.init_args.depth_scale_tau", f"{tau}",
    ]

    # 1. 训练
    cmd = ["pixi", "run", "gs-fit", "--config", str(CONFIG),
           "--data.path", "data/example_grids",
           "-n", f"dev_grid_views/hashgrid-adaptive-smooth/{name}",
           "--overwrite_output"] + extra
    if not run(cmd, env):
        return None

    ckpt = out_dir / "checkpoints" / "epoch=143-step=20000.ckpt"
    if not ckpt.exists():
        ckpts = sorted((out_dir / "checkpoints").glob("*.ckpt"))
        if not ckpts:
            print("!! 无 ckpt", flush=True)
            return None
        ckpt = ckpts[-1]

    # 2. 导出 3D 密度体积
    vol_dir = out_dir / "vol3d"
    vol_env = {"CUDA_VISIBLE_DEVICES": str(gpu), "VOL3D_OUT": str(vol_dir)}
    vol_cmd = ["pixi", "run", "python", "scripts/export_volume_3d_metrics.py", "validate",
               "--config", str(CONFIG), "--data.path", "data/example_grids",
               "--ckpt_path", str(ckpt)] + extra
    if not run(vol_cmd, vol_env):
        return None

    # 3. 统一 shape/topology 指标（JSON 输出）
    vol_path = vol_dir / "density_phase0.000.nii.gz"
    json_path = out_dir / "shape_metrics.json"
    shape_cmd = ["pixi", "run", "python", "scripts/compute_shape_metrics.py",
                 "--gt", GT,
                 "--vols", f"{name}:{vol_path}",
                 "--thr", "0.0344", "--tau", "1.0", "2.0",
                 "--json", str(json_path)]
    if not run(shape_cmd):
        return None

    data = json.loads(json_path.read_text())
    return data["results"]["0.0344"][0]


def objective(trial: optuna.Trial) -> float:
    conc = trial.suggest_float("w_depth_conc", 0.1, 2.0, log=True)
    scale = trial.suggest_float("w_depth_scale", 0.5, 3.0)
    tau = trial.suggest_float("depth_scale_tau", 1.0, 4.0)

    gpu = trial.number % _n_gpus()
    m = train_eval_and_metrics(gpu, conc, scale, tau)
    if m is None:
        return float("-inf")

    # 记录全部指标（供事后查看 / 手工挑选）
    for k, v in m.items():
        if v is not None:
            trial.set_user_attr(k, v)

    # 合成分数：相对 spike 基线的等权相对改进
    score = 0.0
    for key in BASELINE:
        v = m.get(key)
        if v is None:
            return float("-inf")
        score += DIR[key] * (v - BASELINE[key]) / BASELINE[key]
    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=24)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--study-name", type=str, default="depth-sweep")
    p.add_argument("--storage", type=str, default=None,
                   help="optuna storage URL (default: in-memory)")
    args = p.parse_args()
    os.environ["OPTUNA_N_GPUS"] = str(args.gpus)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    print("\n===== Optuna sweep 完成 =====")
    print(f"best score = {study.best_value:.4f}")
    print(f"best params = {study.best_params}")
    print("\nTop trials (按 score 降序):")
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("-inf"), reverse=True)
    for t in trials[:10]:
        ua = t.user_attrs
        print(f"  #{t.number} score={t.value:.4f} "
              f"conc={t.params['w_depth_conc']:.3f} scale={t.params['w_depth_scale']:.3f} tau={t.params['depth_scale_tau']:.3f} | "
              f"NSD1={ua.get('NSD1')} assd={ua.get('assd')} dist_cl={ua.get('dist_avg_cl')} "
              f"pr_auc={ua.get('pr_auc')} w1_len={ua.get('w1_len')}")


if __name__ == "__main__":
    main()
