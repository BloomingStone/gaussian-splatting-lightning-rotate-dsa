#!/bin/bash
# Ablation: 在 Optuna best (scale=2.369, conc=0.110, tau=1.630) 基础上，
# 对比形变场两个改进：
#   -maxres256 : 方案2 降低 max_resolution 2048 -> 256
#   -ctf       : 方案1 coarse-to-fine 分辨率调度 (groups 5,4,4; schedule 4000,12000)
set -u
ROOT=/media/F/sj/Code/GS-contrast-flow
CONFIG=$ROOT/configs/dev_grid_views/hashgrid-adaptive-smooth/flow-mlp-stage1-hashgrid-depth.yaml
cd "$ROOT"

CONC=0.110
SCALE=2.369
TAU=1.630
COMMON=(--model.metric.init_args.w_depth_conc "$CONC"
        --model.metric.init_args.w_depth_scale "$SCALE"
        --model.metric.init_args.depth_scale_tau "$TAU")

run_exp() {
    local gpu=$1 name=$2; shift 2
    local extra=("$@")
    echo "==== $name (gpu $gpu) 开始 $(date +%T) ===="
    CUDA_VISIBLE_DEVICES=$gpu pixi run gs-fit --config "$CONFIG" \
        --data.path data/example_grids \
        -n "dev_grid_views/hashgrid-adaptive-smooth/$name" \
        --overwrite_output "${COMMON[@]}" "${extra[@]}" || { echo "!! 训练失败 $name"; return 1; }

    local out=$ROOT/outputs/dev_grid_views/hashgrid-adaptive-smooth/$name
    local ckpt=$out/checkpoints/epoch=143-step=20000.ckpt
    [ -f "$ckpt" ] || ckpt=$(ls -t "$out"/checkpoints/*.ckpt | head -1)

    VOL3D_OUT="$out/vol3d" CUDA_VISIBLE_DEVICES=$gpu pixi run python \
        scripts/export_volume_3d_metrics.py validate --config "$CONFIG" \
        --data.path data/example_grids --ckpt_path "$ckpt" \
        "${COMMON[@]}" "${extra[@]}" || { echo "!! 导出失败 $name"; return 1; }

    local vol=$out/vol3d/density_phase0.000.nii.gz
    pixi run python scripts/compute_shape_metrics.py \
        --gt data/example_grids/coronary_label.nii.gz \
        --vols "$name:$vol" --thr 0.0344 --tau 1.0 2.0 \
        --json "$out/shape_metrics.json" || { echo "!! 评估失败 $name"; return 1; }
    echo "==== $name 完成 $(date +%T) ===="
}

run_exp 0 hs-depth-scale-2.369-conc-0.110-tau1.630-maxres256 \
    --model.renderer.init_args.deform_model_config.init_args.max_resolution 256 &
PID1=$!

run_exp 1 hs-depth-scale-2.369-conc-0.110-tau1.630-ctf \
    --model.renderer.init_args.deform_model_config.init_args.hashgrid_coarse_to_fine true &
PID2=$!

wait $PID1 $PID2
echo "==== ALL DONE $(date +%T) ===="
