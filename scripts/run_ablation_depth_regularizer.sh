#!/bin/bash
# 深度正则 vs 形变场复杂度 贡献消融。
# 基准 = pper4 (conc=0.212, scale=1.851, tau=1.054, res=128, rot=0.3,
#               phase_periodic=true, phase_multires=4)。
#   abl-noscale   : 移除 depth_scale (scale=0)
#   abl-noconc    : 移除 depth_conc (conc=0)
#   abl-nodeform  : 移除平滑周期形变场 (恢复 config 默认 2048/非周期/pmr6)
set -u
ROOT=/media/F/sj/Code/GS-contrast-flow
CONFIG=$ROOT/configs/dev_grid_views/hashgrid-adaptive-smooth/flow-mlp-stage1-hashgrid-depth.yaml
cd "$ROOT"

SMOOTH_DEFORM=(--model.renderer.init_args.deform_model_config.init_args.max_resolution 128
               --model.renderer.init_args.deform_model_config.init_args.max_rotation 0.3
               --model.renderer.init_args.deform_model_config.init_args.phase_periodic true
               --model.renderer.init_args.deform_model_config.init_args.phase_multires 4)

run_exp() {
    local gpu=$1 name=$2 conc=$3 scale=$4 tau=$5 use_smooth=$6
    local extra=(--model.metric.init_args.w_depth_conc "$conc"
                 --model.metric.init_args.w_depth_scale "$scale"
                 --model.metric.init_args.depth_scale_tau "$tau")
    if [ "$use_smooth" = "1" ]; then
        extra+=("${SMOOTH_DEFORM[@]}")
    fi
    echo "==== $name (gpu $gpu) 开始 $(date +%T) ===="
    CUDA_VISIBLE_DEVICES=$gpu pixi run gs-fit --config "$CONFIG" \
        --data.path data/example_grids \
        -n "dev_grid_views/hashgrid-adaptive-smooth/$name" \
        --overwrite_output "${extra[@]}" || { echo "!! 训练失败 $name"; return 1; }

    local out=$ROOT/outputs/dev_grid_views/hashgrid-adaptive-smooth/$name
    local ckpt=$out/checkpoints/epoch=143-step=20000.ckpt
    [ -f "$ckpt" ] || ckpt=$(ls -t "$out"/checkpoints/*.ckpt | head -1)

    VOL3D_OUT="$out/vol3d" CUDA_VISIBLE_DEVICES=$gpu pixi run python \
        scripts/export_volume_3d_metrics.py validate --config "$CONFIG" \
        --data.path data/example_grids --ckpt_path "$ckpt" \
        "${extra[@]}" || { echo "!! 导出失败 $name"; return 1; }

    local vol=$out/vol3d/density_phase0.000.nii.gz
    pixi run python scripts/compute_shape_metrics.py \
        --gt data/example_grids/coronary_label.nii.gz \
        --vols "$name:$vol" --thr 0.0344 --tau 1.0 2.0 \
        --json "$out/shape_metrics.json" || { echo "!! 评估失败 $name"; return 1; }
    echo "==== $name 完成 $(date +%T) ===="
}

# 消融1: 无 depth_scale
run_exp 0 hs-abl-noscale 0.212 0.0 1.054 1 &
P1=$!
# 消融2: 无 depth_conc
run_exp 1 hs-abl-noconc 0.0 1.851 1.054 1 &
P2=$!
# 消融3: 无平滑周期形变场
run_exp 2 hs-abl-nodeform 0.212 1.851 1.054 0 &
P3=$!

wait $P1 $P2 $P3
echo "==== ALL DONE $(date +%T) ===="
