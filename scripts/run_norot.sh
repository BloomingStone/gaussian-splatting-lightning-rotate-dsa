#!/bin/bash
# Ablation: best 参数 + max_resolution=256 + 禁用 d_rotation（形变场不输出旋转）
# 对比已有 hs-depth-scale-2.369-conc-0.110-tau1.630-maxres256（有旋转）
set -u
ROOT=/media/F/sj/Code/GS-contrast-flow
CONFIG=$ROOT/configs/dev_grid_views/hashgrid-adaptive-smooth/flow-mlp-stage1-hashgrid-depth.yaml
cd "$ROOT"

CONC=0.110
SCALE=2.369
TAU=1.630
NAME=hs-depth-scale-2.369-conc-0.110-tau1.630-maxres256-norot
COMMON=(--model.metric.init_args.w_depth_conc "$CONC"
        --model.metric.init_args.w_depth_scale "$SCALE"
        --model.metric.init_args.depth_scale_tau "$TAU")
EXTRA=(--model.renderer.init_args.deform_model_config.init_args.max_resolution 256
       --model.renderer.init_args.deform_model_config.init_args.enable_rotation false)

echo "==== $NAME 开始 $(date +%T) ===="
CUDA_VISIBLE_DEVICES=0 pixi run gs-fit --config "$CONFIG" \
    --data.path data/example_grids \
    -n "dev_grid_views/hashgrid-adaptive-smooth/$NAME" \
    --overwrite_output "${COMMON[@]}" "${EXTRA[@]}" || { echo "!! 训练失败"; exit 1; }

out=$ROOT/outputs/dev_grid_views/hashgrid-adaptive-smooth/$NAME
ckpt=$out/checkpoints/epoch=143-step=20000.ckpt
[ -f "$ckpt" ] || ckpt=$(ls -t "$out"/checkpoints/*.ckpt | head -1)

VOL3D_OUT="$out/vol3d" CUDA_VISIBLE_DEVICES=0 pixi run python \
    scripts/export_volume_3d_metrics.py validate --config "$CONFIG" \
    --data.path data/example_grids --ckpt_path "$ckpt" \
    "${COMMON[@]}" "${EXTRA[@]}" || { echo "!! 导出失败"; exit 1; }

vol=$out/vol3d/density_phase0.000.nii.gz
pixi run python scripts/compute_shape_metrics.py \
    --gt data/example_grids/coronary_label.nii.gz \
    --vols "$NAME:$vol" --thr 0.0344 --tau 1.0 2.0 \
    --json "$out/shape_metrics.json" || { echo "!! 评估失败"; exit 1; }
echo "==== $NAME 完成 $(date +%T) ===="
