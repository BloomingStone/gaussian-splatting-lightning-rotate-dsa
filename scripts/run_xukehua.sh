#!/bin/bash
# xukehua 数据：motion-only 重建（frame 40~180），用当前最优参数（pper4）。
# 输出 density volume + 预测冠脉标签 到 outputs/xukehua/。
set -u
ROOT=/media/F/sj/Code/GS-contrast-flow
CONFIG=$ROOT/configs/dev_grid_views/hashgrid-adaptive-smooth/flow-mlp-stage1-hashgrid-depth.yaml
cd "$ROOT"

NAME=hs-stage1-motion-pper4
OUT_DIR=$ROOT/outputs/xukehua/$NAME
EXTRA=(--data.path data/xukehua
       --data.parser.init_args.spliter.init_args.end_frame 180
       --model.metric.init_args.w_depth_conc 0.212
       --model.metric.init_args.w_depth_scale 1.851
       --model.metric.init_args.depth_scale_tau 1.054
       --model.renderer.init_args.deform_model_config.init_args.max_resolution 128
       --model.renderer.init_args.deform_model_config.init_args.max_rotation 0.3
       --model.renderer.init_args.deform_model_config.init_args.phase_periodic true
       --model.renderer.init_args.deform_model_config.init_args.phase_multires 4)

echo "==== $NAME 训练开始 $(date +%T) ===="
CUDA_VISIBLE_DEVICES=0 pixi run gs-fit --config "$CONFIG" \
    -n "xukehua/$NAME" --overwrite_output "${EXTRA[@]}" \
    || { echo "!! 训练失败"; exit 1; }

ckpt=$OUT_DIR/checkpoints/epoch=143-step=20000.ckpt
[ -f "$ckpt" ] || ckpt=$(ls -t "$OUT_DIR"/checkpoints/*.ckpt | head -1)

echo "==== 导出 density volume + 冠脉标签 $(date +%T) ===="
VOL3D_OUT="$OUT_DIR/vol3d" CUDA_VISIBLE_DEVICES=0 pixi run python \
    scripts/export_volume_3d_metrics.py validate --config "$CONFIG" \
    --ckpt_path "$ckpt" "${EXTRA[@]}" || { echo "!! 导出失败"; exit 1; }

echo "==== shape 指标 $(date +%T) ===="
vol=$OUT_DIR/vol3d/density_phase0.000.nii.gz
pixi run python scripts/compute_shape_metrics.py \
    --gt data/xukehua/coronary_label.nii.gz \
    --vols "$NAME:$vol" --thr 0.0344 --tau 1.0 2.0 \
    --json "$OUT_DIR/shape_metrics.json" || { echo "!! 评估失败"; exit 1; }

echo "==== ALL DONE $(date +%T) ===="
echo "产物: $OUT_DIR"
