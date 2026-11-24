#!/bin/bash
# 密度引导(flow gate)消融：关闭 w_flow_gate，其余保持默认(全帧 + 时空平滑)。
# 对比 stage2-full-smooth (有 gate) vs stage2-full-smooth-nogate (无 gate)。
set -u
ROOT=/media/F/sj/Code/GS-contrast-flow
CONFIG=$ROOT/configs/gen_4d_output_all/flow-mlp-stage2-flow.yaml
STAGE1=$ROOT/outputs/dev_grid_views/hashgrid-adaptive-smooth/hs-depth-scale-1.851-conc-0.212-tau1.054-res128-rot0.3-pper4
DATA=data/example_grids
cd "$ROOT"
export GRID_EVAL_TIME_THRESHOLD=0.22
export GRID_EVAL_SKIP_TRAIN_FRAMES=1

name=stage2-full-smooth-nogate
extra=(--data.path "$DATA" --model.initialize_from "$STAGE1"
       --data.parser.init_args.spliter.init_args.end_frame 180
       --model.renderer.init_args.deform_model_config.init_args.flow_temporal_smooth true
       --model.metric.init_args.w_flow_spatial_smooth 1.0
       --model.metric.init_args.w_flow_temporal_smooth 1.0
       --model.metric.init_args.w_flow_gate 0.0)

echo "==== $name 训练 $(date +%T) ===="
CUDA_VISIBLE_DEVICES=0 pixi run gs-fit --config "$CONFIG" \
    -n "dev_grid_views/stage2/$name" --overwrite_output "${extra[@]}" \
    || { echo "!! 训练失败"; exit 1; }

out=$ROOT/outputs/dev_grid_views/stage2/$name
ckpt=$(ls -t "$out"/checkpoints/*.ckpt | head -1)

echo "==== $name grid eval $(date +%T) ===="
GRID_EVAL_OUT="$out/grid_eval" CUDA_VISIBLE_DEVICES=0 pixi run python \
    scripts/stage1_grid_eval.py validate --config "$CONFIG" \
    --data.path "$DATA" --ckpt_path "$ckpt" \
    || { echo "!! grid eval 失败"; exit 1; }
echo "==== $name 完成 $(date +%T) ===="
