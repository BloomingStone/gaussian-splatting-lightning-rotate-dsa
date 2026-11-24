#!/bin/bash
# joint 消融实验：同时训练形变场(motion) + 造影剂流动场(flow)，从随机初始化。
# 对比两阶段 (stage2-full-smooth-nogate)。训练后跑 grid eval + 对比图。
set -u
ROOT=/media/F/sj/Code/GS-contrast-flow
CONFIG=$ROOT/configs/gen_4d_output_all/flow-mlp-joint.yaml
DATA=data/example_grids
cd "$ROOT"
export GRID_EVAL_TIME_THRESHOLD=0.22
export GRID_EVAL_SKIP_TRAIN_FRAMES=1

name=stage2-joint

echo "==== $name 训练 $(date +%T) ===="
CUDA_VISIBLE_DEVICES=0 pixi run gs-fit --config "$CONFIG" \
    -n "dev_grid_views/stage2/$name" --overwrite_output \
    --data.path "$DATA" || { echo "!! 训练失败"; exit 1; }

out=$ROOT/outputs/dev_grid_views/stage2/$name
ckpt=$(ls -t "$out"/checkpoints/*.ckpt | head -1)

echo "==== $name grid eval $(date +%T) ===="
GRID_EVAL_OUT="$out/grid_eval" CUDA_VISIBLE_DEVICES=0 pixi run python \
    scripts/stage1_grid_eval.py validate --config "$CONFIG" \
    --data.path "$DATA" --ckpt_path "$ckpt" \
    || { echo "!! grid eval 失败"; exit 1; }

echo "==== $name 对比图 $(date +%T) ===="
pixi run python scripts/make_stage2_grid_compare.py \
    --grid_eval_dir "$out/grid_eval" --exp_name "$name" --frames 20 100 \
    --out_dir "$ROOT/dev_grid_views/stage2-flow"

echo "==== $name 完成 $(date +%T) ===="
