#!/bin/bash
# 对全部 stage2 实验重跑 grid eval（带冠脉区域加权 + artifact 指标）。
set -u
ROOT=/media/F/sj/Code/GS-contrast-flow
CONFIG=$ROOT/configs/gen_4d_output_all/flow-mlp-stage2-flow.yaml
DATA=data/example_only_flow_grids
cd "$ROOT"
export GRID_EVAL_TIME_THRESHOLD=0.22
export GRID_EVAL_SKIP_TRAIN_FRAMES=1

grid_eval() {
    local gpu=$1 name=$2
    local out=$ROOT/outputs/dev_grid_views/stage2/$name
    local ckpt=$(ls -t "$out"/checkpoints/*.ckpt 2>/dev/null | head -1)
    [ -z "$ckpt" ] && { echo "!! 无 ckpt $name"; return 1; }
    echo "==== $name grid eval $(date +%T) ===="
    GRID_EVAL_OUT="$out/grid_eval" CUDA_VISIBLE_DEVICES=$gpu pixi run python \
        scripts/stage1_grid_eval.py validate --config "$CONFIG" \
        --data.path "$DATA" --ckpt_path "$ckpt" \
        || { echo "!! grid eval 失败 $name"; return 1; }
    echo "==== $name 完成 $(date +%T) ===="
}

grid_eval 0 stage2-base &
P1=$!
grid_eval 1 stage2-full &
P2=$!
grid_eval 2 stage2-smooth &
P3=$!
grid_eval 3 stage2-full-smooth &
P4=$!

wait $P1 $P2 $P3 $P4
grid_eval 0 stage2-full-smooth-nogate
echo "==== ALL DONE $(date +%T) ===="
