#!/bin/bash
# stage2 最终实验：
#   1. stage2-full-smooth (全帧 + d_density 时空平滑) 训练
#   2. 对 4 个实验重跑 grid 2D 指标，按归一化时间分组 (前 40 帧 vs 全部)
set -u
ROOT=/media/F/sj/Code/GS-contrast-flow
CONFIG=$ROOT/configs/gen_4d_output_all/flow-mlp-stage2-flow.yaml
STAGE1=$ROOT/outputs/dev_grid_views/hashgrid-adaptive-smooth/hs-depth-scale-1.851-conc-0.212-tau1.054-res128-rot0.3-pper4
DATA=data/example_grids
cd "$ROOT"
export GRID_EVAL_TIME_THRESHOLD=0.22    # frame 39 的归一化 time=0.2179, frame 40=0.2235
export GRID_EVAL_SKIP_TRAIN_FRAMES=1    # stage2 不渲染 train frame (越界)

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
    echo "==== $name grid eval 完成 $(date +%T) ===="
}

run_full_smooth() {
    local name=stage2-full-smooth
    local extra=(--data.path "$DATA" --model.initialize_from "$STAGE1"
                 --data.parser.init_args.spliter.init_args.end_frame 180
                 --model.renderer.init_args.deform_model_config.init_args.flow_temporal_smooth true
                 --model.metric.init_args.w_flow_spatial_smooth 1.0
                 --model.metric.init_args.w_flow_temporal_smooth 1.0)
    echo "==== $name 训练 $(date +%T) ===="
    CUDA_VISIBLE_DEVICES=0 pixi run gs-fit --config "$CONFIG" \
        -n "dev_grid_views/stage2/$name" --overwrite_output "${extra[@]}" \
        || { echo "!! 训练失败 $name"; return 1; }
    echo "==== $name 训练完成 $(date +%T) ===="
}

run_full_smooth &
P0=$!
grid_eval 1 stage2-base &
P1=$!
grid_eval 2 stage2-full &
P2=$!
grid_eval 3 stage2-smooth &
P3=$!

wait $P0 $P1 $P2 $P3
# full-smooth 训练完成后补跑它的 grid eval
grid_eval 0 stage2-full-smooth
echo "==== ALL DONE $(date +%T) ===="
