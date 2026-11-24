#!/bin/bash
# 用正确数据 data/example_grids (冠脉+背景) 重跑 stage2 两阶段消融。
#   stage2-base        : 前40帧, 无平滑, 有gate
#   stage2-full        : 全帧,   无平滑, 有gate
#   stage2-full-smooth : 全帧,   平滑,   有gate
#   stage2-full-smooth-nogate : 全帧, 平滑, 无gate
# 每个实验训练后跑 grid eval (冠脉加权+artifact) + 对比图。
set -u
ROOT=/media/F/sj/Code/GS-contrast-flow
CONFIG=$ROOT/configs/gen_4d_output_all/flow-mlp-stage2-flow.yaml
STAGE1=$ROOT/outputs/dev_grid_views/hashgrid-adaptive-smooth/hs-depth-scale-1.851-conc-0.212-tau1.054-res128-rot0.3-pper4
DATA=data/example_grids
cd "$ROOT"
export GRID_EVAL_TIME_THRESHOLD=0.22
export GRID_EVAL_SKIP_TRAIN_FRAMES=1

run_exp() {
    local gpu=$1 name=$2 end_frame=$3 smooth=$4 gate=$5
    local extra=(--data.path "$DATA" --model.initialize_from "$STAGE1"
                 --data.parser.init_args.spliter.init_args.end_frame "$end_frame")
    if [ "$smooth" = "1" ]; then
        extra+=(--model.renderer.init_args.deform_model_config.init_args.flow_temporal_smooth true
                --model.metric.init_args.w_flow_spatial_smooth 1.0
                --model.metric.init_args.w_flow_temporal_smooth 1.0)
    fi
    extra+=(--model.metric.init_args.w_flow_gate "$gate")

    echo "==== $name 训练 $(date +%T) ===="
    CUDA_VISIBLE_DEVICES=$gpu pixi run gs-fit --config "$CONFIG" \
        -n "dev_grid_views/stage2/$name" --overwrite_output "${extra[@]}" \
        || { echo "!! 训练失败 $name"; return 1; }

    local out=$ROOT/outputs/dev_grid_views/stage2/$name
    local ckpt=$(ls -t "$out"/checkpoints/*.ckpt | head -1)

    echo "==== $name grid eval $(date +%T) ===="
    GRID_EVAL_OUT="$out/grid_eval" CUDA_VISIBLE_DEVICES=$gpu pixi run python \
        scripts/stage1_grid_eval.py validate --config "$CONFIG" \
        --data.path "$DATA" --ckpt_path "$ckpt" \
        || { echo "!! grid eval 失败 $name"; return 1; }

    echo "==== $name 对比图 $(date +%T) ===="
    pixi run python scripts/make_stage2_grid_compare.py \
        --grid_eval_dir "$out/grid_eval" --exp_name "$name" --frames 20 100 \
        --out_dir "$ROOT/dev_grid_views/stage2-flow" || true
    echo "==== $name 完成 $(date +%T) ===="
}

run_exp 0 stage2-base 40 0 1.0 &
P1=$!
run_exp 1 stage2-full 180 0 1.0 &
P2=$!
run_exp 2 stage2-full-smooth 180 1 1.0 &
P3=$!
run_exp 3 stage2-full-smooth-nogate 180 1 0.0 &
P4=$!

wait $P1 $P2 $P3 $P4
echo "==== ALL DONE $(date +%T) ===="
