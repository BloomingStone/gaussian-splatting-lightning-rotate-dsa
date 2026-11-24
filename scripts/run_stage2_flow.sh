#!/bin/bash
# stage2 (flow d_density(t)) 消融对比，initialize_from stage1 最优 (pper4)。
#   stage2-base   : 前 40 帧, 无平滑          (基线)
#   stage2-full   : 全帧(180), 无平滑         (改动1)
#   stage2-smooth : 前 40 帧, d_density 时空平滑 (改动2)
# 每个实验训练后跑 grid 2D 指标 (PSNR/SSIM)。
set -u
ROOT=/media/F/sj/Code/GS-contrast-flow
CONFIG=$ROOT/configs/gen_4d_output_all/flow-mlp-stage2-flow.yaml
STAGE1=$ROOT/outputs/dev_grid_views/hashgrid-adaptive-smooth/hs-depth-scale-1.851-conc-0.212-tau1.054-res128-rot0.3-pper4
DATA=data/example_grids
cd "$ROOT"

run_stage2() {
    local gpu=$1 name=$2 end_frame=$3 smooth=$4
    local extra=(--data.path "$DATA"
                 --model.initialize_from "$STAGE1"
                 --data.parser.init_args.spliter.init_args.end_frame "$end_frame")
    if [ "$smooth" = "1" ]; then
        extra+=(--model.renderer.init_args.deform_model_config.init_args.flow_temporal_smooth true
                --model.metric.init_args.w_flow_spatial_smooth 1.0
                --model.metric.init_args.w_flow_temporal_smooth 1.0)
    fi
    echo "==== $name 训练开始 $(date +%T) ===="
    CUDA_VISIBLE_DEVICES=$gpu pixi run gs-fit --config "$CONFIG" \
        -n "dev_grid_views/stage2/$name" --overwrite_output "${extra[@]}" \
        || { echo "!! 训练失败 $name"; return 1; }

    local out=$ROOT/outputs/dev_grid_views/stage2/$name
    local ckpt=$out/checkpoints/epoch=71-step=10000.ckpt
    [ -f "$ckpt" ] || ckpt=$(ls -t "$out"/checkpoints/*.ckpt | head -1)

    echo "==== $name grid 2D 指标 $(date +%T) ===="
    GRID_EVAL_OUT="$out/grid_eval" CUDA_VISIBLE_DEVICES=$gpu pixi run python \
        scripts/stage1_grid_eval.py validate --config "$CONFIG" \
        --data.path "$DATA" --ckpt_path "$ckpt" "${extra[@]}" \
        || { echo "!! grid eval 失败 $name"; return 1; }
    echo "==== $name 完成 $(date +%T) ===="
}

run_stage2 0 stage2-base 40 0 &
P1=$!
run_stage2 1 stage2-full 180 0 &
P2=$!
run_stage2 2 stage2-smooth 40 1 &
P3=$!

wait $P1 $P2 $P3
echo "==== ALL DONE $(date +%T) ===="
