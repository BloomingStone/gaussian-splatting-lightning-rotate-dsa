#!/bin/bash
# 手动重跑 sweep-v2 失败的 4 个 trial（CUDA OOM），每卡一个实验。
# 修复：PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 减少碎片。
set -u
ROOT=/media/F/sj/Code/GS-contrast-flow
CONFIG=$ROOT/configs/dev_grid_views/hashgrid-adaptive-smooth/flow-mlp-stage1-hashgrid-depth.yaml
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_exp() {
    local gpu=$1 conc=$2 scale=$3 tau=$4 res=$5 rot=$6
    local name="hs-depth-scale-${scale}-conc-${conc}-tau${tau}-res${res}-rot${rot}"
    local extra=(--model.metric.init_args.w_depth_conc "$conc"
                 --model.metric.init_args.w_depth_scale "$scale"
                 --model.metric.init_args.depth_scale_tau "$tau"
                 --model.renderer.init_args.deform_model_config.init_args.max_resolution "$res"
                 --model.renderer.init_args.deform_model_config.init_args.max_rotation "$rot")
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

# trial 1: res=256 rot=0.5
run_exp 0 0.389642 1.754977 2.121631 256 0.5 &
P1=$!
# trial 3: res=256 rot=0.3
run_exp 1 0.289193 1.956855 2.031551 256 0.3 &
P2=$!
# trial 5: res=256 rot=0.5
run_exp 2 0.078395 2.530903 1.913741 256 0.5 &
P3=$!
# trial 7: res=512 rot=0.1
run_exp 3 0.080110 2.797242 2.001691 512 0.1 &
P4=$!

wait $P1 $P2 $P3 $P4
echo "==== ALL DONE $(date +%T) ===="
