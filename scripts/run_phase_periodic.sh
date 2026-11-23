#!/bin/bash
# phase 周期编码改进实验：先等 rerun_failed_trials 重跑完成，再启动。
# 基线对照 = trial 15 (res128+rot0.3, phase_multires=6 非周期)。
#   -pper3 : phase_periodic=true + phase_multires=3 (周期+低频)
#   -pper6 : phase_periodic=true + phase_multires=6 (仅周期，保持频率)
set -u
ROOT=/media/F/sj/Code/GS-contrast-flow
CONFIG=$ROOT/configs/dev_grid_views/hashgrid-adaptive-smooth/flow-mlp-stage1-hashgrid-depth.yaml
cd "$ROOT"

echo "$(date +%T) 等待 rerun_failed 重跑完成..."
while pgrep -f "rerun_failed_trials|gs-fit" > /dev/null 2>&1; do
    sleep 60
done
echo "$(date +%T) 重跑完成，启动 phase 周期编码实验"

CONC=0.212
SCALE=1.851
TAU=1.054
RES=128
ROT=0.3

run_exp() {
    local gpu=$1 pmr=$2
    local name="hs-depth-scale-${SCALE}-conc-${CONC}-tau${TAU}-res${RES}-rot${ROT}-pper${pmr}"
    local extra=(--model.metric.init_args.w_depth_conc "$CONC"
                 --model.metric.init_args.w_depth_scale "$SCALE"
                 --model.metric.init_args.depth_scale_tau "$TAU"
                 --model.renderer.init_args.deform_model_config.init_args.max_resolution "$RES"
                 --model.renderer.init_args.deform_model_config.init_args.max_rotation "$ROT"
                 --model.renderer.init_args.deform_model_config.init_args.phase_periodic true
                 --model.renderer.init_args.deform_model_config.init_args.phase_multires "$pmr")
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

    # 相位平滑性评估（复用 eval_phase_smoothness.py）
    CUDA_VISIBLE_DEVICES=$gpu pixi run python scripts/eval_phase_smoothness.py validate \
        --config "$CONFIG" --data.path data/example_grids --ckpt_path "$ckpt" \
        "${extra[@]}" 2>&1 | grep -A5 "相位平滑性" || true
    echo "==== $name 完成 $(date +%T) ===="
}

run_exp 0 3 &
P1=$!
run_exp 1 6 &
P2=$!

wait $P1 $P2
echo "==== ALL DONE $(date +%T) ===="
