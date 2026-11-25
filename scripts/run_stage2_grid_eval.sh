#!/usr/bin/env bash
# 对 5 个 stage2 实验重算 grid 指标（含 flow 保真度指标）。
# 用法: bash scripts/run_stage2_grid_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE2_CFG=configs/gen_4d_output_all/flow-mlp-stage2-flow.yaml
JOINT_CFG=configs/gen_4d_output_all/flow-mlp-joint.yaml
DATA=data/example_grids
OUT_ROOT=outputs/stage2_grid_eval
THR=${GRID_EVAL_TIME_THRESHOLD:-0.22}
mkdir -p "$OUT_ROOT"

run_one() {
  local name=$1 cfg=$2 ckpt=$3 gpu=$4
  local out="$OUT_ROOT/$name"
  echo "[eval] $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu \
    GRID_EVAL_OUT="$out" GRID_EVAL_TIME_THRESHOLD=$THR \
    pixi run python scripts/stage2_grid_eval.py validate \
      --config "$cfg" --data.path "$DATA" --ckpt_path "$ckpt" \
      > "$out.log" 2>&1 || echo "[eval] $name FAILED (see $out.log)"
}

run_one stage2-base         "$STAGE2_CFG" outputs/dev_grid_views/stage2/stage2-base/checkpoints/epoch=313-step=10000.ckpt 0 &
run_one stage2-full         "$STAGE2_CFG" outputs/dev_grid_views/stage2/stage2-full/checkpoints/epoch=70-step=10000.ckpt 1 &
run_one stage2-full-smooth  "$STAGE2_CFG" outputs/dev_grid_views/stage2/stage2-full-smooth/checkpoints/epoch=70-step=10000.ckpt 2 &
run_one stage2-full-smooth-nogate "$STAGE2_CFG" outputs/dev_grid_views/stage2/stage2-full-smooth-nogate/checkpoints/epoch=70-step=10000.ckpt 3 &
wait

run_one stage2-joint       "$JOINT_CFG"  outputs/dev_grid_views/stage2/stage2-joint/checkpoints/epoch=139-step=20000.ckpt 0
wait

echo "[eval] done"
