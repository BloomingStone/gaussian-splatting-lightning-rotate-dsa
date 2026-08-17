#!/usr/bin/env bash
# Run visualization experiments for Diseased_17 (LCA and RCA)
# 5 methods × 2 branches = 10 experiments
# Runs all 10 experiments sequentially on available GPUs (round-robin)

set -e

BASE_DIR="/media/data3/sj/Code/GS-dev-contrast-flow"
DATA_DIR="data/gen_4d_output_all/flow"
OUTPUT="outputs/vis"
PROJECT="vis"

# Method configs and version prefixes
METHODS=(
  "vis_static:StaticGS"
  "vis_motion_t:DeformGS_t"
  "vis_motion_phi:DeformGS_phi"
  "vis_flow_combined:DeformGS_t_phi"
  "vis_flow_mlp:FlowGS"
)

BRANCHES=("LCA" "RCA")
GPUS=(0 1 2)  # Use GPUs 0,1,2 in round-robin

mkdir -p "$BASE_DIR/$OUTPUT"

pids=()
idx=0

for method_entry in "${METHODS[@]}"; do
  IFS=":" read -r config_name method_name <<< "$method_entry"
  
  for branch in "${BRANCHES[@]}"; do
    version="${method_name}_${branch}"
    data_path="${DATA_DIR}/asoca-diseased__Diseased_17__${branch}"
    config="${BASE_DIR}/configs/gen_4d_output_all/${config_name}.yaml"
    
    gpu_idx=${GPUS[$((idx % ${#GPUS[@]}))]}
    
    echo "[$(date '+%H:%M:%S')] Starting ${version} on GPU ${gpu_idx}..."
    
    CUDA_VISIBLE_DEVICES=$gpu_idx \
      pixi run gs-fit -- \
      --output "$BASE_DIR/$OUTPUT" \
      --config "$config" \
      --data.path "$data_path" \
      -n "vis" \
      -v "$version" \
      --logger wandb \
      --project "$PROJECT" \
      > "${BASE_DIR}/scripts/logs/${version}.log" 2>&1 &
    
    pids+=($!)
    idx=$((idx + 1))
    
    # Wait for GPUs to be populated before starting next batch
    if [ $((idx % ${#GPUS[@]})) -eq 0 ]; then
      echo "--- Batch complete, waiting for all to finish... ---"
      for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
      done
      pids=()
      echo "--- Batch finished at $(date '+%H:%M:%S') ---"
    fi
  done
done

# Wait for remaining processes
for pid in "${pids[@]}"; do
  wait "$pid" 2>/dev/null || true
done

echo ""
echo "============================================"
echo "All experiments completed at $(date)"
echo "============================================"
